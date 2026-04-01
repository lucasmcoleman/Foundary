#!/usr/bin/env python3
"""
Generate training data from ebooks using local LMStudio models.

Reads PDFs/EPUBs, chunks them into coherent sections, and uses a local
LMStudio model to generate Q&A training pairs grounded in each chunk's
specific content. Every fact in the source material gets covered.

Usage:
    # From ebooks directory (PDF/EPUB files)
    python gardener/generate_from_books.py /path/to/ebooks/

    # From already-extracted text files
    python gardener/generate_from_books.py /path/to/texts/ --already-extracted

    # Custom output and model
    python gardener/generate_from_books.py /path/to/ebooks/ \
        --output gardener/book_training_data.jsonl \
        --model hermes-4.3-36b \
        --qa-per-chunk 3

    # Dry run (show chunks, don't generate)
    python gardener/generate_from_books.py /path/to/ebooks/ --dry-run

    # Merge with existing gardener training data
    python gardener/generate_from_books.py /path/to/ebooks/ --merge-into data/zeroclaw_training_data.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, str(Path(__file__).parent))
from system_prompt import SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BookGenConfig:
    input_dir: str = ""
    output_path: str = "gardener/book_training_data.jsonl"
    lmstudio_url: str = "http://localhost:1234"
    model: str = ""                    # auto-detect from LMStudio if empty
    chunk_size: int = 1500             # target words per chunk
    chunk_overlap: int = 100           # overlap words between chunks
    qa_per_chunk: int = 3              # Q&A pairs to generate per chunk
    max_concurrent: int = 3            # parallel API calls
    temperature: float = 0.7
    max_tokens: int = 4096
    already_extracted: bool = False    # input is .txt files, skip extraction
    merge_into: str = ""               # append to existing JSONL file


# ---------------------------------------------------------------------------
# Text extraction (reuses extract_references.py logic)
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: Path) -> tuple[str, str]:
    """Extract text from PDF. Returns (text, title)."""
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    parts = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            parts.append(text)
    doc.close()
    return "\n\n".join(parts), pdf_path.stem


def extract_epub(epub_path: Path) -> tuple[str, str]:
    """Extract text from EPUB. Returns (text, title)."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})

    # Try to get the real title
    title = pdf_title = epub_path.stem
    dc_title = book.get_metadata("DC", "title")
    if dc_title:
        title = dc_title[0][0]

    parts = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n")
            if text.strip():
                parts.append(text)
    return "\n\n".join(parts), title


def extract_all(input_dir: Path, already_extracted: bool) -> list[tuple[str, str]]:
    """Extract text from all supported files. Returns [(text, title), ...]."""
    books = []

    if already_extracted:
        for f in sorted(input_dir.glob("*.txt")):
            text = f.read_text(errors="replace")
            if text.strip():
                books.append((text, f.stem))
                print(f"  Loaded: {f.name} ({len(text.split()):,} words)")
        return books

    extractors = {".pdf": extract_pdf, ".epub": extract_epub}
    files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in extractors
    )

    if not files:
        print(f"No PDF/EPUB files found in {input_dir}")
        sys.exit(1)

    for f in files:
        print(f"  Extracting: {f.name}...", end=" ", flush=True)
        try:
            text, title = extractors[f.suffix.lower()](f)
            if text.strip():
                word_count = len(text.split())
                books.append((text, title))
                print(f"done ({word_count:,} words)")
            else:
                print("empty (no text extracted)")
        except Exception as e:
            print(f"error ({e})")

    return books


# ---------------------------------------------------------------------------
# Smart chunking — splits on headings/paragraphs, not mid-sentence
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 100) -> list[str]:
    """Split text into chunks, preferring breaks at headings and paragraph boundaries."""
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If a single paragraph exceeds chunk_size, split it by sentences
        if para_words > chunk_size:
            if current:
                chunks.append('\n\n'.join(current))
                current = []
                current_words = 0

            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_chunk = []
            sent_words = 0
            for sent in sentences:
                sw = len(sent.split())
                if sent_words + sw > chunk_size and sent_chunk:
                    chunks.append(' '.join(sent_chunk))
                    # Keep overlap
                    keep = []
                    keep_words = 0
                    for s in reversed(sent_chunk):
                        w = len(s.split())
                        if keep_words + w > overlap:
                            break
                        keep.insert(0, s)
                        keep_words += w
                    sent_chunk = keep
                    sent_words = keep_words
                sent_chunk.append(sent)
                sent_words += sw
            if sent_chunk:
                chunks.append(' '.join(sent_chunk))
            continue

        # Would this paragraph push us over the limit?
        if current_words + para_words > chunk_size and current:
            chunks.append('\n\n'.join(current))
            # Keep overlap from end of current chunk
            keep = []
            keep_words = 0
            for p in reversed(current):
                w = len(p.split())
                if keep_words + w > overlap:
                    break
                keep.insert(0, p)
                keep_words += w
            current = keep
            current_words = keep_words

        current.append(para)
        current_words += para_words

    if current:
        chunks.append('\n\n'.join(current))

    # Filter out chunks that are too short to be useful
    chunks = [c for c in chunks if len(c.split()) >= 50]

    return chunks


# ---------------------------------------------------------------------------
# LMStudio API
# ---------------------------------------------------------------------------

def call_lmstudio(
    prompt: str,
    system: str,
    config: BookGenConfig,
) -> str | None:
    """Call LMStudio's OpenAI-compatible API."""
    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }).encode()

    req = urllib.request.Request(
        f"{config.lmstudio_url}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        print(f"  LMStudio error: {e}")
        return None
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print(f"  Response parse error: {e}")
        return None


def detect_model(config: BookGenConfig) -> str:
    """Auto-detect the best available model from LMStudio."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{config.lmstudio_url}/v1/models", timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception:
        print("Error: cannot connect to LMStudio at", config.lmstudio_url)
        sys.exit(1)

    models = [m["id"] for m in data.get("data", [])]
    if not models:
        print("Error: no models loaded in LMStudio")
        sys.exit(1)

    # Prefer larger instruct/chat models, skip embedding models
    candidates = [m for m in models if "embed" not in m.lower() and "ocr" not in m.lower()]
    if not candidates:
        candidates = models

    # Prefer models with more parameters (rough heuristic from name)
    def score(name):
        n = name.lower()
        for size in ["122b", "120b", "36b", "30b", "27b", "20b", "14b", "9b", "8b", "4b", "3b"]:
            if size in n:
                return int(size.replace("b", ""))
        return 1

    candidates.sort(key=score, reverse=True)
    return candidates[0]


# ---------------------------------------------------------------------------
# Q&A generation from a chunk
# ---------------------------------------------------------------------------

TEACHER_SYSTEM = """You are a training data generator. Given a passage from a gardening book, generate realistic Q&A conversations that teach the specific knowledge in the passage.

RULES:
1. Every key fact, technique, or recommendation in the passage MUST appear in at least one Q&A pair.
2. Questions should be natural — how a real gardener would ask. Vary styles: "how do I...", "what's the best...", "my X is doing Y, what's wrong?", "when should I..."
3. Answers must be grounded in the passage content but written in a warm, knowledgeable conversational tone.
4. Include specific numbers, varieties, timing, and measurements from the passage.
5. If the passage mentions a region or climate, adapt to North Carolina (zones 6b-8a) where applicable.
6. Generate both simple single-exchange Q&As and occasional multi-turn conversations where the user asks a follow-up.

OUTPUT: A JSON array of conversation objects. Each object has a "messages" array with user/assistant message pairs. No markdown fences. No text outside the JSON."""


def build_chunk_prompt(chunk: str, book_title: str, n_qa: int) -> str:
    return f"""Generate {n_qa} Q&A training conversations from this passage.

Book: {book_title}
---
{chunk}
---

Output a JSON array of {n_qa} objects, each with a "messages" array of user/assistant pairs. Make at least one multi-turn (2+ exchanges). Ground every answer in the passage content."""


def parse_qa_response(response: str) -> list[dict]:
    """Parse LMStudio response into training examples."""
    if not response:
        return []

    text = response.strip()

    # Strip markdown fences
    if text.startswith("```"):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
        text = text.strip()

    # Try parsing as JSON array
    for attempt in [text, text[text.find('['):text.rfind(']') + 1] if '[' in text else '']:
        if not attempt:
            continue
        try:
            data = json.loads(attempt)
            if isinstance(data, list):
                return _validate_examples(data)
        except json.JSONDecodeError:
            continue

    # Last resort: try to find individual JSON objects
    results = []
    for match in re.finditer(r'\{[^{}]*"messages"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL):
        try:
            obj = json.loads(match.group())
            results.extend(_validate_examples([obj]))
        except json.JSONDecodeError:
            continue

    return results


def _validate_examples(items: list) -> list[dict]:
    """Validate and clean parsed examples, prepend system prompt."""
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        msgs = item.get("messages", [])
        if not isinstance(msgs, list) or len(msgs) < 2:
            continue

        # Validate message structure
        valid = True
        for m in msgs:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                valid = False
                break
            if m["role"] not in ("user", "assistant", "system"):
                valid = False
                break
            if not m["content"] or not m["content"].strip():
                valid = False
                break
        if not valid:
            continue

        # Remove any system messages the model included
        msgs = [m for m in msgs if m["role"] != "system"]
        if len(msgs) < 2:
            continue
        if msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
            continue

        # Prepend system prompt
        full = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
        results.append({"messages": full})

    return results


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_from_chunks(
    chunks: list[tuple[str, str]],  # (chunk_text, book_title)
    config: BookGenConfig,
) -> int:
    """Generate training data from all chunks. Returns count of examples generated."""
    total_chunks = len(chunks)
    completed = 0
    generated = 0
    failed = 0
    start_time = time.time()
    write_lock = threading.Lock()

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def process_chunk(idx, chunk_text, book_title):
        prompt = build_chunk_prompt(chunk_text, book_title, config.qa_per_chunk)
        response = call_lmstudio(prompt, TEACHER_SYSTEM, config)
        examples = parse_qa_response(response)
        return idx, examples

    with open(output_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=config.max_concurrent) as executor:
            futures = {
                executor.submit(process_chunk, i, chunk, title): i
                for i, (chunk, title) in enumerate(chunks)
            }

            for future in as_completed(futures):
                idx, examples = future.result()
                completed += 1

                if examples:
                    with write_lock:
                        for ex in examples:
                            out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        out_f.flush()
                    generated += len(examples)
                else:
                    failed += 1

                elapsed = time.time() - start_time
                rate = generated / elapsed if elapsed > 0 else 0
                eta = (total_chunks - completed) / (completed / elapsed) if completed > 0 else 0

                print(
                    f"  [{completed}/{total_chunks}] "
                    f"{generated} examples | "
                    f"{rate:.1f} ex/sec | "
                    f"ETA: {eta / 60:.0f}min | "
                    f"{failed} failed",
                    flush=True,
                )

    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate gardener training data from ebooks via LMStudio"
    )
    parser.add_argument("input_dir", help="Directory with PDF/EPUB files (or .txt if --already-extracted)")
    parser.add_argument("--output", default="gardener/book_training_data.jsonl", help="Output JSONL path")
    parser.add_argument("--model", default="", help="LMStudio model ID (auto-detect if empty)")
    parser.add_argument("--lmstudio-url", default="http://localhost:1234", help="LMStudio API URL")
    parser.add_argument("--chunk-size", type=int, default=1500, help="Target words per chunk")
    parser.add_argument("--qa-per-chunk", type=int, default=3, help="Q&A pairs per chunk")
    parser.add_argument("--concurrent", type=int, default=3, help="Parallel API calls")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--already-extracted", action="store_true", help="Input is .txt files, skip PDF/EPUB extraction")
    parser.add_argument("--merge-into", default="", help="Append output to this existing JSONL file")
    parser.add_argument("--dry-run", action="store_true", help="Show chunks without generating")
    args = parser.parse_args()

    config = BookGenConfig(
        input_dir=args.input_dir,
        output_path=args.output,
        model=args.model,
        lmstudio_url=args.lmstudio_url,
        chunk_size=args.chunk_size,
        qa_per_chunk=args.qa_per_chunk,
        max_concurrent=args.concurrent,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        already_extracted=args.already_extracted,
    )

    if args.merge_into:
        config.merge_into = args.merge_into

    input_dir = Path(config.input_dir)
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        sys.exit(1)

    # --- Extract ---
    print(f"Reading books from {input_dir}/")
    books = extract_all(input_dir, config.already_extracted)
    if not books:
        print("No books found.")
        sys.exit(1)

    print(f"\n{len(books)} books loaded")

    # --- Chunk ---
    print(f"Chunking (target ~{config.chunk_size} words per chunk)...")
    all_chunks = []  # (chunk_text, book_title)
    for text, title in books:
        chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)
        all_chunks.extend((c, title) for c in chunks)
        print(f"  {title}: {len(chunks)} chunks")

    print(f"\n{len(all_chunks)} total chunks")
    expected = len(all_chunks) * config.qa_per_chunk
    print(f"Will generate ~{expected} training examples ({config.qa_per_chunk} per chunk)\n")

    # --- Dry run ---
    if args.dry_run:
        for i, (chunk, title) in enumerate(all_chunks[:10]):
            words = len(chunk.split())
            preview = chunk[:200].replace('\n', ' ')
            print(f"  [{i+1}] ({title}, {words} words) {preview}...")
        if len(all_chunks) > 10:
            print(f"  ... and {len(all_chunks) - 10} more chunks")
        return

    # --- Connect to LMStudio ---
    if not config.model:
        config.model = detect_model(config)
    print(f"Model: {config.model}")
    print(f"Concurrent: {config.max_concurrent}")
    print(f"Output: {config.output_path}\n")

    # Ensure output file exists
    Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(config.output_path).exists():
        Path(config.output_path).touch()

    # --- Generate ---
    count = generate_from_chunks(all_chunks, config)

    # --- Stats ---
    examples = []
    with open(config.output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    total = len(examples)
    multi = sum(1 for ex in examples if sum(1 for m in ex["messages"] if m["role"] == "assistant") > 1)
    size_kb = Path(config.output_path).stat().st_size / 1024

    print(f"\nDone! Generated {count} examples from {len(books)} books")
    print(f"  Total examples: {total}")
    if total:
        print(f"  Multi-turn: {multi} ({100 * multi / total:.0f}%)")
        print(f"  Single-turn: {total - multi}")
    print(f"  File size: {size_kb:.0f} KB")
    print(f"  Output: {config.output_path}")

    # --- Merge ---
    if config.merge_into:
        merge_path = Path(config.merge_into)
        existing = 0
        if merge_path.exists():
            with open(merge_path) as f:
                existing = sum(1 for line in f if line.strip())

        with open(merge_path, "a") as dest:
            with open(config.output_path) as src:
                for line in src:
                    if line.strip():
                        dest.write(line)

        new_total = existing + total
        print(f"\nMerged into {config.merge_into}: {existing} existing + {total} new = {new_total} total")


if __name__ == "__main__":
    main()
