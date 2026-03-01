from pathlib import Path
import json, hashlib
import re

RAW_DIR = Path("data/raw")
OUT = Path("data/processed/chunks.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def split_by_headings(text: str):
    """
    Split text into sections based on heading-like patterns.
    Works for markdown-style docs.
    """
    # Split at lines that look like headings
    sections = re.split(r"\n(?=[A-Z][^\n]{0,80}\n)", text)
    return [s.strip() for s in sections if len(s.strip()) > 200]

def smart_chunk(text: str, max_chars: int = 1200):
    sections = split_by_headings(text)
    chunks = []

    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            # Fallback to sub-chunking within section
            i = 0
            while i < len(section):
                chunk = section[i:i+max_chars]
                chunks.append(chunk)
                i += max_chars - 200

    return chunks

def make_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def main():
    with OUT.open("w", encoding="utf-8") as f:
        for p in RAW_DIR.glob("*.txt"):
            text = p.read_text(encoding="utf-8")
            for idx, ch in enumerate(smart_chunk(text)):
                rec = {
                    "chunk_id": make_id(f"{p.name}:{idx}:{ch[:50]}"),
                    "source_file": p.name,
                    "chunk_index": idx,
                    "text": ch,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("wrote:", OUT)

if __name__ == "__main__":
    main()
