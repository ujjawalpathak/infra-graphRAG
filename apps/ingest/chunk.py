from pathlib import Path
import json, hashlib

RAW_DIR = Path("data/raw")
OUT = Path("data/processed/chunks.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200):
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+max_chars]
        chunks.append(chunk)
        i += max_chars - overlap
    return chunks

def make_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def main():
    with OUT.open("w", encoding="utf-8") as f:
        for p in RAW_DIR.glob("*.txt"):
            text = p.read_text(encoding="utf-8")
            for idx, ch in enumerate(chunk_text(text)):
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
