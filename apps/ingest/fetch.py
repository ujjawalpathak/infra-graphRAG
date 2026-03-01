import os, yaml, requests
from bs4 import BeautifulSoup
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # remove nav/footer-ish elements
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text("\n")
    # basic cleanup
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)

def main():
    cfg = yaml.safe_load(open("apps/ingest/sources.yaml"))
    for src in cfg["sources"]:
        for url in src["urls"]:
            print("fetch:", url)
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            text = html_to_text(r.text)

            safe_name = url.replace("https://", "").replace("/", "_")
            out = RAW_DIR / f"{src['name']}__{safe_name}.txt"
            out.write_text(text, encoding="utf-8")
            print("saved:", out)

if __name__ == "__main__":
    main()