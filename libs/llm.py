import os
import httpx

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


async def generate(prompt: str) -> str:
    if LLM_PROVIDER != "ollama":
        raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER}")

    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()

            if "response" not in data:
                raise RuntimeError(f"Unexpected Ollama response: {data}")

            return data["response"].strip()

    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to connect to Ollama at {OLLAMA_BASE_URL}: {e}")