import httpx
import json
from typing import AsyncGenerator, List

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"


class LLMEngine:
    def __init__(self, model: str = MODEL_NAME, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    async def stream_response(self, messages: List[dict]) -> AsyncGenerator[str, None]:
        """Stream response tokens from Ollama one by one via SSE."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                    },
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if not data.get("done", False):
                                token = data.get("message", {}).get("content", "")
                                if token:
                                    yield token
                        except json.JSONDecodeError:
                            continue

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
