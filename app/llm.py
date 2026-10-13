import httpx
from app.config import settings


async def ollama_chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    payload: dict = {"model": settings.ollama_model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    url = settings.ollama_base_url.rstrip("/") + "/api/chat"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()["message"]
    except httpx.ConnectError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Start Ollama and run `ollama pull llama3.2`."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}") from exc
