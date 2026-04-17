from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from config import Settings


def _json_request(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} calling {url}: {exc.reason}\n"
            f"Response body:\n{response_body[:4000]}"
        ) from exc


def _binary_request(url: str, payload: bytes, headers: dict[str, str], timeout: int) -> dict[str, Any]:
    req = request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} calling {url}: {exc.reason}\n"
            f"Response body:\n{response_body[:4000]}"
        ) from exc


def _guess_mime_type(path: str | Path, default: str) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or default


def _read_b64(path: str | Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def _extract_text_from_gemini(response: dict[str, Any]) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {response}")
    parts = candidates[0].get("content", {}).get("parts") or []
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
    if not text_parts:
        raise RuntimeError(f"Gemini returned no text parts: {response}")
    return "\n".join(text_parts).strip()


def _extract_text_from_openai(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"Agent model returned no choices: {response}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        merged: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                merged.append(str(item.get("text", "")).strip())
        return "\n".join([part for part in merged if part]).strip()
    raise RuntimeError(f"Agent model returned unsupported message content: {response}")


def extract_json_block(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(cleaned[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        return json.loads(cleaned[first_bracket:last_bracket + 1])

    raise RuntimeError(f"Could not parse JSON from model output: {text[:500]}")


class GeminiClient:
    def __init__(self, model: str, api_key: str, timeout: int = 300, base_url: str = "") -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = base_url.strip()

    def _generate_url(self) -> str:
        if self.base_url:
            if "{model}" in self.base_url:
                return self.base_url.format(model=self.model)
            base = self.base_url.rstrip("/")
            if base.endswith(":generateContent"):
                return base
            return f"{base}/models/{self.model}:generateContent"
        encoded_model = parse.quote(self.model, safe="")
        return f"https://generativelanguage.googleapis.com/v1beta/models/{encoded_model}:generateContent?key={self.api_key}"

    def _upload_file(self, file_path: str | Path, mime_type: str) -> dict[str, Any]:
        if self.base_url:
            raise RuntimeError("Gemini file upload is only implemented for the default Google endpoint")

        start_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={self.api_key}"
        start_headers = {
            "Content-Type": "application/json",
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(Path(file_path).stat().st_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
        }
        start_payload = {"file": {"display_name": Path(file_path).name}}
        req = request.Request(
            start_url,
            data=json.dumps(start_payload).encode("utf-8"),
            headers=start_headers,
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            upload_url = response.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise RuntimeError("Gemini file upload did not return X-Goog-Upload-URL")

        upload_headers = {
            "Content-Type": mime_type,
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        }
        uploaded = _binary_request(upload_url, Path(file_path).read_bytes(), upload_headers, self.timeout)
        file_info = uploaded.get("file", uploaded)
        if not isinstance(file_info, dict) or "uri" not in file_info:
            raise RuntimeError(f"Gemini file upload response missing uri: {uploaded}")
        return file_info

    def generate(
        self,
        prompt: str,
        extra_parts: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
        response_mime_type: str = "application/json",
        temperature: float = 0.2,
    ) -> str:
        parts: list[dict[str, Any]] = [{"text": prompt}]
        if extra_parts:
            parts.extend(extra_parts)
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": response_mime_type,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        headers = {"Content-Type": "application/json"}
        if self.base_url:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return _extract_text_from_gemini(_json_request(self._generate_url(), payload, headers, self.timeout))

    def generate_from_images(
        self,
        prompt: str,
        image_paths: list[str | Path],
        system_instruction: str | None = None,
        response_mime_type: str = "application/json",
    ) -> str:
        parts: list[dict[str, Any]] = []
        for image_path in image_paths:
            mime_type = _guess_mime_type(image_path, "image/jpeg")
            parts.append({"inline_data": {"mime_type": mime_type, "data": _read_b64(image_path)}})
        return self.generate(
            prompt=prompt,
            extra_parts=parts,
            system_instruction=system_instruction,
            response_mime_type=response_mime_type,
        )

    def generate_from_file(
        self,
        prompt: str,
        file_path: str | Path,
        mime_type: str | None = None,
        system_instruction: str | None = None,
        response_mime_type: str = "application/json",
        inline_limit_bytes: int = 18 * 1024 * 1024,
    ) -> str:
        resolved_mime = mime_type or _guess_mime_type(file_path, "application/octet-stream")
        size = Path(file_path).stat().st_size
        if size <= inline_limit_bytes:
            parts = [{"inline_data": {"mime_type": resolved_mime, "data": _read_b64(file_path)}}]
        else:
            file_info = self._upload_file(file_path, resolved_mime)
            parts = [{"file_data": {"mime_type": resolved_mime, "file_uri": file_info["uri"]}}]
        return self.generate(
            prompt=prompt,
            extra_parts=parts,
            system_instruction=system_instruction,
            response_mime_type=response_mime_type,
        )


class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, timeout: int = 300, base_url: str = "") -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2, max_tokens: int | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            token_key = "max_completion_tokens" if self.model.startswith(("gpt-5", "o1", "o3", "o4")) else "max_tokens"
            payload[token_key] = max_tokens
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        return _extract_text_from_openai(_json_request(url, payload, headers, self.timeout))


def build_clients(settings: Settings) -> tuple[GeminiClient, GeminiClient, OpenAICompatibleClient]:
    video_client = GeminiClient(
        model=settings.video_model,
        api_key=settings.video_api_key,
        base_url=settings.video_base_url,
        timeout=settings.request_timeout,
    )
    audio_client = GeminiClient(
        model=settings.audio_model,
        api_key=settings.audio_api_key,
        base_url=settings.audio_base_url,
        timeout=settings.request_timeout,
    )
    agent_client = OpenAICompatibleClient(
        model=settings.agent_model,
        api_key=settings.agent_api_key,
        base_url=settings.agent_base_url,
        timeout=settings.request_timeout,
    )
    return video_client, audio_client, agent_client
