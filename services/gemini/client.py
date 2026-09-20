"""Google AI Studio Gemini Client with Circuit Breaker and Token Normalization."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import httpx
from pydantic import BaseModel, Field

from services.telemetry.models import StreamTelemetry

# Auto-reload environment variables on import
load_dotenv(override=True)


class GeminiResponse(BaseModel):
    """Normalized response from Google AI Studio Gemini API."""

    content: str
    telemetry: StreamTelemetry
    wire_request: Optional[Dict[str, Any]] = None
    wire_response: Optional[Dict[str, Any]] = None
    timestamp_sent: str
    timestamp_received: str
    grounded: bool = False
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None


class GeminiClient:
    """Async & sync client for Google AI Studio (Gemini 3.8 Flash)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    ):
        load_dotenv(override=True)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.base_url = base_url

    async def generate_async(
        self,
        user_message: str,
        system_instruction: Optional[str] = None,
        kb_chunks: Optional[List[Dict[str, Any]]] = None,
        api_key_override: Optional[str] = None,
        model_override: Optional[str] = None,
        temperature: float = 0.6,
        timeout: float = 25.0,
        intent: Optional[str] = None,
        is_urgent: float = 0.0,
        frustration_score: float = 0.0,
        use_jev_intent_tag: bool = False,
        max_tokens: Optional[int] = None,
    ) -> GeminiResponse:
        """Generate content via Google AI Studio REST API asynchronously with circuit breaker."""
        load_dotenv(override=True)
        effective_key = api_key_override or self.api_key or os.getenv("GEMINI_API_KEY")
        effective_model = model_override or self.model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

        t_sent = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()

        # Circuit Breaker: Google Cloud Project ID passed instead of API Key
        if effective_key and effective_key.strip().startswith("gen-lang-client-"):
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            t_recv = datetime.now(timezone.utc).isoformat()
            proj_id = effective_key.strip()
            key_msg = (
                f"'{proj_id}' is a Google Cloud Project ID, not an API key. "
                "In Google AI Studio (aistudio.google.com/app/apikey), click the 'Key' column to copy your actual API key starting with 'AIzaSy...'."
            )
            return GeminiResponse(
                content=key_msg,
                telemetry=StreamTelemetry(
                    engine_name="google_gemini_flash_jev" if use_jev_intent_tag else "google_gemini_flash",
                    model=effective_model,
                    latency_ms=latency_ms,
                    input_tokens=0,
                    output_tokens=0,
                    reasoning_tokens=0,
                    total_tokens=0,
                    tokens_per_sec=0.0,
                    status="invalid_key",
                    error_message=key_msg,
                ),
                wire_request={
                    "endpoint": f"{self.base_url}/models/{effective_model}:generateContent",
                    "model": effective_model,
                    "timestamp": t_sent,
                    "status": "rejected_project_id_as_key",
                },
                wire_response={
                    "status_code": 400,
                    "timestamp": t_recv,
                    "error": key_msg,
                },
                timestamp_sent=t_sent,
                timestamp_received=t_recv,
            )

        # Circuit Breaker: Missing API Key
        if not effective_key:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            t_recv = datetime.now(timezone.utc).isoformat()
            missing_msg = (
                "Google AI Studio API Key is not configured. "
                "Please add your GEMINI_API_KEY to .env or paste it in the top navigation drawer."
            )
            return GeminiResponse(
                content=missing_msg,
                telemetry=StreamTelemetry(
                    engine_name="google_gemini_flash_jev" if use_jev_intent_tag else "google_gemini_flash",
                    model=effective_model,
                    latency_ms=latency_ms,
                    input_tokens=0,
                    output_tokens=0,
                    reasoning_tokens=0,
                    total_tokens=0,
                    tokens_per_sec=0.0,
                    status="missing_key",
                    error_message=missing_msg,
                ),
                wire_request={
                    "endpoint": f"{self.base_url}/models/{effective_model}:generateContent",
                    "model": effective_model,
                    "timestamp": t_sent,
                    "status": "bypassed_missing_key",
                },
                wire_response={
                    "status_code": 401,
                    "timestamp": t_recv,
                    "error": "Missing GEMINI_API_KEY",
                },
                timestamp_sent=t_sent,
                timestamp_received=t_recv,
            )

        endpoint = f"{self.base_url}/models/{effective_model}:generateContent?key={effective_key}"

        # Construct Google AI Studio payload with zero-thinking budget for lowest turn latency
        contents: List[Dict[str, Any]] = [
            {"role": "user", "parts": [{"text": user_message}]}
        ]
        output_tokens_limit = max_tokens or 512
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": output_tokens_limit,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        # Build grounded system instruction
        grounded = False
        parts: List[str] = []
        default_base = (
            "You are a helpful customer service assistant for Apex Telecom. "
            "Provide direct, concise, and professional answers formatted in clean Markdown with bullet points. "
            "CRITICAL: Output ONLY the final customer-facing response. "
            "Do NOT include internal reasoning or <think> tags. Speak directly to the customer."
        )
        parts.append(system_instruction or default_base)

        if use_jev_intent_tag and intent:
            urgency_label = "HIGH" if is_urgent >= 0.75 else "NORMAL"
            parts.append(f"\n[CLASSIFIED INTENT: {intent.upper()} | URGENCY: {urgency_label}]")

        if kb_chunks and len(kb_chunks) > 0:
            grounded = True
            context_blocks: List[str] = []
            for idx, chunk in enumerate(kb_chunks[:2], start=1):
                doc_id = chunk.get("doc_id", f"KB-DOC-{idx}")
                title = chunk.get("title", "Telco Policy Document")
                plan = chunk.get("plan_code", "N/A")
                apn = chunk.get("regional_apn", "N/A")
                sla = chunk.get("sla", "Standard SLA")
                raw_body = chunk.get("summary") or chunk.get("content", "")
                body = raw_body.strip().replace("\n", " ")[:160]
                context_blocks.append(
                    f"[{doc_id}: {title} | Plan: {plan} | APN: {apn} | SLA: {sla}]\nFact: {body}"
                )
            parts.append(
                "\n[GROUNDING KNOWLEDGE BASE FACTS]\n"
                "Incorporate these verified specifications when answering:\n"
                + "\n".join(context_blocks)
            )

        payload["systemInstruction"] = {
            "parts": [{"text": "\n".join(parts)}]
        }

        sanitized_endpoint = f"{self.base_url}/models/{effective_model}:generateContent?key=REDACTED"
        wire_req = {
            "endpoint": sanitized_endpoint,
            "method": "POST",
            "model": effective_model,
            "timestamp": t_sent,
            "payload": payload,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.post(endpoint, json=payload)
                latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                t_recv = datetime.now(timezone.utc).isoformat()

                # If thinkingConfig caused an error on older endpoints, fallback without it
                if res.status_code == 400 and "thinkingConfig" in res.text:
                    del payload["generationConfig"]["thinkingConfig"]
                    res = await client.post(endpoint, json=payload)
                    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                    t_recv = datetime.now(timezone.utc).isoformat()

                if res.status_code != 200:
                    err_body = res.text
                    try:
                        err_json = res.json()
                        err_body = err_json.get("error", {}).get("message", res.text)
                    except Exception:
                        pass

                    return GeminiResponse(
                        content=f"Google AI Studio Error (HTTP {res.status_code}): {err_body}",
                        telemetry=StreamTelemetry(
                            engine_name="google_gemini_flash",
                            model=effective_model,
                            latency_ms=latency_ms,
                            status="error",
                            error_message=f"HTTP {res.status_code}: {err_body}",
                        ),
                        wire_request=wire_req,
                        wire_response={
                            "status_code": res.status_code,
                            "timestamp": t_recv,
                            "error": err_body,
                        },
                        timestamp_sent=t_sent,
                        timestamp_received=t_recv,
                    )

                data = res.json()
                wire_resp = {
                    "status_code": 200,
                    "timestamp": t_recv,
                    "raw_body": data,
                }

                # Extract generated text
                candidates = data.get("candidates", [])
                text_content = ""
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text_content = "".join(p.get("text", "") for p in parts)

                # Extract token usage
                usage = data.get("usageMetadata", {})
                prompt_tokens = usage.get("promptTokenCount", 0)
                candidates_tokens = usage.get("candidatesTokenCount", 0)
                total_tokens = usage.get("totalTokenCount", prompt_tokens + candidates_tokens)

                tokens_per_sec = 0.0
                if latency_ms > 0 and candidates_tokens > 0:
                    tokens_per_sec = round(candidates_tokens / (latency_ms / 1000.0), 1)

                telemetry = StreamTelemetry(
                    engine_name="google_gemini_flash_jev" if use_jev_intent_tag else "google_gemini_flash",
                    model=effective_model,
                    latency_ms=latency_ms,
                    input_tokens=prompt_tokens,
                    output_tokens=candidates_tokens,
                    reasoning_tokens=0,
                    total_tokens=total_tokens,
                    tokens_per_sec=tokens_per_sec,
                    status="success",
                )

                return GeminiResponse(
                    content=text_content,
                    telemetry=telemetry,
                    wire_request=wire_req,
                    wire_response=wire_resp,
                    timestamp_sent=t_sent,
                    timestamp_received=t_recv,
                    grounded=grounded,
                    retrieved_chunks=kb_chunks,
                )

        except httpx.TimeoutException:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            t_recv = datetime.now(timezone.utc).isoformat()
            err_msg = f"Request to Google AI Studio timed out after {timeout}s. Stream A executed normally."
            return GeminiResponse(
                content=f"Stream B Timed Out: {err_msg}",
                telemetry=StreamTelemetry(
                    engine_name="google_gemini_flash",
                    model=effective_model,
                    latency_ms=latency_ms,
                    status="error",
                    error_message=err_msg,
                ),
                wire_request=wire_req,
                wire_response={
                    "status_code": 504,
                    "timestamp": t_recv,
                    "error": err_msg,
                },
                timestamp_sent=t_sent,
                timestamp_received=t_recv,
            )
        except Exception as e:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            t_recv = datetime.now(timezone.utc).isoformat()
            err_msg = f"Network or execution failure connecting to Google AI Studio: {type(e).__name__}: {e}"
            return GeminiResponse(
                content=f"Stream B Unavailable: {err_msg}",
                telemetry=StreamTelemetry(
                    engine_name="google_gemini_flash",
                    model=effective_model,
                    latency_ms=latency_ms,
                    status="error",
                    error_message=err_msg,
                ),
                wire_request=wire_req,
                wire_response={
                    "status_code": 503,
                    "timestamp": t_recv,
                    "error": str(e),
                },
                timestamp_sent=t_sent,
                timestamp_received=t_recv,
            )

    def generate_sync(
        self,
        user_message: str,
        system_instruction: Optional[str] = None,
        api_key_override: Optional[str] = None,
        model_override: Optional[str] = None,
        temperature: float = 0.6,
        timeout: float = 8.0,
    ) -> GeminiResponse:
        """Synchronous wrapper for generate_async using asyncio."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(
                        asyncio.run,
                        self.generate_async(
                            user_message=user_message,
                            system_instruction=system_instruction,
                            api_key_override=api_key_override,
                            model_override=model_override,
                            temperature=temperature,
                            timeout=timeout,
                        ),
                    ).result()
            else:
                return loop.run_until_complete(
                    self.generate_async(
                        user_message=user_message,
                        system_instruction=system_instruction,
                        api_key_override=api_key_override,
                        model_override=model_override,
                        temperature=temperature,
                        timeout=timeout,
                    )
                )
        except Exception:
            return asyncio.run(
                self.generate_async(
                    user_message=user_message,
                    system_instruction=system_instruction,
                    api_key_override=api_key_override,
                    model_override=model_override,
                    temperature=temperature,
                    timeout=timeout,
                )
            )
