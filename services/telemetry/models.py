"""Normalized Telemetry & Token Models for Dual-Stream Benchmark."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StreamTelemetry(BaseModel):
    """Normalized token and performance metrics across local and cloud engines."""

    engine_name: str = Field(..., description="Engine identifier (e.g., 'ollama_qwen3', 'google_gemini_flash')")
    model: str = Field(..., description="Specific model tag (e.g., 'qwen3:1.7b', 'gemini-2.5-flash')")
    latency_ms: float = Field(..., description="End-to-end execution latency in milliseconds")
    input_tokens: int = Field(default=0, description="Prompt / context tokens ingested")
    output_tokens: int = Field(default=0, description="Completion / generation tokens produced")
    reasoning_tokens: int = Field(default=0, description="Extracted chain-of-thought tokens (e.g., <think>)")
    total_tokens: int = Field(default=0, description="Total tokens consumed (input + output)")
    tokens_per_sec: float = Field(default=0.0, description="Token generation speed")
    status: str = Field(default="success", description="'success', 'error', or 'missing_key'")
    error_message: Optional[str] = None


class GroundingAuditVerdict(BaseModel):
    """Automated comparison of technical grounding and policy adherence."""

    verified_doc_cited: bool = False
    doc_id: Optional[str] = None
    plan_code_cited: bool = False
    plan_code: Optional[str] = None
    apn_cited: bool = False
    regional_apn: Optional[str] = None
    sla_cited: bool = False
    sla: Optional[str] = None
    emergency_guardrail_triggered: bool = False
    summary: str = ""


class DualComparisonSummary(BaseModel):
    """Comparative delta between Stream A (Jev+Qwen RAG) and Stream B (Gemini Flash)."""

    latency_delta_ms: float = 0.0
    faster_stream: str = "stream_b"
    stream_a_grounded: bool = True
    stream_b_grounded: bool = False
    stream_1_grounded: bool = True
    stream_2_grounded: bool = True
    stream_3_grounded: bool = True
    stream_4_grounded: bool = True
    audit: GroundingAuditVerdict = Field(default_factory=GroundingAuditVerdict)
