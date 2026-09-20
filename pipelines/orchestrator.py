"""Central Orchestration Router for Telco AI Intent Routing & RAG Workbench.

Flow:
User Turn -> Jev System One Classifier -> Deterministic Guardrails
          -> LanceDB Hybrid Retriever (if KB domain)
          -> Qwen 3 Generator (Dynamic Grounding & Safety Guardrails)
          -> Handover Context Card Generation (if Escalated)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import re
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from rag.retriever import TelcoRetriever
from services.gemini.client import GeminiClient, GeminiResponse
from services.jev.classifier import IntentHierarchy, JevClassifier, JevResult
from services.telemetry.models import DualComparisonSummary, GroundingAuditVerdict, StreamTelemetry
from services.vllm.generator import GenerationResult, QwenGenerator


class HandoverContextCard(BaseModel):
    handover_id: str
    timestamp: float
    trigger_reason: str
    dispatch_queue: str
    customer_state: Dict[str, Any]
    metrics: Dict[str, float]
    recommended_actions: List[str]
    grounding_references: List[Dict[str, Any]]


class OrchestrationResult(BaseModel):
    user_message: str
    jev_result: JevResult
    hierarchy: IntentHierarchy
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    generation_result: GenerationResult
    handover_card: Optional[HandoverContextCard] = None
    latency_breakdown: Dict[str, float] = Field(default_factory=dict)
    wire_request: Optional[Dict[str, Any]] = None
    wire_response: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_latency_ms: float


class DualStreamResult(BaseModel):
    user_message: str
    stream_a: OrchestrationResult
    stream_b: GeminiResponse
    stream_raw: Optional[GenerationResult] = None
    stream_1: Optional[OrchestrationResult] = None
    stream_2: Optional[GenerationResult] = None
    stream_3: Optional[GeminiResponse] = None
    stream_4: Optional[GeminiResponse] = None
    comparison: DualComparisonSummary
    total_turn_latency_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TriStreamResult(BaseModel):
    user_message: str
    stream_1: OrchestrationResult = Field(..., description="[1] Jev + Qwen Grounded (Local with Jev Intent & Urgency)")
    stream_2: GenerationResult = Field(..., description="[2] Control Qwen Grounded (Local Baseline)")
    stream_3: GeminiResponse = Field(..., description="[3] Jev + Gemini Flash Grounded (Frontier with Jev Intent & Urgency)")
    stream_4: Optional[GeminiResponse] = Field(None, description="[4] Control Gemini Flash Grounded (Frontier Baseline)")
    stream_a: Optional[OrchestrationResult] = None
    stream_raw: Optional[GenerationResult] = None
    stream_b: Optional[GeminiResponse] = None
    comparison: DualComparisonSummary
    total_turn_latency_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class QuadStreamResult(BaseModel):
    user_message: str
    stream_1: OrchestrationResult = Field(..., description="[1] Jev + Qwen Grounded (Local with Jev Intent & Urgency)")
    stream_2: GenerationResult = Field(..., description="[2] Control Qwen Grounded (Local Baseline)")
    stream_3: GeminiResponse = Field(..., description="[3] Jev + Gemini Flash Grounded (Frontier with Jev Intent & Urgency)")
    stream_4: GeminiResponse = Field(..., description="[4] Control Gemini Flash Grounded (Frontier Baseline)")
    stream_a: Optional[OrchestrationResult] = None
    stream_raw: Optional[GenerationResult] = None
    stream_b: Optional[GeminiResponse] = None
    comparison: DualComparisonSummary
    total_turn_latency_ms: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PipelineOrchestrator:
    """Coordinates end-to-end intent classification, RAG retrieval, and response generation."""

    def __init__(
        self,
        classifier: Optional[JevClassifier] = None,
        retriever: Optional[TelcoRetriever] = None,
        generator: Optional[QwenGenerator] = None,
        gemini: Optional[GeminiClient] = None,
    ):
        self.classifier = classifier or JevClassifier()
        self.retriever = retriever or TelcoRetriever()
        self.generator = generator or QwenGenerator()
        self.gemini = gemini or GeminiClient()

    def process_turn(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        account_metadata: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """Process a single turn through the full Telco AI workbench pipeline."""
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Step 1: Jev System One Classification & Guardrail Evaluation
        jev_res = self.classifier.classify(user_message)
        jev_ms = jev_res.latency_ms

        # Step 2: Knowledge Base Retrieval (skip on human escalation or extreme frustration)
        kb_chunks: List[Dict[str, Any]] = []
        rag_start = time.perf_counter()
        if (
            jev_res.selected_choice not in ["human_escalation", "other_or_unclear"]
            and jev_res.frustration_score < 0.78
            and not jev_res.is_emergency
        ):
            try:
                kb_chunks = self.retriever.retrieve(
                    query=user_message,
                    intent=jev_res.selected_choice,
                    top_k=2,
                )
            except Exception as e:
                jev_res.trace_logs.append(f"[RAG ERROR] Retrieval failed: {e}")
        rag_ms = (time.perf_counter() - rag_start) * 1000.0

        # Step 3: Generation with Local Qwen 3 (Ollama / vLLM)
        gen_res = self.generator.generate(
            user_message=user_message,
            kb_chunks=kb_chunks,
            is_urgent=jev_res.is_urgent,
            frustration_score=jev_res.frustration_score,
            intent=jev_res.selected_choice,
            history=history,
        )

        # Step 4: Handover Context Card Generation on Escalation
        handover_card: Optional[HandoverContextCard] = None
        if jev_res.is_escalated or jev_res.is_emergency:
            handover_card = self._create_handover_card(
                user_message=user_message,
                jev_res=jev_res,
                kb_chunks=kb_chunks,
                account_metadata=account_metadata or {},
            )

        total_latency_ms = round(jev_ms + rag_ms + gen_res.latency_ms, 1)
        breakdown = {
            "jev_ms": round(jev_ms, 1),
            "rag_ms": round(rag_ms, 1),
            "llm_ms": round(gen_res.latency_ms, 1),
            "total_ms": round(total_latency_ms, 1),
        }

        return OrchestrationResult(
            user_message=user_message,
            jev_result=jev_res,
            hierarchy=jev_res.hierarchy,
            retrieved_chunks=kb_chunks,
            generation_result=gen_res,
            handover_card=handover_card,
            latency_breakdown=breakdown,
            wire_request=gen_res.wire_request,
            wire_response=gen_res.wire_response,
            timestamp=timestamp,
            total_latency_ms=round(total_latency_ms, 2),
        )

    def _create_handover_card(
        self,
        user_message: str,
        jev_res: JevResult,
        kb_chunks: List[Dict[str, Any]],
        account_metadata: Dict[str, Any],
    ) -> HandoverContextCard:
        """Synthesize structured context card for the receiving Tier-2 / Senior Human Agent."""
        handover_id = f"CARD-{uuid.uuid4().hex[:8].upper()}"

        recommended_actions: List[str] = []
        if jev_res.is_emergency:
            recommended_actions.extend([
                "Confirm customer safety and verify physical terminal / ONT power status.",
                "Check regional NOC ticket board for concurrent cell tower or GPON cuts.",
                "Authorize emergency mobile hotspot failover data pass.",
                "Dispatch physical field truck roll if line loopback diagnostic fails.",
            ])
        elif jev_res.selected_choice == "billing_inquiry" or jev_res.frustration_score >= 0.80:
            recommended_actions.extend([
                "Acknowledge dispute empathetically and review recent unbilled roaming sessions.",
                "Review invoice CDR timestamps against active roaming pass purchase dates.",
                "Authorize discretionary courtesy adjustment up to $150 if notification delayed.",
                "Apply network carrier latch to prevent future unnotified transit charges.",
            ])
        elif jev_res.selected_choice == "account_security":
            recommended_actions.extend([
                "Initiate immediate cryptographic line lock under protocol SEC-FRAUD-HOLD.",
                "Terminate active web portal sessions across all registered devices.",
                "Verify customer identity using two-factor biometric or government ID portal.",
                "Revoke pending eSIM QR profile and issue new physical SIM with manual PIN.",
            ])
        else:
            recommended_actions.extend([
                "Review customer message history and verify account verification status.",
                "Apply relevant technical troubleshooting steps from referenced KB articles.",
                "Log customer resolution summary in CRM ticket.",
            ])

        return HandoverContextCard(
            handover_id=handover_id,
            timestamp=time.time(),
            trigger_reason=jev_res.dispatch_reason,
            dispatch_queue=jev_res.dispatch_queue,
            customer_state={
                "customer_name": account_metadata.get("customer_name", "Valued Customer"),
                "account_number": account_metadata.get("account_number", "ACCT-VERIFIED-PRIMARY"),
                "phone_number": account_metadata.get("phone_number", "+1-800-555-0199"),
                "last_message": user_message,
            },
            metrics={
                "frustration_score": jev_res.frustration_score,
                "urgency_noul": jev_res.is_urgent,
                "classifier_confidence": jev_res.confidence,
            },
            recommended_actions=recommended_actions,
            grounding_references=[
                {"doc_id": c.get("doc_id"), "title": c.get("title"), "plan_code": c.get("plan_code")}
                for c in kb_chunks
            ],
        )

    async def process_turn_tri(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        account_metadata: Optional[Dict[str, Any]] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
    ) -> TriStreamResult:
        """Executes [1] Jev+Qwen Grounded, [2] Grounded Qwen Baseline, and [3] Grounded Gemini Flash concurrently."""
        overall_start = time.perf_counter()

        # Step 1: Jev Fast-Path Classification (< 1ms)
        jev_res = self.classifier.classify(user_message)
        jev_ms = jev_res.latency_ms

        # Step 2: Shared LanceDB Knowledge Base Retrieval (Grounding context for all 3 streams)
        kb_chunks: List[Dict[str, Any]] = []
        rag_start = time.perf_counter()
        if jev_res.selected_choice != "other_or_unclear":
            try:
                kb_chunks = self.retriever.retrieve(
                    query=user_message,
                    intent=jev_res.selected_choice,
                    top_k=2,
                )
            except Exception as e:
                jev_res.trace_logs.append(f"[RAG ERROR] Retrieval failed: {e}")
        rag_ms = (time.perf_counter() - rag_start) * 1000.0

        def run_local_streams():
            # 1. Specialized Grounded Stack: Jev Intent + Guardrails + LanceDB RAG + Local Qwen
            gen_res_1 = self.generator.generate(
                user_message=user_message,
                kb_chunks=kb_chunks,
                is_urgent=jev_res.is_urgent,
                frustration_score=jev_res.frustration_score,
                intent=jev_res.selected_choice,
                hierarchy=jev_res.hierarchy,
                history=history,
                use_jev_intent_tag=True,
            )
            total_latency_ms_1 = round(jev_ms + rag_ms + gen_res_1.latency_ms, 1)
            breakdown_1 = {
                "jev_ms": round(jev_ms, 1),
                "rag_ms": round(rag_ms, 1),
                "llm_ms": round(gen_res_1.latency_ms, 1),
                "total_ms": round(total_latency_ms_1, 1),
            }
            res_1 = OrchestrationResult(
                user_message=user_message,
                jev_result=jev_res,
                hierarchy=jev_res.hierarchy,
                retrieved_chunks=kb_chunks,
                generation_result=gen_res_1,
                handover_card=None,
                latency_breakdown=breakdown_1,
                wire_request=gen_res_1.wire_request,
                wire_response=gen_res_1.wire_response,
                timestamp=datetime.now(timezone.utc).isoformat(),
                total_latency_ms=total_latency_ms_1,
            )

            # 2. Local Qwen Grounded Baseline (Identical base prompt + identical LanceDB RAG chunks, fixed 180 tokens, no intent tag)
            res_2 = self.generator.generate(
                user_message=user_message,
                system_instruction=None,
                kb_chunks=kb_chunks,
                is_urgent=0.0,
                frustration_score=0.0,
                intent=None,
                history=history,
                max_tokens=180,
                use_jev_intent_tag=False,
            )
            return res_1, res_2

        # Run local generation in background thread
        local_task = asyncio.to_thread(run_local_streams)

        # Cloud Task 1: Jev + Gemini Flash Grounded (Frontier with Intent prior tag & adaptive token budget)
        gemini_jev_task = self.gemini.generate_async(
            user_message=user_message,
            kb_chunks=kb_chunks,
            api_key_override=gemini_api_key,
            model_override=gemini_model,
            intent=jev_res.selected_choice,
            is_urgent=jev_res.is_urgent,
            frustration_score=jev_res.frustration_score,
            use_jev_intent_tag=True,
            max_tokens=220,
        )

        # Cloud Task 2: Control Gemini Flash Grounded (Frontier baseline, standard 512 tokens, no intent tag)
        gemini_ctrl_task = self.gemini.generate_async(
            user_message=user_message,
            kb_chunks=kb_chunks,
            api_key_override=gemini_api_key,
            model_override=gemini_model,
            use_jev_intent_tag=False,
            max_tokens=512,
        )

        # Concurrently execute local streams and both cloud streams
        (stream_1_res, stream_2_res), stream_3_res, stream_4_res = await asyncio.gather(
            local_task, gemini_jev_task, gemini_ctrl_task, return_exceptions=False
        )

        total_turn_latency_ms = round((time.perf_counter() - overall_start) * 1000.0, 2)

        # Ensure telemetry objects exist
        if not stream_1_res.generation_result.telemetry:
            stream_1_res.generation_result.telemetry = StreamTelemetry(
                engine_name="ollama_qwen3",
                model=stream_1_res.generation_result.model,
                latency_ms=stream_1_res.generation_result.latency_ms,
                input_tokens=len(user_message.split()) + 150,
                output_tokens=len(stream_1_res.generation_result.content.split()),
                total_tokens=len(user_message.split()) + 150 + len(stream_1_res.generation_result.content.split()),
                status="success",
            )
        if not stream_2_res.telemetry:
            stream_2_res.telemetry = StreamTelemetry(
                engine_name="ollama_qwen3_grounded",
                model=stream_2_res.model,
                latency_ms=stream_2_res.latency_ms,
                input_tokens=len(user_message.split()) + 80,
                output_tokens=len(stream_2_res.content.split()),
                total_tokens=len(user_message.split()) + 80 + len(stream_2_res.content.split()),
                status="success",
            )

        comparison = self._compute_grounding_audit(
            stream_a=stream_1_res,
            stream_b=stream_3_res,
        )

        return QuadStreamResult(
            user_message=user_message,
            stream_1=stream_1_res,
            stream_2=stream_2_res,
            stream_3=stream_3_res,
            stream_4=stream_4_res,
            stream_a=stream_1_res,
            stream_raw=stream_2_res,
            stream_b=stream_3_res,
            comparison=comparison,
            total_turn_latency_ms=total_turn_latency_ms,
        )

    async def process_turn_dual(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        account_metadata: Optional[Dict[str, Any]] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
    ) -> DualStreamResult:
        """Executes all 4 streams and returns backward-compatible DualStreamResult populated with stream_1, stream_2, stream_3, stream_4."""
        quad_res = await self.process_turn_tri(
            user_message=user_message,
            history=history,
            account_metadata=account_metadata,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
        )
        return DualStreamResult(
            user_message=user_message,
            stream_a=quad_res.stream_1,
            stream_b=quad_res.stream_3,
            stream_raw=quad_res.stream_2,
            stream_1=quad_res.stream_1,
            stream_2=quad_res.stream_2,
            stream_3=quad_res.stream_3,
            stream_4=quad_res.stream_4,
            comparison=quad_res.comparison,
            total_turn_latency_ms=quad_res.total_turn_latency_ms,
        )

    def _compute_grounding_audit(
        self, stream_a: OrchestrationResult, stream_b: GeminiResponse
    ) -> DualComparisonSummary:
        """Evaluates grounding adherence and compares Stream A vs Stream B."""
        verdict = GroundingAuditVerdict()
        chunks = stream_a.retrieved_chunks

        if chunks and len(chunks) > 0:
            top_chunk = chunks[0]
            doc_id = top_chunk.get("doc_id")
            plan_code = top_chunk.get("plan_code")
            apn = top_chunk.get("regional_apn")
            sla = top_chunk.get("sla")

            verdict.doc_id = doc_id
            verdict.plan_code = plan_code
            verdict.regional_apn = apn
            verdict.sla = sla

            content_a_lower = stream_a.generation_result.content.lower()
            verdict.verified_doc_cited = bool(doc_id and doc_id.lower() in content_a_lower)
            verdict.plan_code_cited = bool(plan_code and plan_code.lower() in content_a_lower)
            verdict.apn_cited = bool(apn and apn.lower() in content_a_lower)
            verdict.sla_cited = bool(sla and (sla.lower() in content_a_lower or "sla" in content_a_lower))

        verdict.emergency_guardrail_triggered = stream_a.jev_result.is_urgent >= 0.75 or stream_a.jev_result.is_emergency

        # Determine faster stream
        lat_a = stream_a.total_latency_ms
        lat_b = stream_b.telemetry.latency_ms
        faster = "stream_b" if lat_b < lat_a else "stream_a"
        delta = round(abs(lat_a - lat_b), 2)

        summary_parts = []
        if verdict.verified_doc_cited or verdict.plan_code_cited or verdict.apn_cited:
            summary_parts.append(f"Stream A accurately grounded in {verdict.doc_id or 'verified telco specs'}.")
        else:
            summary_parts.append("Stream A followed general company assistance guidelines.")

        if stream_b.telemetry.status == "success":
            if stream_b.grounded:
                summary_parts.append("Stream B (Gemini 3.8 Flash) grounded in verified LanceDB internal specifications.")
            else:
                summary_parts.append("Stream B provided broad cloud-based conversational advice without internal SLA binding.")
        elif stream_b.telemetry.status == "missing_key":
            summary_parts.append("Stream B awaiting Google AI Studio API key.")
        else:
            summary_parts.append(f"Stream B encountered: {stream_b.telemetry.error_message}.")

        verdict.summary = " ".join(summary_parts)

        return DualComparisonSummary(
            latency_delta_ms=delta,
            faster_stream=faster,
            stream_a_grounded=verdict.plan_code_cited or verdict.apn_cited or bool(chunks),
            stream_b_grounded=bool(stream_b.grounded or (chunks and len(chunks) > 0)),
            stream_1_grounded=verdict.plan_code_cited or verdict.apn_cited or bool(chunks),
            stream_2_grounded=bool(chunks and len(chunks) > 0),
            stream_3_grounded=bool(stream_b.grounded or (chunks and len(chunks) > 0)),
            audit=verdict,
        )
