"""FastAPI Local Server for Telco AI Intent Routing & RAG Pipeline Workbench.

Serves:
1. Interactive Web Studio at http://localhost:8001/
2. Jev System One classifier at POST /v1/classifier
3. End-to-end RAG + Qwen 3 orchestrator at POST /v1/orchestrate
4. LanceDB retrieval explorer at GET /v1/rag/search
5. System health and engine status at GET /v1/health
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Always load .env
load_dotenv(override=True)

from pipelines.orchestrator import PipelineOrchestrator
from rag.retriever import TelcoRetriever
from services.jev.classifier import JevClassifier

app = FastAPI(
    title="Telco AI Intent Routing & RAG Workbench API",
    version="1.13-SPEC",
    description="Deterministic Next-Token Intent Routing, Noul Urgency, Score Frustration, and Grounded Qwen 3 Answering Service",
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons
orchestrator = PipelineOrchestrator()
classifier = orchestrator.classifier
retriever = orchestrator.retriever


class ClassifyRequest(BaseModel):
    text: Optional[str] = Field(None, description="Input user message")
    state: Optional[str] = Field(None, description="Alternative state field (simple-jev compatibility)")
    questions: Optional[Dict[str, Any]] = None
    confidence_threshold: Optional[float] = 0.70


class OrchestrateRequest(BaseModel):
    message: Optional[str] = None
    text: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    customer_name: Optional[str] = "Valued Customer"
    account_number: Optional[str] = "ACCT-PRIMARY-01"


class DualOrchestrateRequest(BaseModel):
    message: Optional[str] = None
    text: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    customer_name: Optional[str] = "Valued Customer"
    account_number: Optional[str] = "ACCT-PRIMARY-01"
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_studio_ui():
    """Serves the interactive Jev Classifier Studio HTML application."""
    studio_path = Path("Docs/jev_intent_classifier_studio.html")
    if not studio_path.exists():
        return HTMLResponse("<h2>Error: Docs/jev_intent_classifier_studio.html not found.</h2>", status_code=404)
    with open(studio_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/v1/health")
async def health_check():
    """System status and telemetry."""
    load_dotenv(override=True)
    lancedb_ready = retriever._table is not None
    table_rows = len(retriever._table) if lancedb_ready else 0

    gem_key = os.getenv("GEMINI_API_KEY") or orchestrator.gemini.api_key
    gem_model = os.getenv("GEMINI_MODEL") or orchestrator.gemini.model

    return {
        "status": "online",
        "jev_engine": classifier.api_key and "vercel_ai_gateway" or "offline_deterministic_jev",
        "gateway_url": classifier.gateway_url,
        "api_key_configured": bool(classifier.api_key),
        "gemini_configured": bool(gem_key),
        "gemini_model": gem_model,
        "ollama_endpoint": orchestrator.generator.ollama_url,
        "ollama_model": orchestrator.generator.ollama_model,
        "lancedb_records": table_rows,
        "active_taxonomy": classifier.choices,
    }


TELCO_TAXONOMY_TREE = [
    {
        "l1_id": "billing_inquiry",
        "l1_name": "Billing & Payments",
        "desc": "Invoices, charges, disputed transit fees, payment methods, refunds",
        "queue": "Billing & Accounts Queue",
        "color": "sky",
        "l2_categories": [
            {
                "l2_name": "Invoice Dispute",
                "l3_intents": ["double_billing_deduction", "unrecognized_roaming_charge", "disputed_line_item_fee"]
            },
            {
                "l2_name": "Payment Methods & AutoPay",
                "l3_intents": ["autopay_discount_setup", "declined_card_update", "payment_extension_request"]
            },
            {
                "l2_name": "Billing Cycle & Proration",
                "l3_intents": ["prorated_charge_calculation", "statement_billing_date_change"]
            }
        ]
    },
    {
        "l1_id": "network_connectivity",
        "l1_name": "Mobile Network & Wireless",
        "desc": "Signal drops, 5G NR reception, cellular APN configs, VoLTE/VoWiFi",
        "queue": "Wireless Network Ops",
        "color": "indigo",
        "l2_categories": [
            {
                "l2_name": "APN & Data Configuration",
                "l3_intents": ["manual_5g_apn_setup", "volte_vowifi_provisioning", "mobile_hotspot_tethering_fail"]
            },
            {
                "l2_name": "Signal & Coverage",
                "l3_intents": ["weak_5g_nr_indoor_coverage", "dropped_calls_tower_handover"]
            },
            {
                "l2_name": "Cellular Outage Incident",
                "l3_intents": ["cell_tower_degraded_service", "emergency_regional_blackout"]
            }
        ]
    },
    {
        "l1_id": "fiber_broadband",
        "l1_name": "Fiber FTTH & Broadband",
        "desc": "Optical drop cable, ONT LOS red light, Wi-Fi 6 mesh, speed diagnostics",
        "queue": "Broadband & Fiber NOC",
        "color": "emerald",
        "l2_categories": [
            {
                "l2_name": "Optical Link Hardware Failure",
                "l3_intents": ["ont_los_red_blinking_optical_cut", "pon_unregistered_flashing", "optical_attenuation_high"]
            },
            {
                "l2_name": "Wi-Fi & Home Networking",
                "l3_intents": ["mesh_wifi6_node_pairing", "bridge_mode_vlan_tagging", "slow_wifi_throughput_diagnostic"]
            },
            {
                "l2_name": "Field Operations Dispatch",
                "l3_intents": ["technician_truck_roll_booking", "physical_drop_cable_severed"]
            }
        ]
    },
    {
        "l1_id": "roaming_passes",
        "l1_name": "Global Roaming & Travel",
        "desc": "Zones 1-4 day passes, foreign carrier steering, FUP throttles, flight passes",
        "queue": "Global Roaming Desk",
        "color": "amber",
        "l2_categories": [
            {
                "l2_name": "Roaming Pass Activation",
                "l3_intents": ["zone_1_eu_day_pass", "zone_2_americas_bundle", "zone_4_inflight_maritime_pass"]
            },
            {
                "l2_name": "Carrier Steering & Handshake",
                "l3_intents": ["foreign_carrier_auth_error_403", "roaming_data_apn_not_connecting", "partner_network_manual_latch"]
            },
            {
                "l2_name": "Fair Usage Policy (FUP)",
                "l3_intents": ["fup_speed_throttle_reset", "international_boost_topup"]
            }
        ]
    },
    {
        "l1_id": "account_security",
        "l1_name": "SIM & Identity Security",
        "desc": "eSIM QR provisioning, SIM swap fraud protection, PUK codes, KYC holds",
        "queue": "Fraud & Security Tier-2",
        "color": "rose",
        "l2_categories": [
            {
                "l2_name": "eSIM Provisioning",
                "l3_intents": ["qr_code_activation_fail", "eid_pairing_validation_error", "physical_to_esim_migration"]
            },
            {
                "l2_name": "SIM Swap Fraud Protection",
                "l3_intents": ["unauthorized_sim_swap_cooling_hold", "emergency_account_lockdown", "biometric_identity_validation"]
            },
            {
                "l2_name": "Security Credentials",
                "l3_intents": ["puk_code_retrieval", "master_account_pin_reset"]
            }
        ]
    },
    {
        "l1_id": "plan_upgrade",
        "l1_name": "Plan Tiers & Subscriptions",
        "desc": "Prepaid/postpaid upgrades, family data pooling, corporate retention discounts",
        "queue": "Retention & Sales Desk",
        "color": "purple",
        "l2_categories": [
            {
                "l2_name": "Consumer Plan Migration",
                "l3_intents": ["5g_infinite_tier_upgrade", "flexipay_rollover_bundle"]
            },
            {
                "l2_name": "Commercial Enterprise Pooling",
                "l3_intents": ["enterprise_volume_pooling_negotiation", "multi_line_business_agreement"]
            },
            {
                "l2_name": "Retention & Churn Prevention",
                "l3_intents": ["loyalty_contract_renewal_discount", "port_out_retention_offer"]
            }
        ]
    },
    {
        "l1_id": "human_escalation",
        "l1_name": "Customer Care Escalations",
        "desc": "Explicit supervisor demands, regulatory complaints, senior tier-2 transfers",
        "queue": "Senior Human Queue",
        "color": "red",
        "l2_categories": [
            {
                "l2_name": "Managerial Escalation",
                "l3_intents": ["senior_operations_supervisor_handoff", "regulatory_ombudsman_threat"]
            },
            {
                "l2_name": "Specialist Agent Dispatch",
                "l3_intents": ["tier2_specialist_live_transfer", "fraud_investigator_warm_handoff"]
            }
        ]
    },
    {
        "l1_id": "other_or_unclear",
        "l1_name": "General Triage Fallback",
        "desc": "Ambiguous chatter, greeting, unclassifiable customer inquiries",
        "queue": "General Triage Fallback",
        "color": "slate",
        "isFallback": True,
        "l2_categories": [
            {
                "l2_name": "Ambiguous Inquiries",
                "l3_intents": ["unclassified_customer_inquiry", "greeting_casual"]
            }
        ]
    }
]


@app.get("/v1/taxonomy")
async def get_taxonomy_tree():
    """Returns the full 3-level hierarchical telco intent taxonomy (L1 -> L2 -> L3)."""
    return {
        "levels": ["L1_Domain", "L2_Subcategory", "L3_Leaf_Intent"],
        "taxonomy": TELCO_TAXONOMY_TREE,
    }


@app.post("/v1/classifier")
async def classify_text(req: ClassifyRequest, request: Request):
    """Jev System One classification endpoint compatible with Studio and simple-jev."""
    input_text = req.text or req.state
    if not input_text:
        try:
            body = await request.json()
            input_text = body.get("text") or body.get("state")
        except Exception:
            pass

    if not input_text:
        raise HTTPException(status_code=400, detail="Missing 'text' or 'state' in request payload.")

    auth_header = request.headers.get("Authorization")
    if auth_header and "Bearer " in auth_header:
        bearer_key = auth_header.replace("Bearer ", "").strip()
        if bearer_key and not classifier.api_key:
            classifier.api_key = bearer_key

    result = classifier.classify(input_text)
    return {
        "selectedChoice": result.selected_choice,
        "confidence": result.confidence,
        "distribution": [d.model_dump() for d in result.distribution],
        "hierarchy": result.hierarchy.model_dump(),
        "noulUrgency": result.is_urgent,
        "frustrationScore": result.frustration_score,
        "dispatchQueue": result.dispatch_queue,
        "dispatchReason": result.dispatch_reason,
        "latencyMs": result.latency_ms,
        "engine": result.engine_used,
        "traceLogs": result.trace_logs,
        "isEscalated": result.is_escalated,
        "isEmergency": result.is_emergency,
    }


@app.post("/v1/orchestrate")
async def orchestrate_turn(req: OrchestrateRequest):
    """End-to-end pipeline execution: Jev -> RAG (LanceDB 30K) -> Qwen 3 -> Escalation Card."""
    input_text = req.message or req.text
    if not input_text:
        raise HTTPException(status_code=400, detail="Missing 'message' or 'text' in request payload.")

    meta = {
        "customer_name": req.customer_name,
        "account_number": req.account_number,
    }
    result = orchestrator.process_turn(
        user_message=input_text,
        history=req.history,
        account_metadata=meta,
    )
    return result.model_dump()


@app.post("/v1/orchestrate/dual")
async def orchestrate_dual_turn(req: DualOrchestrateRequest, request: Request):
    """Executes [1] Jev+Qwen, [2] Raw Qwen, and [3] Google Gemini Flash concurrently."""
    input_text = req.message or req.text
    if not input_text:
        raise HTTPException(status_code=400, detail="Missing 'message' or 'text' in request payload.")

    gemini_key = req.gemini_api_key or request.headers.get("X-Gemini-Key")
    meta = {
        "customer_name": req.customer_name,
        "account_number": req.account_number,
    }
    result = await orchestrator.process_turn_dual(
        user_message=input_text,
        history=req.history,
        account_metadata=meta,
        gemini_api_key=gemini_key,
        gemini_model=req.gemini_model,
    )
    return result.model_dump()


@app.post("/v1/orchestrate/tri")
async def orchestrate_tri_turn(req: DualOrchestrateRequest, request: Request):
    """Executes [1] Jev+Qwen, [2] Qwen Control, [3] Jev+Gemini, and [4] Gemini Control in parallel."""
    input_text = req.message or req.text
    if not input_text:
        raise HTTPException(status_code=400, detail="Missing 'message' or 'text' in request payload.")

    gemini_key = req.gemini_api_key or request.headers.get("X-Gemini-Key")
    meta = {
        "customer_name": req.customer_name,
        "account_number": req.account_number,
    }
    result = await orchestrator.process_turn_tri(
        user_message=input_text,
        history=req.history,
        account_metadata=meta,
        gemini_api_key=gemini_key,
        gemini_model=req.gemini_model,
    )
    return result.model_dump()


@app.post("/v1/orchestrate/quad")
async def orchestrate_quad_turn(req: DualOrchestrateRequest, request: Request):
    """Executes full 4-way benchmark: [1] Jev+Qwen, [2] Qwen Control, [3] Jev+Gemini, and [4] Gemini Control in parallel."""
    return await orchestrate_tri_turn(req, request)


class GeminiTestRequest(BaseModel):
    message: str
    api_key: Optional[str] = None
    model: Optional[str] = None


@app.post("/v1/gemini")
async def test_gemini_endpoint(req: GeminiTestRequest, request: Request):
    """Test Google AI Studio Gemini connection standalone."""
    key = req.api_key or request.headers.get("X-Gemini-Key")
    res = await orchestrator.gemini.generate_async(
        user_message=req.message,
        api_key_override=key,
        model_override=req.model,
    )
    return res.model_dump()


@app.get("/v1/rag/search")
async def search_knowledge_base(query: str, intent: Optional[str] = None, limit: int = 3):
    """Query 30,000 telco articles from LanceDB."""
    chunks = retriever.retrieve(query=query, intent=intent, top_k=limit)
    return {"query": query, "intent": intent, "count": len(chunks), "results": chunks}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start Telco AI Workbench Local Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind (default: 8001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")

    args = parser.parse_args()
    print("=" * 70)
    print(f" Starting Telco AI Intent Routing & RAG Workbench Server")
    print(f" Studio Web UI:  http://localhost:{args.port}/")
    print(f" API Health:     http://localhost:{args.port}/v1/health")
    print(f" Classification: POST http://localhost:{args.port}/v1/classifier")
    print(f" Orchestration:  POST http://localhost:{args.port}/v1/orchestrate")
    print("=" * 70)

    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)
