"""Expanded Multi-Domain Scientific Benchmark: 4 Domains x 3 Query Complexities x 3 KB Densities x 9 Streams.

Addresses Tier-1 Peer Review Critiques:
- W1: Implements Zero-Fluff Prompt Invariant to test whether token budgeting can preserve facts without pleasantry bloat.
- W2: Expands scope to 4 diverse enterprise domains: Fiber Broadband, Billing Disputes, 5G Network Ops, Contract Upgrades.
- W3: 28 Canonical Domain Grounding Targets with Token Efficiency Ratios (% recall per output token).
- W4: Rigorous 3x3 comparison: Small Local (1.7B), Medium Local (7B), Frontier Cloud (Gemini) x [Control, Regex, Jev (TypeSafe AI)].
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import sys
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(override=True)

from services.jev.classifier import JevClassifier
from services.vllm.generator import QwenGenerator
from services.gemini.client import GeminiClient

# Zero-Fluff Prompt Invariant (Resolves Reviewer W1)
ZERO_FLUFF_PROMPT = (
    "You are a technical customer operations specialist for Apex Telecom. "
    "Provide direct, concise, and professional answers formatted in clean Markdown with bullet points. "
    "CRITICAL CONSTRAINTS: "
    "1. Do NOT include polite greetings or apologies (e.g. do NOT say 'Thank you for reaching out' or 'We apologize'). "
    "2. Do NOT restate or repeat the customer's problem. "
    "3. Answer the customer directly and cite the exact Plan codes, APN/Billing codes, SLAs, and compensation amounts immediately. "
    "4. Output ONLY the final customer-facing bullet points."
)

# -----------------------------------------------------------------------------
# 1. 4 ENTERPRISE DOMAINS DATASET
# -----------------------------------------------------------------------------
DOMAINS = {
    "Fiber": {
        "queries": {
            "Short": "Fiber line cut outside, red LOS blinking.",
            "Mid": "My fiber optic drop cable was severed by utility workers outside my house. The Optical Terminal has a blinking red LOS light and internet is completely down.",
            "Long": "I am writing regarding my fiber internet service which went completely dead about 45 minutes ago after municipal excavation crews dug up the sidewalk in front of my driveway. They visibly severed the thick black fiber drop cable. On my ONT terminal inside the garage, the Power light is solid green, but the LOS indicator is rapidly blinking red. I run a home business and urgently need this repaired or escalated to an outside plant field technician. My account is in good standing.",
        },
        "kb": {
            "Short": [
                {
                    "doc_id": "FIBER-DOC-1",
                    "title": "GPON Drop Restoration SOP",
                    "plan_code": "TELCO-XGSPON-SYMM-10G",
                    "regional_apn": "internet.telco.us-east",
                    "sla": "24-Hour Resolution SLA",
                    "summary": "Physical drop severance requires emergency outside plant technician dispatch under ticket GPON-OUTAGE-L3. Standard repair SLA is 24 hours.",
                }
            ],
            "Mid": [
                {
                    "doc_id": "FIBER-DOC-1",
                    "title": "GPON Drop Restoration SOP",
                    "plan_code": "TELCO-XGSPON-SYMM-10G",
                    "regional_apn": "internet.telco.us-east",
                    "sla": "24-Hour Resolution SLA",
                    "summary": "Physical drop severance requires emergency outside plant technician dispatch under ticket GPON-OUTAGE-L3. Standard repair SLA is 24 hours.",
                },
                {
                    "doc_id": "FIBER-DOC-2",
                    "title": "Automatic SLA Outage Credit",
                    "plan_code": "TELCO-XGSPON-SYMM-10G",
                    "regional_apn": "internet.telco.us-east",
                    "sla": "Service Credit",
                    "summary": "If complete service outage exceeds 24 hours, an automatic $25 per day credit is applied to account under billing code SLA-CREDIT-AUTO.",
                },
            ],
            "Long": [
                {
                    "doc_id": "FIBER-DOC-1",
                    "title": "GPON Drop Restoration SOP",
                    "plan_code": "TELCO-XGSPON-SYMM-10G",
                    "regional_apn": "internet.telco.us-east",
                    "sla": "24-Hour Resolution SLA",
                    "summary": "Physical drop severance requires emergency outside plant technician dispatch under ticket GPON-OUTAGE-L3. Standard repair SLA is 24 hours.",
                },
                {
                    "doc_id": "FIBER-DOC-2",
                    "title": "Automatic SLA Outage Credit",
                    "plan_code": "TELCO-XGSPON-SYMM-10G",
                    "regional_apn": "internet.telco.us-east",
                    "sla": "Service Credit",
                    "summary": "If complete service outage exceeds 24 hours, an automatic $25 per day credit is applied to account under billing code SLA-CREDIT-AUTO.",
                },
                {
                    "doc_id": "FIBER-DOC-3",
                    "title": "Temporary Backup Hotspot Pass",
                    "plan_code": "TELCO-XGSPON-SYMM-10G",
                    "regional_apn": "5g.telco.nr-data",
                    "sla": "Instant Hotspot",
                    "summary": "Confirmed severed drop subscribers qualify for an immediate 100GB 5G backup hotspot pass issued to their linked mobile number.",
                },
                {
                    "doc_id": "FIBER-DOC-4",
                    "title": "Optical Line Safety Warning",
                    "plan_code": "All",
                    "regional_apn": "N/A",
                    "sla": "Hazard Protocol",
                    "summary": "Customers must not look into cut optical fiber ends due to invisible class 3B laser radiation causing retinal burns.",
                },
            ],
        },
        "targets": {
            "Short": ["telco-xgspon", "internet.telco.us-east", "24-hour"],
            "Mid": ["telco-xgspon", "internet.telco.us-east", "24-hour", "sla-credit-auto", "$25"],
            "Long": ["telco-xgspon", "internet.telco.us-east", "24-hour", "sla-credit-auto", "$25", "100gb", "laser"],
        },
    },
    "Billing": {
        "queries": {
            "Short": "Why is my bill $45 higher this month?",
            "Mid": "I was charged twice on my credit card for international roaming while traveling in London, and autopay took the second unauthorized deduction of $45 yesterday.",
            "Long": "I am extremely frustrated with my latest invoice statement. I returned from a business trip to the UK last week and specifically purchased a single 7-day international data pass for $35. However, my checking account was hit with two separate charges of $45 and $35 under billing code ROAM-EUR-DATA, causing an overdraft. AutoPay already executed this morning without my approval. I demand an immediate reversal of the duplicate $45 charge and a waiver of any dispute fees.",
        },
        "kb": {
            "Short": [
                {
                    "doc_id": "BILL-DOC-1",
                    "title": "Duplicate Roaming Charge Reversal",
                    "plan_code": "GLOBAL-ROAM-EUR",
                    "regional_apn": "N/A",
                    "sla": "48-Hour Bank Credit SLA",
                    "summary": "Duplicate roaming deductions are reversed within 48 hours directly to the original payment method under adjustment code ADJ-ROAM-DUP.",
                }
            ],
            "Mid": [
                {
                    "doc_id": "BILL-DOC-1",
                    "title": "Duplicate Roaming Charge Reversal",
                    "plan_code": "GLOBAL-ROAM-EUR",
                    "regional_apn": "N/A",
                    "sla": "48-Hour Bank Credit SLA",
                    "summary": "Duplicate roaming deductions are reversed within 48 hours directly to the original payment method under adjustment code ADJ-ROAM-DUP.",
                },
                {
                    "doc_id": "BILL-DOC-2",
                    "title": "AutoPay Error Courtesy Guarantee",
                    "plan_code": "GLOBAL-ROAM-EUR",
                    "regional_apn": "N/A",
                    "sla": "Compensation Credit",
                    "summary": "When duplicate charges execute via AutoPay, customers receive an automatic $15 courtesy statement credit under policy code SLA-AUTOPAY-REV.",
                },
            ],
            "Long": [
                {
                    "doc_id": "BILL-DOC-1",
                    "title": "Duplicate Roaming Charge Reversal",
                    "plan_code": "GLOBAL-ROAM-EUR",
                    "regional_apn": "N/A",
                    "sla": "48-Hour Bank Credit SLA",
                    "summary": "Duplicate roaming deductions are reversed within 48 hours directly to the original payment method under adjustment code ADJ-ROAM-DUP.",
                },
                {
                    "doc_id": "BILL-DOC-2",
                    "title": "AutoPay Error Courtesy Guarantee",
                    "plan_code": "GLOBAL-ROAM-EUR",
                    "regional_apn": "N/A",
                    "sla": "Compensation Credit",
                    "summary": "When duplicate charges execute via AutoPay, customers receive an automatic $15 courtesy statement credit under policy code SLA-AUTOPAY-REV.",
                },
                {
                    "doc_id": "BILL-DOC-3",
                    "title": "Collection Dispute Suspension",
                    "plan_code": "All",
                    "regional_apn": "N/A",
                    "sla": "Billing Protection",
                    "summary": "Disputed transactions are placed under a 30-day collection freeze under tracking code DISPUTE-SUSPEND-30 preventing late fees.",
                },
                {
                    "doc_id": "BILL-DOC-4",
                    "title": "Cross-Border VAT Tax Harmonization",
                    "plan_code": "GLOBAL-ROAM-EUR",
                    "regional_apn": "N/A",
                    "sla": "Tax Refund",
                    "summary": "Foreign VAT assessments on reversed roaming charges are credited under European statutory tax rebate code TAX-EUR-CORRECT.",
                },
            ],
        },
        "targets": {
            "Short": ["adj-roam-dup", "48-hour", "duplicate"],
            "Mid": ["adj-roam-dup", "48-hour", "duplicate", "sla-autopay-rev", "$15"],
            "Long": ["adj-roam-dup", "48-hour", "duplicate", "sla-autopay-rev", "$15", "dispute-suspend-30", "tax-eur"],
        },
    },
    "Network": {
        "queries": {
            "Short": "No 5G cellular signal, SOS only displayed.",
            "Mid": "My iPhone has dropped from 5G to SOS Only in downtown Austin. Voice calls fail immediately and cellular data will not attach to the network.",
            "Long": "Since 8:00 AM this morning, my primary business phone has been completely disconnected from the cellular grid across the downtown metro corridor. The device status bar displays 'SOS Only' with zero signal bars. I restarted my phone and toggled Airplane mode, but cellular data refuses to attach. I cannot receive two-factor SMS codes or customer phone calls. Is there a local cell tower outage, or is my eSIM corrupted? Please investigate tower status immediately.",
        },
        "kb": {
            "Short": [
                {
                    "doc_id": "NET-DOC-1",
                    "title": "Cellular Radio Outage Triage SOP",
                    "plan_code": "APEX-5G-MOBILE-PRO",
                    "regional_apn": "broadband.5g.apex",
                    "sla": "2-Hour Network Triage SLA",
                    "summary": "SOS Only conditions in metro cells are logged under emergency NOC ticket code CELL-NOC-L2. Core VoLTE-EVS voice gateway triage within 2 hours.",
                }
            ],
            "Mid": [
                {
                    "doc_id": "NET-DOC-1",
                    "title": "Cellular Radio Outage Triage SOP",
                    "plan_code": "APEX-5G-MOBILE-PRO",
                    "regional_apn": "broadband.5g.apex",
                    "sla": "2-Hour Network Triage SLA",
                    "summary": "SOS Only conditions in metro cells are logged under emergency NOC ticket code CELL-NOC-L2. Core VoLTE-EVS voice gateway triage within 2 hours.",
                },
                {
                    "doc_id": "NET-DOC-2",
                    "title": "Emergency Wi-Fi Calling Override",
                    "plan_code": "APEX-5G-MOBILE-PRO",
                    "regional_apn": "broadband.5g.apex",
                    "sla": "Instant Workaround",
                    "summary": "Users experiencing cell tower radio disconnects can enable emergency Wi-Fi calling via secure carrier tunnel pass WIFI-CALL-OVERRIDE.",
                },
            ],
            "Long": [
                {
                    "doc_id": "NET-DOC-1",
                    "title": "Cellular Radio Outage Triage SOP",
                    "plan_code": "APEX-5G-MOBILE-PRO",
                    "regional_apn": "broadband.5g.apex",
                    "sla": "2-Hour Network Triage SLA",
                    "summary": "SOS Only conditions in metro cells are logged under emergency NOC ticket code CELL-NOC-L2. Core VoLTE-EVS voice gateway triage within 2 hours.",
                },
                {
                    "doc_id": "NET-DOC-2",
                    "title": "Emergency Wi-Fi Calling Override",
                    "plan_code": "APEX-5G-MOBILE-PRO",
                    "regional_apn": "broadband.5g.apex",
                    "sla": "Instant Workaround",
                    "summary": "Users experiencing cell tower radio disconnects can enable emergency Wi-Fi calling via secure carrier tunnel pass WIFI-CALL-OVERRIDE.",
                },
                {
                    "doc_id": "NET-DOC-3",
                    "title": "Fast-Track eSIM Reissue Protocol",
                    "plan_code": "All Mobile",
                    "regional_apn": "N/A",
                    "sla": "15-Minute Reissue",
                    "summary": "If physical or virtual SIM fails attachment, automated eSIM profile refresh generates a replacement QR within 15 minutes under code ESIM-REISSUE-FAST.",
                },
                {
                    "doc_id": "NET-DOC-4",
                    "title": "Outage Compensation Credit",
                    "plan_code": "APEX-5G-MOBILE-PRO",
                    "regional_apn": "N/A",
                    "sla": "Account Credit",
                    "summary": "Service outages lasting over 4 hours qualify for an automated $10 per day credit under customer guarantee code SLA-NET-OUTAGE.",
                },
            ],
        },
        "targets": {
            "Short": ["cell-noc-l2", "broadband.5g.apex", "volte-evs"],
            "Mid": ["cell-noc-l2", "broadband.5g.apex", "volte-evs", "wifi-call-override", "2-hour"],
            "Long": ["cell-noc-l2", "broadband.5g.apex", "volte-evs", "wifi-call-override", "2-hour", "esim-reissue-fast", "$10"],
        },
    },
    "Contract": {
        "queries": {
            "Short": "Want to upgrade to 5G Pro plan.",
            "Mid": "I want to upgrade my 4 lines to the new 5G Pro Unlimited tier and trade in my phones, but I need early termination fees waived.",
            "Long": "I have been an Apex Telecom subscriber for 7 years on our legacy Unlimited Plus plan. I received a promotional mailer offering a trade-in upgrade to the new 5G Pro Unlimited plan with complimentary device subsidies. However, when I look at the online checkout, the cart is charging me a $200 early termination fee for two lines and dropping my 20% loyalty discount. If Apex cannot honor the promotional waiver and preserve my multiline pooling, I will migrate my entire family account to a competitor.",
        },
        "kb": {
            "Short": [
                {
                    "doc_id": "CONT-DOC-1",
                    "title": "5G Pro Tier Migration SOP",
                    "plan_code": "PLAN-5G-PRO-UNLIM",
                    "regional_apn": "N/A",
                    "sla": "Instant Tier Shift",
                    "summary": "Upgrading to PLAN-5G-PRO-UNLIM qualifies tenured subscribers for early termination fee cancellation under waiver code ETF-WAIVE-PROMO.",
                }
            ],
            "Mid": [
                {
                    "doc_id": "CONT-DOC-1",
                    "title": "5G Pro Tier Migration SOP",
                    "plan_code": "PLAN-5G-PRO-UNLIM",
                    "regional_apn": "N/A",
                    "sla": "Instant Tier Shift",
                    "summary": "Upgrading to PLAN-5G-PRO-UNLIM qualifies tenured subscribers for early termination fee cancellation under waiver code ETF-WAIVE-PROMO.",
                },
                {
                    "doc_id": "CONT-DOC-2",
                    "title": "Multiline Family Pool Lock",
                    "plan_code": "PLAN-5G-PRO-UNLIM",
                    "regional_apn": "N/A",
                    "sla": "Discount Protection",
                    "summary": "Subscribers migrating 4 or more lines preserve their 20% loyalty discount under family pool code MAX-POOL-SHARE with loyalty override RET-LOYALTY-7YR.",
                },
            ],
            "Long": [
                {
                    "doc_id": "CONT-DOC-1",
                    "title": "5G Pro Tier Migration SOP",
                    "plan_code": "PLAN-5G-PRO-UNLIM",
                    "regional_apn": "N/A",
                    "sla": "Instant Tier Shift",
                    "summary": "Upgrading to PLAN-5G-PRO-UNLIM qualifies tenured subscribers for early termination fee cancellation under waiver code ETF-WAIVE-PROMO.",
                },
                {
                    "doc_id": "CONT-DOC-2",
                    "title": "Multiline Family Pool Lock",
                    "plan_code": "PLAN-5G-PRO-UNLIM",
                    "regional_apn": "N/A",
                    "sla": "Discount Protection",
                    "summary": "Subscribers migrating 4 or more lines preserve their 20% loyalty discount under family pool code MAX-POOL-SHARE with loyalty override RET-LOYALTY-7YR.",
                },
                {
                    "doc_id": "CONT-DOC-3",
                    "title": "Device Trade-In Subsidy Terms",
                    "plan_code": "PLAN-5G-PRO-UNLIM",
                    "regional_apn": "N/A",
                    "sla": "Hardware Subsidy",
                    "summary": "Trade-in hardware applies an $800 recurring 24-month statement credit per line under promotional subsidy agreement SUB-TRADE-MAX.",
                },
                {
                    "doc_id": "CONT-DOC-4",
                    "title": "30-Day Satisfaction Guarantee",
                    "plan_code": "All Contracts",
                    "regional_apn": "N/A",
                    "sla": "Return Policy",
                    "summary": "Subscribers retain a 30-day grace period to revert to their prior plan with zero penalty under clause TERMS-RETURN-30.",
                },
            ],
        },
        "targets": {
            "Short": ["plan-5g-pro-unlim", "etf-waive-promo"],
            "Mid": ["plan-5g-pro-unlim", "etf-waive-promo", "max-pool-share", "ret-loyalty-7yr", "20%"],
            "Long": ["plan-5g-pro-unlim", "etf-waive-promo", "max-pool-share", "ret-loyalty-7yr", "20%", "sub-trade-max", "$800"],
        },
    },
}


def score_grounding(content: str, targets: List[str]) -> float:
    """Computes Grounding Recall % based on target entities present in the response."""
    if not targets:
        return 100.0
    text_lower = content.lower()
    matches = sum(1 for t in targets if t.lower() in text_lower)
    return round((matches / len(targets)) * 100.0, 1)


def regex_classify_turn(text: str) -> Dict[str, Any]:
    """Lightweight rule-based regex intent classifier."""
    lower = text.lower()
    if re.search(r"\b(fiber|los|ont|cut|severed|red light)\b", lower):
        return {"intent": "fiber_broadband", "urgent": 1.0, "budget": 80}
    elif re.search(r"\b(bill|charged|charge|invoice|autopay|refund|fee)\b", lower):
        return {"intent": "billing_inquiry", "urgent": 0.2, "budget": 90}
    elif re.search(r"\b(5g|sos|signal|tower|data|network|calling)\b", lower):
        return {"intent": "network_connectivity", "urgent": 0.8, "budget": 85}
    elif re.search(r"\b(upgrade|plan|tier|etf|contract|trade-in)\b", lower):
        return {"intent": "plan_upgrade", "urgent": 0.3, "budget": 95}
    return {"intent": "general_inquiry", "urgent": 0.0, "budget": 120}


# -----------------------------------------------------------------------------
# 2. STREAM EXECUTOR
# -----------------------------------------------------------------------------
async def execute_stream_turn(
    stream_id: str,
    query: str,
    kb_chunks: List[Dict[str, Any]],
    classifier: JevClassifier,
    gen_17b: QwenGenerator,
    gen_7b: QwenGenerator,
    gemini: GeminiClient,
    jev_cache: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a single stream under controlled experimental conditions."""

    # 1. Obtain classification context if applicable
    jev_intent: Optional[str] = None
    urgency: float = 0.0
    frustration: float = 0.0
    token_budget: int = 180
    use_tag: bool = False

    # Regex classification
    if "Regex" in stream_id:
        reg_res = regex_classify_turn(query)
        jev_intent = reg_res["intent"]
        urgency = reg_res["urgent"]
        token_budget = reg_res["budget"]
        use_tag = True

    # Jev (TypeSafe AI) classification
    elif "Jev" in stream_id:
        if query not in jev_cache:
            jev_res = classifier.classify(query)
            jev_cache[query] = jev_res
        else:
            jev_res = jev_cache[query]
        jev_intent = jev_res.selected_choice
        urgency = jev_res.is_urgent
        frustration = jev_res.frustration_score
        # Adaptive budget
        token_budget = 80 if urgency > 0.7 else (90 if frustration > 0.7 else 100)
        use_tag = True

    temp = 0.0  # Greedy decoding for strict scientific repeatability

    # --- Small Local (Qwen 1.7B) ---
    if stream_id in ["S1_1.7B_Ctrl", "S2_1.7B_Regex", "S3_1.7B_Jev"]:
        effective_budget = 180 if "Ctrl" in stream_id else token_budget
        res = gen_17b.generate(
            user_message=query,
            system_instruction=ZERO_FLUFF_PROMPT,
            kb_chunks=kb_chunks,
            is_urgent=urgency,
            frustration_score=frustration,
            intent=jev_intent,
            temperature=temp,
            max_tokens=effective_budget,
            use_jev_intent_tag=use_tag,
        )
        return {
            "content": res.content,
            "latency_ms": res.telemetry.latency_ms if res.telemetry else res.latency_ms,
            "in_tokens": res.telemetry.input_tokens if res.telemetry else 0,
            "out_tokens": res.telemetry.output_tokens if res.telemetry else 0,
            "tps": res.telemetry.tokens_per_sec if res.telemetry else 0.0,
        }

    # --- Medium Local (Qwen 2.5 7B 4-bit) ---
    elif stream_id in ["S4_7B_Ctrl", "S5_7B_Regex", "S6_7B_Jev"]:
        effective_budget = 180 if "Ctrl" in stream_id else token_budget
        res = gen_7b.generate(
            user_message=query,
            system_instruction=ZERO_FLUFF_PROMPT,
            kb_chunks=kb_chunks,
            is_urgent=urgency,
            frustration_score=frustration,
            intent=jev_intent,
            temperature=temp,
            max_tokens=effective_budget,
            use_jev_intent_tag=use_tag,
        )
        return {
            "content": res.content,
            "latency_ms": res.telemetry.latency_ms if res.telemetry else res.latency_ms,
            "in_tokens": res.telemetry.input_tokens if res.telemetry else 0,
            "out_tokens": res.telemetry.output_tokens if res.telemetry else 0,
            "tps": res.telemetry.tokens_per_sec if res.telemetry else 0.0,
        }

    # --- Frontier Cloud (Gemini 3.8 Flash) ---
    elif stream_id in ["S7_Gemini_Ctrl", "S8_Gemini_Regex", "S9_Gemini_Jev"]:
        effective_budget = 512 if "Ctrl" in stream_id else 220
        res = await gemini.generate_async(
            user_message=query,
            system_instruction=ZERO_FLUFF_PROMPT,
            kb_chunks=kb_chunks,
            temperature=temp,
            intent=jev_intent,
            is_urgent=urgency,
            frustration_score=frustration,
            use_jev_intent_tag=use_tag,
            max_tokens=effective_budget,
        )
        return {
            "content": res.content,
            "latency_ms": res.telemetry.latency_ms if res.telemetry else 0.0,
            "in_tokens": res.telemetry.input_tokens if res.telemetry else 0,
            "out_tokens": res.telemetry.output_tokens if res.telemetry else 0,
            "tps": res.telemetry.tokens_per_sec if res.telemetry else 0.0,
        }

    return {}


# -----------------------------------------------------------------------------
# 3. BENCHMARK HARNESS RUNNER
# -----------------------------------------------------------------------------
async def run_expanded_benchmark(shots: int = 2):
    print("=" * 115, flush=True)
    print("EXPANDED MULTI-DOMAIN FACTORIAL BENCHMARK: 4 DOMAINS x 3 COMPLEXITIES x 3 DENSITIES x 9 STREAMS", flush=True)
    print(f"Protocol: Zero-Fluff Prompt Invariant | Shot 0 Discarded Warm-up | {shots} Measured Repetitions (Temp = 0.0)", flush=True)
    print("Hardware: NVIDIA RTX 4070 Laptop GPU + Google AI Studio + TypeSafe AI Jev Cloud", flush=True)
    print("=" * 115, flush=True)

    # Initialize components
    classifier = JevClassifier()
    gen_17b = QwenGenerator(ollama_model="qwen3:1.7b")
    gen_7b = QwenGenerator(ollama_model="qwen2.5:7b")
    gemini = GeminiClient()

    jev_cache: Dict[str, Any] = {}

    # Define the 9 streams
    streams_by_phase = [
        ("Phase 1: Small Local SLM (Qwen 1.7B)", [
            ("S1_1.7B_Ctrl", "1.7B Control (Raw RAG)"),
            ("S2_1.7B_Regex", "1.7B + Regex Trie"),
            ("S3_1.7B_Jev", "1.7B + TypeSafe AI Jev"),
        ]),
        ("Phase 2: Medium Local MLM (Qwen 2.5 7B 4-bit)", [
            ("S4_7B_Ctrl", "7B Control (Raw RAG)"),
            ("S5_7B_Regex", "7B + Regex Trie"),
            ("S6_7B_Jev", "7B + TypeSafe AI Jev"),
        ]),
        ("Phase 3: Frontier Cloud LLM (Gemini 3.8 Flash)", [
            ("S7_Gemini_Ctrl", "Gemini Control (Raw RAG)"),
            ("S8_Gemini_Regex", "Gemini + Regex Trie"),
            ("S9_Gemini_Jev", "Gemini + TypeSafe AI Jev"),
        ]),
    ]

    # Pre-build scenario catalog (4 domains x 3 complexities x 3 KB densities = 36 scenarios)
    scenario_catalog = []
    for d_name, d_data in DOMAINS.items():
        for q_comp, q_text in d_data["queries"].items():
            for kb_dens, kb_chunks in d_data["kb"].items():
                targets = d_data["targets"][kb_dens]
                scenario_catalog.append({
                    "key": f"{d_name}_{q_comp}_{kb_dens}",
                    "domain": d_name,
                    "complexity": q_comp,
                    "kb_density": kb_dens,
                    "query": q_text,
                    "kb": kb_chunks,
                    "targets": targets,
                })

    out_file = "expanded_multidomain_results.json"
    results_store: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                results_store = json.load(f)
        except Exception:
            results_store = {}

    for sc in scenario_catalog:
        k = sc["key"]
        if k not in results_store:
            results_store[k] = {
                "domain": sc["domain"],
                "complexity": sc["complexity"],
                "kb_density": sc["kb_density"],
                "query": sc["query"],
                "streams": {},
            }

    # Execute Phase by Phase to keep GPU memory resident
    total_scenarios = len(scenario_catalog)

    for phase_name, stream_group in streams_by_phase:
        print(f"\n{'=' * 115}", flush=True)
        print(f"STARTING {phase_name}", flush=True)
        print(f"{'=' * 115}", flush=True)

        for s_id, s_desc in stream_group:
            print(f"\n--- Stream: [{s_id}] ({s_desc}) ---", flush=True)

            # Warm up once per stream to ensure model resident
            first_sc = scenario_catalog[0]
            try:
                _ = await execute_stream_turn(
                    s_id, first_sc["query"], first_sc["kb"], classifier, gen_17b, gen_7b, gemini, jev_cache
                )
            except Exception as e:
                print(f"  [Warm-up Note: {e}]", flush=True)

            for sc_idx, sc in enumerate(scenario_catalog, 1):
                sc_key = sc["key"]
                q_text = sc["query"]
                kb_data = sc["kb"]
                targets = sc["targets"]

                print(f"  [{sc_idx:02d}/{total_scenarios}] {sc_key:<24}", end="", flush=True)

                latencies: List[float] = []
                in_toks: List[int] = []
                out_toks: List[int] = []
                tps_vals: List[float] = []
                grounding_vals: List[float] = []
                snippet: str = ""

                for shot in range(1, shots + 1):
                    try:
                        out = await execute_stream_turn(
                            s_id, q_text, kb_data, classifier, gen_17b, gen_7b, gemini, jev_cache
                        )
                        lat = out.get("latency_ms", 0.0)
                        latencies.append(lat)
                        in_toks.append(out.get("in_tokens", 0))
                        out_toks.append(out.get("out_tokens", 0))
                        tps_vals.append(out.get("tps", 0.0))
                        c = out.get("content", "")
                        g_val = score_grounding(c, targets)
                        grounding_vals.append(g_val)
                        snippet = c
                    except Exception as e:
                        print(f" [Err: {e}]", end="", flush=True)

                if latencies:
                    med_lat = round(statistics.median(latencies), 1)
                    mean_lat = round(statistics.mean(latencies), 1)
                    std_lat = round(statistics.stdev(latencies), 1) if len(latencies) > 1 else 0.0
                    avg_out = round(statistics.mean(out_toks), 1)
                    avg_in = round(statistics.mean(in_toks), 1)
                    avg_tps = round(statistics.mean(tps_vals), 1)
                    avg_ground = round(statistics.mean(grounding_vals), 1)
                    efficiency = round(avg_ground / max(avg_out, 1.0), 3)

                    results_store[sc_key]["streams"][s_id] = {
                        "median_ms": med_lat,
                        "mean_ms": mean_lat,
                        "std_ms": std_lat,
                        "in_tokens": avg_in,
                        "out_tokens": avg_out,
                        "tps": avg_tps,
                        "grounding_pct": avg_ground,
                        "token_efficiency": efficiency,
                        "sample_snippet": snippet[:140].replace("\n", " ").strip(),
                    }

                    print(
                        f" -> Med: {med_lat:>6.1f} ms | Out: {avg_out:>4.0f} tok | "
                        f"Grd: {avg_ground:>5.1f}% | Eff: {efficiency:>5.3f}",
                        flush=True,
                    )

            # Checkpoint save after each complete stream
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results_store, f, indent=2)
            print(f"  --> Checkpointed {s_id} results to {out_file}", flush=True)

    print("\n" + "=" * 115, flush=True)
    print(f"ALL 324 EXPERIMENTAL RUNS COMPLETED. Results written to {out_file}", flush=True)
    print("=" * 115, flush=True)


if __name__ == "__main__":
    asyncio.run(run_expanded_benchmark(shots=2))
