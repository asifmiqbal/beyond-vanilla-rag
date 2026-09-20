"""Jev System One Intent Classifier Service.

Supports:
1. Vercel AI Gateway (https://ai-gateway.vercel.sh/typesafe/v1/systemone)
2. Local simple-jev HTTP server (http://localhost:8000/v1/classifier)
3. High-fidelity deterministic offline Jev heuristic emulator (matching Studio spec)
"""

from __future__ import annotations

import math
import os
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class IntentChoice(str, Enum):
    BILLING_INQUIRY = "billing_inquiry"
    NETWORK_CONNECTIVITY = "network_connectivity"
    ROAMING_PASSES = "roaming_passes"
    FIBER_BROADBAND = "fiber_broadband"
    ACCOUNT_SECURITY = "account_security"
    PLAN_UPGRADE = "plan_upgrade"
    HUMAN_ESCALATION = "human_escalation"
    OTHER_OR_UNCLEAR = "other_or_unclear"


INTENT_QUEUES: Dict[str, str] = {
    IntentChoice.BILLING_INQUIRY.value: "Billing & Accounts Queue",
    IntentChoice.NETWORK_CONNECTIVITY.value: "Wireless Network Ops",
    IntentChoice.ROAMING_PASSES.value: "Global Roaming Desk",
    IntentChoice.FIBER_BROADBAND.value: "Broadband & Fiber NOC",
    IntentChoice.ACCOUNT_SECURITY.value: "Fraud & Security Tier-2",
    IntentChoice.PLAN_UPGRADE.value: "Retention & Sales Desk",
    IntentChoice.HUMAN_ESCALATION.value: "Senior Human Queue",
    IntentChoice.OTHER_OR_UNCLEAR.value: "General Triage Fallback",
}


class JevDistributionItem(BaseModel):
    choice: str
    probability: float


class IntentHierarchy(BaseModel):
    l1: str = Field(..., description="Top-level domain e.g. Billing & Payments")
    l2: str = Field(..., description="Functional sub-category e.g. Invoice Dispute")
    l3: str = Field(..., description="Actionable leaf intent e.g. double_billing_deduction")
    full_path: str = Field(..., description="Dot-separated path e.g. billing.invoice_dispute.double_billing_deduction")
    l1_confidence: float
    l2_confidence: float
    l3_confidence: float


class JevResult(BaseModel):
    selected_choice: str
    confidence: float
    distribution: List[JevDistributionItem]
    hierarchy: IntentHierarchy
    is_urgent: float = Field(..., description="Noul urgency probability [0.0 - 1.0]")
    frustration_score: float = Field(..., description="Continuous frustration score [0.0 - 1.0]")
    dispatch_queue: str
    dispatch_reason: str
    latency_ms: float
    engine_used: str
    trace_logs: List[str] = Field(default_factory=list)
    is_escalated: bool = False
    is_emergency: bool = False
    is_fallback: bool = False


class JevClassifier:
    """Jev System One Classifier evaluating Choice, Noul (urgency), and Score (frustration)."""

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        urgency_threshold: float = 0.75,
        frustration_threshold: float = 0.80,
        gateway_url: Optional[str] = None,
        api_key: Optional[str] = None,
        simple_jev_url: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.urgency_threshold = urgency_threshold
        self.frustration_threshold = frustration_threshold
        self.typesafe_api_url = os.getenv(
            "TYPESAFE_API_URL", "https://api.typesafe.ai/v1/systemone"
        )
        self.gateway_url = gateway_url or os.getenv(
            "VERCEL_AI_GATEWAY_URL",
            os.getenv("JEV_GATEWAY_URL", "https://ai-gateway.vercel.sh/typesafe/v1/systemone"),
        )
        self.api_key = (
            api_key
            or os.getenv("TYPESAFE_API_KEY")
            or os.getenv("JEV_API_KEY")
            or os.getenv("VERCEL_AI_GATEWAY_API_KEY")
            or os.getenv("VERCEL_API_KEY")
        )
        self.simple_jev_url = simple_jev_url or os.getenv(
            "SIMPLE_JEV_URL", "http://localhost:8000/v1/classifier"
        )
        self.jev_mode = os.getenv("JEV_MODE", "auto").lower()
        self._gateway_failed = False
        self._simple_jev_failed = False
        self.choices = [c.value for c in IntentChoice]

    def classify(self, text: str) -> JevResult:
        """Evaluate text turn against Choice, Noul, and Score primitives with deterministic guardrails."""
        start_time = time.perf_counter()
        trace_logs: List[str] = []

        raw_result: Optional[Dict[str, Any]] = None
        engine_used = "offline_heuristic_jev"

        # 1. Attempt official TypeSafe AI direct endpoint or Vercel Gateway if API key is present
        if self.api_key and not self._gateway_failed and self.jev_mode not in ["offline", "heuristic", "local"]:
            try:
                # If key is a direct TypeSafe key or default auto/typesafe mode, query api.typesafe.ai
                if self.jev_mode in ["typesafe", "cloud", "api", "auto"] or "api.typesafe.ai" in self.typesafe_api_url:
                    raw_result = self._query_typesafe_ai_direct(text)
                    engine_used = "typesafe_ai_cloud"
                else:
                    raw_result = self._query_gateway(text)
                    engine_used = "vercel_ai_gateway_jev"
            except Exception as e:
                self._gateway_failed = True
                trace_logs.append(f"[TYPESAFE CLOUD FALLBACK] TypeSafe AI Cloud failed ({e}). Switched to zero-latency local Jev fast-path.")

        # 2. Attempt simple-jev local server only if explicitly configured
        if raw_result is None and self.jev_mode in ["simple_jev", "local_server"] and not self._simple_jev_failed:
            try:
                raw_result = self._query_simple_jev(text)
                engine_used = "simple_jev_local"
            except Exception:
                self._simple_jev_failed = True

        # 3. Fallback to offline deterministic Jev heuristic emulator
        if raw_result is None:
            raw_result = self._evaluate_offline_heuristic(text)
            engine_used = "offline_deterministic_jev"

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Apply Deterministic Guardrails
        guardrail_output = self._apply_guardrails(raw_result, trace_logs)

        # Resolve Hierarchical L1, L2, L3 Intent
        hierarchy = self._resolve_hierarchy(
            text=text,
            top_choice=guardrail_output["selected_choice"],
            confidence=guardrail_output["confidence"],
        )

        return JevResult(
            selected_choice=guardrail_output["selected_choice"],
            confidence=guardrail_output["confidence"],
            distribution=raw_result["distribution"],
            hierarchy=hierarchy,
            is_urgent=raw_result["noulUrgency"],
            frustration_score=raw_result["frustrationScore"],
            dispatch_queue=guardrail_output["dispatch_queue"],
            dispatch_reason=guardrail_output["dispatch_reason"],
            latency_ms=round(latency_ms, 2),
            engine_used=engine_used,
            trace_logs=trace_logs,
            is_escalated=guardrail_output["is_escalated"],
            is_emergency=guardrail_output["is_emergency"],
            is_fallback=guardrail_output["is_fallback"],
        )

    def _resolve_hierarchy(self, text: str, top_choice: str, confidence: float) -> IntentHierarchy:
        """Evaluates 3-tier hierarchical intent (L1 Domain -> L2 Functional Subcategory -> L3 Actionable Leaf)."""
        lower = text.lower()
        l1 = top_choice
        l2 = "general_support"
        l3 = "standard_inquiry"

        if top_choice == IntentChoice.BILLING_INQUIRY.value:
            l1 = "Billing & Payments"
            if re.search(r"\b(twice|double|wrong|stolen|dispute|refund|overcharge)\b", lower):
                l2 = "Invoice Dispute"
                l3 = "double_billing_deduction" if "twice" in lower or "double" in lower else "unrecognized_roaming_charge" if "roam" in lower else "disputed_line_item_fee"
            elif re.search(r"\b(autopay|card|bank|mastercard|visa|direct debit|pay)\b", lower):
                l2 = "Payment Methods & AutoPay"
                l3 = "autopay_discount_setup" if "autopay" in lower else "declined_card_update"
            elif re.search(r"\b(prorated|cycle|statement|date)\b", lower):
                l2 = "Billing Cycle & Proration"
                l3 = "prorated_charge_calculation"
            else:
                l2 = "Account Balance & Invoicing"
                l3 = "standard_balance_inquiry"

        elif top_choice == IntentChoice.NETWORK_CONNECTIVITY.value:
            l1 = "Mobile Network & Wireless"
            if re.search(r"\b(apn|pdp|attach|volte|settings)\b", lower):
                l2 = "APN & Data Configuration"
                l3 = "manual_5g_apn_setup"
            elif re.search(r"\b(outage|down|blackout|tower|maintenance)\b", lower):
                l2 = "Cellular Outage Incident"
                l3 = "cell_tower_degraded_service"
            else:
                l2 = "Signal & Coverage"
                l3 = "weak_5g_nr_indoor_coverage"

        elif top_choice == IntentChoice.FIBER_BROADBAND.value:
            l1 = "Fiber FTTH & Home Broadband"
            if re.search(r"\b(los|red light|optical|cut|pon)\b", lower):
                l2 = "Optical Link Hardware Failure"
                l3 = "ont_los_red_blinking_optical_cut"
            elif re.search(r"\b(wifi|wi-fi|router|mesh|node)\b", lower):
                l2 = "Wi-Fi & Home Networking"
                l3 = "mesh_wifi6_node_pairing"
            elif re.search(r"\b(technician|truck|dispatch|visit)\b", lower):
                l2 = "Field Operations Dispatch"
                l3 = "technician_truck_roll_booking"
            else:
                l2 = "GPON Broadband Performance"
                l3 = "broadband_throughput_diagnostic"

        elif top_choice == IntentChoice.ROAMING_PASSES.value:
            l1 = "Global Roaming & Travel"
            if re.search(r"\b(zone 1|zone 2|zone 3|zone 4|day pass|pass|bundle)\b", lower):
                l2 = "Roaming Pass Activation"
                l3 = "zone_2_americas_bundle" if "zone 2" in lower or "americas" in lower else "zone_1_eu_day_pass" if "zone 1" in lower or "europe" in lower else "zone_4_inflight_maritime_pass"
            elif re.search(r"\b(throttle|fup|slow|cap)\b", lower):
                l2 = "Fair Usage Policy (FUP)"
                l3 = "fup_speed_throttle_reset"
            else:
                l2 = "Carrier Steering & Handshake"
                l3 = "foreign_carrier_auth_error_403"

        elif top_choice == IntentChoice.ACCOUNT_SECURITY.value:
            l1 = "SIM & Identity Security"
            if re.search(r"\b(esim|e-sim|qr|eid)\b", lower):
                l2 = "eSIM Provisioning"
                l3 = "qr_code_activation_fail"
            elif re.search(r"\b(swap|fraud|stolen|hacked|lockdown)\b", lower):
                l2 = "SIM Swap Fraud Protection"
                l3 = "unauthorized_sim_swap_cooling_hold"
            elif re.search(r"\b(puk|pin|code)\b", lower):
                l2 = "Security Credentials"
                l3 = "puk_code_retrieval"
            else:
                l2 = "KYC Biometric Verification"
                l3 = "biometric_identity_validation"

        elif top_choice == IntentChoice.PLAN_UPGRADE.value:
            l1 = "Plan Tiers & Subscriptions"
            if re.search(r"\b(business|enterprise|pool|lines|corporate)\b", lower):
                l2 = "Commercial Enterprise Pooling"
                l3 = "enterprise_volume_pooling_negotiation"
            elif re.search(r"\b(cancel|port|leave|competitor)\b", lower):
                l2 = "Retention & Churn Prevention"
                l3 = "loyalty_contract_renewal_discount"
            else:
                l2 = "Tier Migration"
                l3 = "5g_infinite_tier_upgrade"

        elif top_choice == IntentChoice.HUMAN_ESCALATION.value:
            l1 = "Customer Care Escalations"
            if re.search(r"\b(supervisor|manager|director)\b", lower):
                l2 = "Managerial Escalation"
                l3 = "senior_operations_supervisor_handoff"
            else:
                l2 = "Specialist Agent Dispatch"
                l3 = "tier2_specialist_live_transfer"

        else:
            l1 = "General Triage"
            l2 = "Ambiguous Inquiries"
            l3 = "unclassified_customer_inquiry"

        l2_conf = round(confidence * 0.96, 4)
        l3_conf = round(confidence * 0.92, 4)
        full_path = f"{l1.lower().replace(' ', '_').replace('&', 'and')}.{l2.lower().replace(' ', '_').replace('&', 'and')}.{l3}"

        return IntentHierarchy(
            l1=l1,
            l2=l2,
            l3=l3,
            full_path=full_path,
            l1_confidence=round(confidence, 4),
            l2_confidence=l2_conf,
            l3_confidence=l3_conf,
        )

    def _query_typesafe_ai_direct(self, text: str) -> Dict[str, Any]:
        """Query official TypeSafe AI evaluation endpoint directly (https://api.typesafe.ai/v1/systemone)."""
        url = self.typesafe_api_url
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "state": text,
            "model": "jev-latest",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Which department or issue category best describes this customer request?",
                    "criteria": {
                        "billing_inquiry": "Invoices, charges, disputed fees, payment methods, refunds",
                        "network_connectivity": "Signal drops, 5G reception, cellular APN configs",
                        "roaming_passes": "International travel packs, foreign roaming rates",
                        "fiber_broadband": "Home fiber optic, ONT LOS red light, Wi-Fi 6 router",
                        "account_security": "eSIM, SIM swap, PUK codes, credentials",
                        "plan_upgrade": "Tier changes, family pooling, retention offers",
                        "human_escalation": "Supervisor demands, manager complaints, human agent requests",
                        "other_or_unclear": "Ambiguous chatter, greeting, unclassifiable inquiries",
                    },
                },
                "is_urgent": {
                    "type": "noul",
                    "instructions": "Does this request convey an urgent outage, severe service disruption, or emergency?",
                    "criteria": {
                        "true": "Customer has no service, severed line, or urgent outage requiring immediate response",
                        "false": "Standard non-emergency inquiry or general question",
                    },
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated or agitated is the customer?",
                    "criteria": [
                        "Calm, neutral, or constructive customer tone",
                        "Mildly annoyed or impatient customer tone",
                        "Frustrated, agitated, angry, or demanding customer tone",
                    ],
                },
            },
        }
        with httpx.Client(timeout=8.0) as client:
            res = client.post(url, json=payload, headers=headers)
            if res.status_code == 401:
                raise PermissionError("TypeSafe AI 401 Unauthorized: Invalid API Key. Check TYPESAFE_API_KEY in .env.")
            res.raise_for_status()
            data = res.json()
            answers = data.get("answers", {})

            intent_ans = answers.get("intent", {})
            selected = intent_ans.get("choice", "other_or_unclear")
            confidence = intent_ans.get("confidence", 0.90)
            probs = intent_ans.get("probabilities", {})

            distribution = [
                JevDistributionItem(choice=k, probability=round(v, 4))
                for k, v in probs.items()
            ] if probs else [JevDistributionItem(choice=selected, probability=confidence)]
            distribution.sort(key=lambda x: x.probability, reverse=True)

            noul_val = float(answers.get("is_urgent", {}).get("noul", 0.0))
            raw_score = float(answers.get("frustration", {}).get("score", 0.0))
            # TypeSafe score returns float index along criteria levels (0 to 2). Normalize to [0.0, 1.0]
            score_val = round(min(1.0, max(0.0, raw_score / 2.0)), 2) if raw_score > 1.0 else round(raw_score, 2)

            return {
                "selectedChoice": selected,
                "confidence": confidence,
                "distribution": distribution,
                "noulUrgency": noul_val,
                "frustrationScore": score_val,
            }

    def _query_gateway(self, text: str) -> Dict[str, Any]:
        """Query Vercel AI Gateway endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "typesafe-ai/jev",
            "state": text,
            "questions": {
                "intent": {
                    "type": "choice",
                    "choices": self.choices,
                    "criteria": {
                        "billing_inquiry": "Invoices, charges, disputed fees, payment methods, refunds",
                        "network_connectivity": "Signal drops, 5G reception, cellular APN configs",
                        "roaming_passes": "International travel packs, foreign roaming rates",
                        "fiber_broadband": "Home fiber optic, ONT LOS red light, Wi-Fi 6 router",
                        "account_security": "eSIM, SIM swap, PUK codes, credentials",
                        "plan_upgrade": "Tier changes, family pooling, retention offers",
                        "human_escalation": "Supervisor demands, manager complaints, human agent requests",
                        "other_or_unclear": "Ambiguous chatter, greeting, unclassifiable inquiries",
                    },
                },
                "is_urgent": {
                    "type": "noul",
                    "criteria": {},
                },
                "frustration": {
                    "type": "score",
                    "criteria": [
                        "Calm, neutral or constructive customer tone",
                        "Frustrated, agitated, angry or demanding customer tone",
                    ],
                },
            },
        }
        with httpx.Client(timeout=6.0) as client:
            res = client.post(self.gateway_url, json=payload, headers=headers)
            if res.status_code == 403:
                body = res.json()
                msg = body.get("error", {}).get("message") or res.text
                raise PermissionError(f"Vercel AI Gateway 403: {msg}")
            res.raise_for_status()
            data = res.json()
            q = data.get("questions", {})
            return {
                "selectedChoice": q["intent"]["selected"],
                "confidence": q["intent"]["confidence"],
                "distribution": [
                    JevDistributionItem(choice=k, probability=v)
                    for k, v in q["intent"].get("distribution", {}).items()
                ],
                "noulUrgency": q.get("is_urgent", {}).get("noul", 0.0),
                "frustrationScore": q.get("frustration", {}).get("score", 0.0),
            }

    def _query_simple_jev(self, text: str) -> Dict[str, Any]:
        """Query local simple-jev endpoint."""
        payload = {
            "model": "jev-1.13.0",
            "state": text,
            "questions": {
                "intent": {"type": "choice", "choices": self.choices},
                "is_urgent": {"type": "noul"},
                "frustration": {"type": "score"},
            },
        }
        with httpx.Client(timeout=1.5) as client:
            res = client.post(self.simple_jev_url, json=payload)
            res.raise_for_status()
            data = res.json()
            q = data.get("questions", {})
            return {
                "selectedChoice": q["intent"]["selected"],
                "confidence": q["intent"]["confidence"],
                "distribution": [
                    JevDistributionItem(choice=item["choice"], probability=item["probability"])
                    for item in q["intent"].get("distribution", [])
                ],
                "noulUrgency": q.get("is_urgent", {}).get("noul", 0.0),
                "frustrationScore": q.get("frustration", {}).get("score", 0.0),
            }

    def _evaluate_offline_heuristic(self, text: str) -> Dict[str, Any]:
        """Deterministic offline Jev emulator calculating normalized next-token logits over choices."""
        lower = text.lower()

        # Prior base scores
        raw_scores: Dict[str, float] = {c: 0.05 for c in self.choices}

        # 1. Billing Inquiry
        bill_matches = len(re.findall(r"\b(bill|billing|charge|charged|charging|invoice|payment|paid|refund|overcharge|cost|rate|pricing|card|mastercard|visa|prorated|fee|direct debit)\b", lower))
        if bill_matches > 0:
            raw_scores[IntentChoice.BILLING_INQUIRY.value] += 3.2 + min(bill_matches * 0.4, 1.2)

        # 2. Network Connectivity
        net_matches = len(re.findall(r"\b(5g|4g|lte|signal|reception|cell|tower|bars|dropped|drop|disconnect|apn|network|slow data|attach|volte|packet loss|latency)\b", lower))
        if net_matches > 0:
            raw_scores[IntentChoice.NETWORK_CONNECTIVITY.value] += 3.2 + min(net_matches * 0.4, 1.2)

        # 3. Roaming Passes
        roam_matches = len(re.findall(r"\b(roam|roaming|overseas|abroad|international|flight|maritime|zone 1|zone 2|zone 3|zone 4|fup|throttle|day pass|travel)\b", lower))
        if roam_matches > 0:
            raw_scores[IntentChoice.ROAMING_PASSES.value] += 3.3 + min(roam_matches * 0.4, 1.2)

        # 4. Fiber Broadband
        fiber_matches = len(re.findall(r"\b(fiber|fibre|ftth|ont|modem|router|wifi|wi-fi|gpon|xgs-pon|broadband|los|pon|red light|ethernet|bridge mode)\b", lower))
        if fiber_matches > 0:
            raw_scores[IntentChoice.FIBER_BROADBAND.value] += 3.3 + min(fiber_matches * 0.4, 1.2)

        # 5. Account Security
        sec_matches = len(re.findall(r"\b(sim|esim|e-sim|qr code|swap|eid|puk|pin|kyc|otp|hacked|stolen|imei|verification|identity|passport|password|reset)\b", lower))
        if sec_matches > 0:
            raw_scores[IntentChoice.ACCOUNT_SECURITY.value] += 3.3 + min(sec_matches * 0.4, 1.2)

        # 6. Plan Upgrade
        plan_matches = len(re.findall(r"\b(upgrade|downgrade|change plan|new plan|flexipay|infinite|postpaid|prepaid|contract|renew|add line|family pool|unlimited)\b", lower))
        if plan_matches > 0:
            raw_scores[IntentChoice.PLAN_UPGRADE.value] += 3.2 + min(plan_matches * 0.4, 1.2)

        # 7. Human Escalation
        esc_matches = len(re.findall(r"\b(human|agent|operator|manager|supervisor|representative|person|transfer me|speak to someone|stop the bot|real person)\b", lower))
        if esc_matches > 0:
            raw_scores[IntentChoice.HUMAN_ESCALATION.value] += 3.6 + min(esc_matches * 0.4, 1.2)

        # 8. Ambiguous / vague check
        words = text.strip().split()
        if len(words) < 5 and not re.search(
            r"(bill|refund|broken|down|outage|sim|esim|roam|fiber|apn|wifi|signal)", lower
        ):
            raw_scores[IntentChoice.OTHER_OR_UNCLEAR.value] += 2.0

        # Compute Softmax
        exp_vals = {k: math.exp(v) for k, v in raw_scores.items()}
        exp_sum = sum(exp_vals.values())
        distribution = [
            JevDistributionItem(choice=k, probability=round(v / exp_sum, 4))
            for k, v in exp_vals.items()
        ]
        distribution.sort(key=lambda x: x.probability, reverse=True)

        top_choice = distribution[0].choice
        top_prob = distribution[0].probability

        # Noul (Urgency calculation)
        urgency = 0.08
        if re.search(
            r"\b(urgent|urgently|emergency|asap|critical|outage|fiber cut|los red|no service|total blackout|stranded|hospital|danger)\b",
            lower,
        ):
            urgency += 0.72
        if "!" in text and re.search(r"\b(urgent|emergency|outage|down|broken)\b", lower):
            urgency += 0.15
        urgency = round(min(0.99, max(0.01, urgency)), 2)

        # Frustration Score calculation
        frustration = 0.06
        if re.search(
            r"\b(furious|ridiculous|unacceptable|useless|stolen|demand|refund|worst|terrible|horrible|lawsuit|attorney|sue|fed up|scam|right now|twice)\b",
            lower,
        ):
            frustration += 0.76
        if re.search(r"\b(again|repeatedly|third time|second time|waiting for hours)\b", lower):
            frustration += 0.25
        if text.isupper() and len(text) > 12:
            frustration += 0.22
        frustration = round(min(0.99, max(0.01, frustration)), 2)

        return {
            "selectedChoice": top_choice,
            "confidence": top_prob,
            "distribution": distribution,
            "noulUrgency": urgency,
            "frustrationScore": frustration,
        }

    def _apply_guardrails(self, raw: Dict[str, Any], logs: List[str]) -> Dict[str, Any]:
        """Apply deterministic routing rules with threshold gating."""
        top_choice = raw["selectedChoice"]
        confidence = raw["confidence"]
        urgency = raw["noulUrgency"]
        frustration = raw["frustrationScore"]

        logs.append(f"[T0] Received turn state: eval_questions(choice, noul, score)")
        logs.append(f"[T1] Argmax choice: '{top_choice}' with p = {confidence * 100:.1f}%")

        is_escalated = False
        is_emergency = False
        is_fallback = False

        # Guardrail 1: Acute Urgency Override (Noul >= 0.75)
        if urgency >= self.urgency_threshold:
            dispatch = "Tier-2 Emergency Queue"
            reason = f"Urgency Override: Noul {urgency:.2f} >= {self.urgency_threshold:.2f}"
            is_emergency = True
            is_escalated = True
            logs.append(f"[GUARDRAIL HIT] {reason} -> Dispatched to Emergency Queue.")

        # Guardrail 2: High Frustration or Explicit Escalation (Score >= 0.80)
        elif frustration >= self.frustration_threshold or top_choice == IntentChoice.HUMAN_ESCALATION.value:
            dispatch = "Senior Human Queue"
            reason = (
                f"Frustration Escalation: Score {frustration:.2f} >= {self.frustration_threshold:.2f}"
                if frustration >= self.frustration_threshold
                else "Explicit customer demand for human agent"
            )
            is_escalated = True
            logs.append(f"[GUARDRAIL HIT] {reason} -> Automatic Human Handover.")

        # Guardrail 3: Low Confidence (p < tau) or other_or_unclear Escape Hatch
        elif confidence < self.confidence_threshold or top_choice == IntentChoice.OTHER_OR_UNCLEAR.value:
            dispatch = INTENT_QUEUES[IntentChoice.OTHER_OR_UNCLEAR.value]
            reason = (
                f"Low Confidence: p={confidence * 100:.1f}% < threshold {self.confidence_threshold * 100:.0f}%"
                if confidence < self.confidence_threshold
                else "Fallback escape hatch triggered"
            )
            is_fallback = True
            logs.append(f"[FALLBACK TRIGGERED] {reason} -> Routed to General Triage.")

        # Guardrail 4: Normal Deterministic Route
        else:
            dispatch = INTENT_QUEUES.get(top_choice, "General Support Queue")
            reason = f"High Confidence intent match (p={confidence * 100:.1f}% >= tau)"
            logs.append(f"[ROUTED] Dispatched to '{dispatch}' based on categorical certainty.")

        return {
            "selected_choice": top_choice,
            "confidence": confidence,
            "dispatch_queue": dispatch,
            "dispatch_reason": reason,
            "is_escalated": is_escalated,
            "is_emergency": is_emergency,
            "is_fallback": is_fallback,
        }
