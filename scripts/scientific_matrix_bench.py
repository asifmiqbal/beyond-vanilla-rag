"""Scientific Matrix Benchmark: 3x3 Factorial Matrix across 6 Model Architectures.

Variables:
- Independent Variable 1: Model Architecture (6 levels)
  1. Jev + Qwen 1.7B (Local)
  2. Control Qwen 1.7B (Local Baseline)
  3. Jev + Qwen 2.5 7B (Local 4-bit)
  4. Control Qwen 2.5 7B (Local 4-bit Baseline)
  5. Jev + Gemini 3.8 Flash (Frontier Cloud)
  6. Control Gemini 3.8 Flash (Frontier Cloud Baseline)

- Independent Variable 2: Query Length (3 levels: Short, Mid, Long)
- Independent Variable 3: Knowledge Base Grounding Density (3 levels: Short, Mid, Long)

Statistical Protocol:
- 1 Warm-up Shot (discarded to eliminate cold-load artifacts)
- 3 Measured Repetitions (N = 3) with greedy decoding (temperature = 0.0)
- Metrics: Median Latency (ms), Mean (ms), Tokens (In/Out), Decode Speed (t/s), Grounding Recall (%)
"""

from __future__ import annotations

import asyncio
import json
import os
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

# Standard Base Prompt across all 6 streams
STANDARDIZED_BASE_PROMPT = (
    "You are a helpful customer service assistant for Apex Telecom. "
    "Provide direct, concise, and professional answers formatted in clean Markdown with bullet points. "
    "CRITICAL: Output ONLY the final customer-facing response. "
    "Do NOT include internal reasoning or <think> tags. Speak directly to the customer."
)

# -----------------------------------------------------------------------------
# 1. 3x3 SCENARIOS (Query Length x KB Density)
# -----------------------------------------------------------------------------
QUERIES = {
    "Short": "Fiber line cut outside, red LOS blinking.",
    "Mid": "My fiber optic drop cable was severed by utility workers outside my house. The Optical Terminal has a blinking red LOS light and internet is completely down.",
    "Long": "I am writing regarding my fiber internet service which went completely dead about 45 minutes ago after municipal excavation crews dug up the sidewalk in front of my driveway. They visibly severed the thick black fiber drop cable. On my ONT terminal inside the garage, the Power light is solid green, but the LOS indicator is rapidly blinking red. I run a home business and urgently need this repaired or escalated to an outside plant field technician. My account is in good standing.",
}

KB_CHUNKS = {
    "Short": [
        {
            "doc_id": "KB-DOC-1",
            "title": "GPON Drop Restoration",
            "plan_code": "TELCO-XGSPON-SYMM-10G",
            "regional_apn": "internet.telco.us-east",
            "sla": "24-Hour Resolution SLA",
            "summary": "Outside drop cut requires outside-plant splice technician dispatch. 24-hour resolution SLA.",
        }
    ],
    "Mid": [
        {
            "doc_id": "KB-DOC-1",
            "title": "GPON Drop Restoration",
            "plan_code": "TELCO-XGSPON-SYMM-10G",
            "regional_apn": "internet.telco.us-east",
            "sla": "24-Hour Resolution SLA",
            "summary": "Physical fiber severance outside customer premises requires emergency outside plant splice crew dispatch within 4 hours. Standard resolution SLA is 24 hours.",
        },
        {
            "doc_id": "KB-DOC-2",
            "title": "Service Credit Policy",
            "plan_code": "TELCO-XGSPON-SYMM-10G",
            "regional_apn": "internet.telco.us-east",
            "sla": "Credit Adjustment",
            "summary": "Automatic SLA credit adjustment applied to next billing statement if complete service outage exceeds 24-hour window from initial report timestamp.",
        },
    ],
    "Long": [
        {
            "doc_id": "KB-DOC-1",
            "title": "Outside Plant Fiber Restoration SOP",
            "plan_code": "TELCO-XGSPON-SYMM-10G",
            "regional_apn": "internet.telco.us-east",
            "sla": "24-Hour Resolution SLA",
            "summary": "Physical drop severance requires emergency outside plant technician dispatch under ticket code GPON-OUTAGE-L3. Standard repair SLA is 24 hours. Hardware note: leave ONT powered on to allow automated loopback optical reflection diagnostics from central office OLT.",
        },
        {
            "doc_id": "KB-DOC-2",
            "title": "Service Credit & Compensation Guarantee",
            "plan_code": "TELCO-XGSPON-SYMM-10G",
            "regional_apn": "internet.telco.us-east",
            "sla": "Credit Policy",
            "summary": "If full service disruption exceeds 24 hours from initial customer report timestamp, an automatic prorated monthly SLA credit of $25 per day applies directly to the account statement under billing code SLA-CREDIT-AUTO.",
        },
        {
            "doc_id": "KB-DOC-3",
            "title": "Temporary Emergency Hotspot Pass",
            "plan_code": "TELCO-XGSPON-SYMM-10G",
            "regional_apn": "5g.telco.nr-data",
            "sla": "Instant Hotspot",
            "summary": "Eligible residential gigabit subscribers experiencing confirmed physical fiber cuts qualify for an immediate complimentary 100GB 5G backup hotspot pass issued to their linked mobile number.",
        },
        {
            "doc_id": "KB-DOC-4",
            "title": "Field Safety & Hazardous Line Protocol",
            "plan_code": "All",
            "regional_apn": "N/A",
            "sla": "Safety First",
            "summary": "Customers must not touch, cut, or look directly into severed optical fiber ends due to invisible class 3B laser radiation which can cause permanent retinal damage.",
        },
    ],
}

# Grounding entities to check for recall scoring
GROUNDING_TARGETS = {
    "Short": ["telco-xgspon", "internet.telco.us-east", "24-hour"],
    "Mid": ["telco-xgspon", "internet.telco.us-east", "24-hour", "credit"],
    "Long": ["telco-xgspon", "internet.telco.us-east", "24-hour", "credit", "hotspot", "safety"],
}


def score_grounding(content: str, kb_level: str) -> float:
    """Computes Grounding Recall % based on target entities present in the response."""
    targets = GROUNDING_TARGETS.get(kb_level, [])
    if not targets:
        return 100.0
    text_lower = content.lower()
    matches = sum(1 for t in targets if t in text_lower)
    return round((matches / len(targets)) * 100.0, 1)


# -----------------------------------------------------------------------------
# 2. STREAM EXECUTORS
# -----------------------------------------------------------------------------
async def execute_stream(
    stream_id: str,
    query: str,
    kb_chunks: List[Dict[str, Any]],
    classifier: JevClassifier,
    gen_17b: QwenGenerator,
    gen_7b: QwenGenerator,
    gemini: GeminiClient,
) -> Dict[str, Any]:
    """Execute a single stream under controlled parameters."""

    # 1. Jev classification (< 1ms)
    jev_res = classifier.classify(query)

    # Base settings
    temp = 0.0  # Greedy decoding for strict scientific repeatability

    if stream_id == "S1_Jev_Qwen1.7B":
        # Stream 1: Jev + Qwen 1.7B (Adaptive tokens + Intent tag)
        res = gen_17b.generate(
            user_message=query,
            kb_chunks=kb_chunks,
            is_urgent=jev_res.is_urgent,
            frustration_score=jev_res.frustration_score,
            intent=jev_res.selected_choice,
            hierarchy=jev_res.hierarchy,
            temperature=temp,
            use_jev_intent_tag=True,
        )
        return {
            "content": res.content,
            "latency_ms": res.telemetry.latency_ms if res.telemetry else res.latency_ms,
            "in_tokens": res.telemetry.input_tokens if res.telemetry else 0,
            "out_tokens": res.telemetry.output_tokens if res.telemetry else 0,
            "tps": res.telemetry.tokens_per_sec if res.telemetry else 0.0,
        }

    elif stream_id == "S2_Ctrl_Qwen1.7B":
        # Stream 2: Control Qwen 1.7B (Fixed tokens, no intent tag)
        res = gen_17b.generate(
            user_message=query,
            system_instruction=None,
            kb_chunks=kb_chunks,
            is_urgent=0.0,
            frustration_score=0.0,
            intent=None,
            temperature=temp,
            max_tokens=180,
            use_jev_intent_tag=False,
        )
        return {
            "content": res.content,
            "latency_ms": res.telemetry.latency_ms if res.telemetry else res.latency_ms,
            "in_tokens": res.telemetry.input_tokens if res.telemetry else 0,
            "out_tokens": res.telemetry.output_tokens if res.telemetry else 0,
            "tps": res.telemetry.tokens_per_sec if res.telemetry else 0.0,
        }

    elif stream_id == "S3_Jev_Qwen7B":
        # Stream 3: Jev + Qwen 2.5 7B (Adaptive tokens + Intent tag)
        res = gen_7b.generate(
            user_message=query,
            kb_chunks=kb_chunks,
            is_urgent=jev_res.is_urgent,
            frustration_score=jev_res.frustration_score,
            intent=jev_res.selected_choice,
            hierarchy=jev_res.hierarchy,
            temperature=temp,
            use_jev_intent_tag=True,
        )
        return {
            "content": res.content,
            "latency_ms": res.telemetry.latency_ms if res.telemetry else res.latency_ms,
            "in_tokens": res.telemetry.input_tokens if res.telemetry else 0,
            "out_tokens": res.telemetry.output_tokens if res.telemetry else 0,
            "tps": res.telemetry.tokens_per_sec if res.telemetry else 0.0,
        }

    elif stream_id == "S4_Ctrl_Qwen7B":
        # Stream 4: Control Qwen 2.5 7B (Fixed tokens, no intent tag)
        res = gen_7b.generate(
            user_message=query,
            system_instruction=None,
            kb_chunks=kb_chunks,
            is_urgent=0.0,
            frustration_score=0.0,
            intent=None,
            temperature=temp,
            max_tokens=180,
            use_jev_intent_tag=False,
        )
        return {
            "content": res.content,
            "latency_ms": res.telemetry.latency_ms if res.telemetry else res.latency_ms,
            "in_tokens": res.telemetry.input_tokens if res.telemetry else 0,
            "out_tokens": res.telemetry.output_tokens if res.telemetry else 0,
            "tps": res.telemetry.tokens_per_sec if res.telemetry else 0.0,
        }

    elif stream_id == "S5_Jev_Gemini":
        # Stream 5: Jev + Gemini Flash (Intent tag + adaptive max tokens 220)
        res = await gemini.generate_async(
            user_message=query,
            kb_chunks=kb_chunks,
            temperature=temp,
            intent=jev_res.selected_choice,
            is_urgent=jev_res.is_urgent,
            frustration_score=jev_res.frustration_score,
            use_jev_intent_tag=True,
            max_tokens=220,
        )
        return {
            "content": res.content,
            "latency_ms": res.telemetry.latency_ms if res.telemetry else 0.0,
            "in_tokens": res.telemetry.input_tokens if res.telemetry else 0,
            "out_tokens": res.telemetry.output_tokens if res.telemetry else 0,
            "tps": res.telemetry.tokens_per_sec if res.telemetry else 0.0,
        }

    elif stream_id == "S6_Ctrl_Gemini":
        # Stream 6: Control Gemini Flash (Standard 512 tokens, no intent tag)
        res = await gemini.generate_async(
            user_message=query,
            kb_chunks=kb_chunks,
            temperature=temp,
            use_jev_intent_tag=False,
            max_tokens=512,
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
# 3. BENCHMARK RUNNER (STREAM-BY-STREAM TO PREVENT VRAM RE-LOAD CHURN)
# -----------------------------------------------------------------------------
async def run_scientific_matrix(shots: int = 3):
    print("=" * 110, flush=True)
    print("SCIENTIFIC FACTORIAL BENCHMARK: 3x3 MATRIX (9 SCENARIOS) x 6 MODEL ARCHITECTURES", flush=True)
    print(f"Protocol: 1 Warm-up Shot (discarded) + {shots} Measured Repetitions (Greedy Temperature = 0.0)", flush=True)
    print("Optimization: Models grouped sequentially to keep VRAM resident and eliminate swapping thrashing.", flush=True)
    print("=" * 110, flush=True)

    # Initialize engines
    classifier = JevClassifier()
    gen_17b = QwenGenerator(ollama_model="qwen3:1.7b")
    gen_7b = QwenGenerator(ollama_model="qwen2.5:7b")
    gemini = GeminiClient()

    stream_names = [
        ("S1_Jev_Qwen1.7B", "Qwen 1.7B (Jev-Assisted)"),
        ("S2_Ctrl_Qwen1.7B", "Qwen 1.7B (Control Baseline)"),
        ("S3_Jev_Qwen7B", "Qwen 2.5 7B 4-bit (Jev-Assisted)"),
        ("S4_Ctrl_Qwen7B", "Qwen 2.5 7B 4-bit (Control Baseline)"),
        ("S5_Jev_Gemini", "Gemini 3.8 Flash (Jev-Assisted)"),
        ("S6_Ctrl_Gemini", "Gemini 3.8 Flash (Control Baseline)"),
    ]

    # Scenario matrix definitions
    scenarios = []
    for q_name, q_text in QUERIES.items():
        for kb_name, kb_data in KB_CHUNKS.items():
            scenarios.append({
                "key": f"{q_name}_{kb_name}",
                "query_level": q_name,
                "query_text": q_text,
                "kb_level": kb_name,
                "kb_data": kb_data,
            })

    # Load existing intermediate results if present
    out_file = "scientific_matrix_results.json"
    matrix_store: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    matrix_store = saved
                elif isinstance(saved, list):
                    for item in saved:
                        k = f"{item['query_level']}_{item['kb_level']}"
                        matrix_store[k] = item
        except Exception:
            matrix_store = {}

    for sc in scenarios:
        k = sc["key"]
        if k not in matrix_store:
            matrix_store[k] = {
                "query_level": sc["query_level"],
                "kb_level": sc["kb_level"],
                "streams": {},
            }

    # Run stream by stream to keep Ollama model resident in VRAM
    for s_id, s_label in stream_names:
        print(f"\n{'#' * 110}", flush=True)
        print(f"STARTING STREAM: [{s_id}] - {s_label}", flush=True)
        print(f"{'#' * 110}", flush=True)

        # Prime the stream with a one-time warm-up to ensure model is in VRAM
        first_sc = scenarios[0]
        print(f"  [Warm-up / VRAM Priming] Running shot 0 on '{first_sc['key']}'...", flush=True)
        t_warm_0 = time.perf_counter()
        try:
            _ = await execute_stream(
                s_id, first_sc["query_text"], first_sc["kb_data"], classifier, gen_17b, gen_7b, gemini
            )
            print(f"  [Warm-up Finished] Prime latency: {round((time.perf_counter() - t_warm_0) * 1000, 1)} ms (Discarded).", flush=True)
        except Exception as e:
            print(f"  [Warm-up Warning] {e}", flush=True)

        for sc_idx, sc in enumerate(scenarios, 1):
            q_name = sc["query_level"]
            kb_name = sc["kb_level"]
            q_text = sc["query_text"]
            kb_data = sc["kb_data"]
            sc_key = sc["key"]

            print(f"  [{sc_idx}/9] Scenario {q_name} Query x {kb_name} KB ...", end="", flush=True)

            latencies: List[float] = []
            in_tokens_list: List[int] = []
            out_tokens_list: List[int] = []
            tps_list: List[float] = []
            grounding_scores: List[float] = []
            sample_content = ""

            for shot in range(1, shots + 1):
                try:
                    out = await execute_stream(
                        s_id, q_text, kb_data, classifier, gen_17b, gen_7b, gemini
                    )
                    lat = out.get("latency_ms", 0.0)
                    latencies.append(lat)
                    in_tokens_list.append(out.get("in_tokens", 0))
                    out_tokens_list.append(out.get("out_tokens", 0))
                    tps_list.append(out.get("tps", 0.0))
                    c = out.get("content", "")
                    g_score = score_grounding(c, kb_name)
                    grounding_scores.append(g_score)
                    sample_content = c
                except Exception as e:
                    print(f" [Shot {shot} Err: {e}]", end="", flush=True)

            if latencies:
                med_lat = round(statistics.median(latencies), 1)
                mean_lat = round(statistics.mean(latencies), 1)
                std_lat = round(statistics.stdev(latencies), 1) if len(latencies) > 1 else 0.0
                avg_out = round(statistics.mean(out_tokens_list), 1)
                avg_in = round(statistics.mean(in_tokens_list), 1)
                avg_tps = round(statistics.mean(tps_list), 1)
                avg_ground = round(statistics.mean(grounding_scores), 1)

                matrix_store[sc_key]["streams"][s_id] = {
                    "median_ms": med_lat,
                    "mean_ms": mean_lat,
                    "std_ms": std_lat,
                    "in_tokens": avg_in,
                    "out_tokens": avg_out,
                    "tps": avg_tps,
                    "grounding_pct": avg_ground,
                    "sample_snippet": sample_content[:140].replace("\n", " ").strip(),
                }

                print(
                    f" -> Median: {med_lat:>6.1f} ms | Out: {avg_out:>4.0f} tok | "
                    f"Speed: {avg_tps:>5.1f} t/s | Grounding: {avg_ground:>5.1f}%",
                    flush=True,
                )

        # Save checkpoint after each complete stream
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(matrix_store, f, indent=2)
        print(f"  --> Stream {s_id} complete. Saved to {out_file}.", flush=True)

    # Convert store to sorted table
    results_table = list(matrix_store.values())

    # Print Final Markdown Matrix
    print_markdown_matrix(results_table)


def print_markdown_matrix(data: List[Dict[str, Any]]):
    print("\n" + "=" * 120, flush=True)
    print("FINAL SCIENTIFIC BENCHMARK MATRIX (MEDIAN COMPUTE LATENCY ms | GROUNDING RECALL %)", flush=True)
    print("=" * 120, flush=True)

    header = "| Query | KB | S1: Jev+1.7B | S2: Ctrl 1.7B | S3: Jev+7B (4-bit) | S4: Ctrl 7B | S5: Jev+Gemini | S6: Ctrl Gemini |"
    sep = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    print(header, flush=True)
    print(sep, flush=True)

    for row in data:
        q = row["query_level"]
        kb = row["kb_level"]
        st = row.get("streams", {})

        def fmt(sid):
            s = st.get(sid, {})
            m = s.get("median_ms", "--")
            g = s.get("grounding_pct", "--")
            return f"{m}ms ({g}%)"

        line = (
            f"| {q:<5} | {kb:<5} | {fmt('S1_Jev_Qwen1.7B')} | {fmt('S2_Ctrl_Qwen1.7B')} | "
            f"{fmt('S3_Jev_Qwen7B')} | {fmt('S4_Ctrl_Qwen7B')} | {fmt('S5_Jev_Gemini')} | {fmt('S6_Ctrl_Gemini')} |"
        )
        print(line, flush=True)


if __name__ == "__main__":
    asyncio.run(run_scientific_matrix(shots=3))
