"""3-Way Benchmark Test: Jev + Qwen (Stream A) vs Raw Qwen (Stream B) vs Gemini 3.8 Flash (Stream C).

Compares:
1. Jev System One + LanceDB RAG + Local Qwen (Specialized Grounded Telco Stack)
2. Raw Local Qwen (Local Baseline without Jev or RAG)
3. Google AI Studio Gemini 3.8 Flash (Cloud Frontier Baseline)
"""

import asyncio
import os
import sys
import time
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(override=True)

from services.jev.classifier import JevClassifier
from rag.retriever import TelcoRetriever
from services.vllm.generator import QwenGenerator
from services.gemini.client import GeminiClient
from pipelines.orchestrator import PipelineOrchestrator


async def run_benchmark():
    print("=" * 90)
    print("3-WAY COMPARATIVE BENCHMARK: [1] Jev+Qwen  |  [2] Raw Qwen  |  [3] Gemini 3.8 Flash")
    print("=" * 90)

    # Initialize components
    classifier = JevClassifier()
    retriever = TelcoRetriever()
    generator = QwenGenerator()
    gemini = GeminiClient()
    orchestrator = PipelineOrchestrator(
        classifier=classifier,
        retriever=retriever,
        generator=generator,
        gemini=gemini
    )

    test_queries = [
        "Emergency! Construction cut our fiber drop line, ONT LOS red light blinking!",
        "What is the 5G roaming APN and data setup instructions for international travel?",
    ]

    for q_idx, query in enumerate(test_queries, 1):
        print(f"\n" + "#" * 90)
        print(f"QUERY #{q_idx}: \"{query}\"")
        print("#" * 90)

        # ----------------------------------------------------------------------
        # TEST 1: Jev + LanceDB RAG + Local Qwen (Specialized Grounded Stack)
        # ----------------------------------------------------------------------
        print("\n>>> Running [1] Jev + LanceDB RAG + Local Qwen...")
        t0 = time.perf_counter()
        orch_res = orchestrator.process_turn(query)
        t1 = time.perf_counter()
        jev_qwen_latency = round((t1 - t0) * 1000.0, 1)

        gen_a = orch_res.generation_result
        tel_a = gen_a.telemetry
        jev_res = orch_res.jev_result

        # ----------------------------------------------------------------------
        # TEST 2: Raw Local Qwen (Direct query, zero Jev, zero RAG)
        # ----------------------------------------------------------------------
        print(">>> Running [2] Raw Local Qwen (Zero Jev, Zero RAG)...")
        t0 = time.perf_counter()
        raw_qwen_res = generator.generate(
            user_message=query,
            system_instruction="You are a helpful customer service assistant for a telecommunications provider. Keep responses concise and direct.",
            kb_chunks=None, # No RAG
            is_urgent=0.0,  # No Jev Noul
            frustration_score=0.0, # No Jev Score
            intent=None,     # No Jev Intent
            max_tokens=256,
        )
        t1 = time.perf_counter()
        raw_qwen_latency = round((t1 - t0) * 1000.0, 1)
        tel_raw = raw_qwen_res.telemetry

        # ----------------------------------------------------------------------
        # TEST 3: Gemini 3.8 Flash (Google AI Studio Cloud Frontier)
        # ----------------------------------------------------------------------
        print(">>> Running [3] Google AI Studio Gemini 3.8 Flash...")
        t0 = time.perf_counter()
        gemini_res = await gemini.generate_async(query)
        t1 = time.perf_counter()
        gemini_latency = round((t1 - t0) * 1000.0, 1)
        tel_gemini = gemini_res.telemetry

        # ----------------------------------------------------------------------
        # Print Detailed Comparison Matrix
        # ----------------------------------------------------------------------
        print("\n" + "=" * 90)
        print(f"PERFORMANCE TELEMETRY MATRIX (Query {q_idx})")
        print("=" * 90)
        print(f"{'Metric':<25} | {'[1] Jev + Qwen (Grounded)':<25} | {'[2] Raw Qwen':<18} | {'[3] Gemini 3.8 Flash':<20}")
        print("-" * 90)
        print(f"{'Total Latency (ms)':<25} | {jev_qwen_latency:<25} | {raw_qwen_latency:<18} | {gemini_latency:<20}")
        
        inp_a = tel_a.input_tokens if tel_a else "N/A"
        inp_raw = tel_raw.input_tokens if tel_raw else "N/A"
        inp_gem = tel_gemini.input_tokens if tel_gemini else "N/A"
        print(f"{'Input Tokens':<25} | {str(inp_a):<25} | {str(inp_raw):<18} | {str(inp_gem):<20}")

        out_a = tel_a.output_tokens if tel_a else "N/A"
        out_raw = tel_raw.output_tokens if tel_raw else "N/A"
        out_gem = tel_gemini.output_tokens if tel_gemini else "N/A"
        print(f"{'Output Tokens':<25} | {str(out_a):<25} | {str(out_raw):<18} | {str(out_gem):<20}")

        tps_a = tel_a.tokens_per_sec if tel_a else "N/A"
        tps_raw = tel_raw.tokens_per_sec if tel_raw else "N/A"
        tps_gem = tel_gemini.tokens_per_sec if tel_gemini else "N/A"
        print(f"{'Throughput (Tokens/s)':<25} | {str(tps_a):<25} | {str(tps_raw):<18} | {str(tps_gem):<20}")

        print(f"{'Intent Class':<25} | {jev_res.selected_choice:<25} | {'(None)':<18} | {'(None)':<20}")
        print(f"{'Urgency (Noul)':<25} | {str(jev_res.is_urgent):<25} | {'(None)':<18} | {'(None)':<20}")
        print(f"{'RAG Citations':<25} | {str(len(orch_res.retrieved_chunks)) + ' docs injected':<25} | {'0 (Unassisted)':<18} | {'0 (Unassisted)':<20}")
        print(f"{'Action Triage':<25} | {orch_res.handover_card.dispatch_queue if orch_res.handover_card else 'Contained':<25} | {'No triage':<18} | {'No triage':<20}")

        print("\n" + "-" * 90)
        print("OUTPUT COMPARISON:")
        print("-" * 90)
        print("\n[1] JEV + QWEN (GROUNDED RESPONSE):")
        print(gen_a.content.strip())

        print("\n[2] RAW QWEN (UNGROUNDED BASELINE):")
        print(raw_qwen_res.content.strip())

        print("\n[3] GEMINI 3.8 FLASH (CLOUD FRONTIER):")
        print(gemini_res.content.strip())
        print("-" * 90)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
