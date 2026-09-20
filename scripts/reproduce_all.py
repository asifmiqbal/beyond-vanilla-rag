"""Single-Command Master Reproduction Script for the Beyond Vanilla RAG Benchmark.
Author: Asif Muhammad Iqbal, Founder, Logic42 AI (logic42.ai)
"""

import subprocess
import sys
import os

def run_step(desc, cmd):
    print(f"\n{'='*70}")
    print(f"STEP: {desc}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*70}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[ERROR] Step failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[SUCCESS] {desc} completed successfully.")

def main():
    print("======================================================================")
    print("Logic42 AI - Beyond Vanilla RAG Benchmark Reproduction Pipeline")
    print("Author: Asif Muhammad Iqbal, Founder, Logic42 AI (logic42.ai)")
    print("======================================================================")
    
    # 1. Verify environment & data
    if not os.path.exists("expanded_multidomain_results.json"):
        print("[ERROR] expanded_multidomain_results.json not found in repository root.")
        sys.exit(1)
        
    # 2. Compute full inferential statistical battery
    run_step("Computing Effect Sizes (Cohen's d) & Holm-Bonferroni FWER Corrections",
             f"{sys.executable} scripts/compute_significance.py")

    # 3. Summarize tables
    run_step("Compiling Master Empirical Data Tables",
             f"{sys.executable} scripts/summarize_results.py")

    # 4. Generate all 7 publication-grade figures
    run_step("Generating 7-Figure Tier-1 Journal Publication Suite (with Error Whiskers)",
             f"{sys.executable} scripts/generate_all_journal_figures.py")

    print("\n" + "="*70)
    print("ALL REPRODUCTION STEPS COMPLETED SUCCESSFULLY!")
    print("Output figures saved to: Docs/figures/")
    print("Output data tables saved to: Docs/summary_tables.md")
    print("Manuscript available at: Docs/RESEARCH_PAPER.md")
    print("======================================================================\n")

if __name__ == "__main__":
    main()
