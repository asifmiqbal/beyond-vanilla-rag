"""Prepares the complete arXiv submission package in arxiv_submission/ (git-ignored).
Author: Asif Muhammad Iqbal, Founder, Logic42 Lab
"""

import os
import shutil
import tarfile
import zipfile

SUB_DIR = "arxiv_submission"
os.makedirs(SUB_DIR, exist_ok=True)
os.makedirs(os.path.join(SUB_DIR, "latex_source"), exist_ok=True)

# 1. Copy PDF to submission folder
pdf_src = "Beyond_Vanilla_RAG_v1.0.pdf"
pdf_dest = os.path.join(SUB_DIR, "Beyond_Vanilla_RAG_v1.0.pdf")
if os.path.exists(pdf_src):
    shutil.copy2(pdf_src, pdf_dest)
    print(f"Copied {pdf_src} to {pdf_dest}")

# 2. Copy all 7 figures to latex_source
fig_dir = "Docs/figures"
latex_dir = os.path.join(SUB_DIR, "latex_source")
for fn in [
    "fig1_multidomain_pareto.png",
    "fig2_multidomain_token_efficiency.png",
    "fig3_multidomain_speedup.png",
    "fig4_tail_latency_cdf.png",
    "fig5_autoregressive_law_fit.png",
    "fig6_attention_dilution_heatmap.png",
    "fig7_finops_cost_scaling.png"
]:
    shutil.copy2(os.path.join(fig_dir, fn), os.path.join(latex_dir, fn))

# 3. Create clean LaTeX main.tex for arXiv
latex_content = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{authblk}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!70!black,
    citecolor=blue!70!black,
    urlcolor=blue!70!black
}

\title{\textbf{Beyond Vanilla RAG: How Pre-Generation Decision Models Reshape Latency, Token Economy, and Grounding in Local SLMs and Frontier LLMs}}

\author{\textbf{Asif Muhammad Iqbal}}
\affil{Founder, Logic42 Lab \\ \texttt{asif@logic42.ai} \\ \url{https://logic42.ai}}

\date{September 20, 2026 \\[0.5em] \small Version: v1.0.0 --- Preprint. Work in progress.}

\begin{document}

\maketitle

\begin{abstract}
Building enterprise customer support and incident triage systems with Retrieval-Augmented Generation (RAG) exposes a fundamental tension between \textbf{generation latency}, \textbf{token expenditure}, and \textbf{grounding fidelity}. While practitioners routinely deploy either unconstrained \textbf{Vanilla RAG} or rigid \textbf{Regex-assisted pipelines}, recent architectural advances introduce \textbf{Decision Models}---lightweight, sub-millisecond classification engines designed to evaluate intent, urgency, and token budgets before generative decoding begins. To investigate this, we present an exhaustive, multi-domain empirical benchmark across \textbf{324 full-factorial experimental conditions} ($4\text{ Operational Domains} \times 3\text{ Query Complexities} \times 3\text{ Knowledge Base Densities} \times 3\text{ Model Architectures} \times 3\text{ Classification Regimes}$) evaluating \textbf{Small Local (Qwen 1.7B)}, \textbf{Medium Local (Qwen 2.5 7B, 4-bit Q4\_K\_M)}, and \textbf{Frontier Cloud (Gemini 3.8 Flash)} under three triage regimes: \textbf{Vanilla RAG (Control)}, \textbf{Compiled Regex Trie}, and \textbf{TypeSafe AI Jev (Calibrated Decision Model)} on an NVIDIA GeForce RTX 4070 Mobile GPU (8GB VRAM) and Google AI Studio. Our empirical findings demonstrate: (1) Compiled Regex cuts SLM latency from 905.2 ms to 567.5 ms (37.3\% speedup), but induces an acute grounding penalty, dropping recall from 83.1\% to 60.7\% (-22.4\%); (2) The Jev Decision Model matches the sub-600 ms latency of Regex (576.2 ms vs. 567.5 ms, $t = 8.138, p = 1.38 \times 10^{-9}$, Cohen's $d = 1.356$), while recovering recall to 65.7\% (+5.0\% over Regex overall, +18.0\% in Contracts) and achieving benchmark-record token efficiency ($\eta = 0.930$, $p = 0.0133$, Cohen's $d = 0.435$); (3) Decision models paired with the \textbf{Zero-Fluff Invariant} make cheap local 1.7B SLMs enterprise-viable, delivering verified technical dispatch codes in 526 ms to 576 ms at zero cloud API cost; and (4) Frontier cloud LLM latency is WAN-dominated (~1.5s transit), but Jev preserves reasoning fidelity (73.2\% vs. 70.4\% for Regex), enables instant operational dispatch ($<1\text{ ms}$), and empowers a hybrid router to safely offload 80\% of routine traffic, achieving a \textbf{79.1\% net enterprise FinOps cost reduction}.
\end{abstract}

\vspace{1em}
\noindent \textbf{Keywords:} Retrieval-Augmented Generation, Decision Models, Small Language Models, Memory-Bandwidth Bounds, Compound AI Systems, Token Economy, Enterprise Incident Triage.

\vspace{1em}
\noindent \textbf{Full Technical Report:} The complete manuscript, experimental logs, interactive workbench, and open-source reproduction suite are available at: \\
\url{https://github.com/asifmiqbal/beyond-vanilla-rag}

\section{Introduction}
When designing conversational AI interfaces for mission-critical enterprise operations---such as telecommunications customer care, network operations centers (NOC), and billing dispute resolution---system architects face a foundational trilemma between generation latency, grounding fidelity, and inference cost.

We evaluate three generational paradigms across 324 experimental conditions:
\begin{enumerate}
    \item \textbf{Vanilla RAG (Control):} Unconstrained generation paired with retrieved knowledge base chunks.
    \item \textbf{Compiled Regex Trie + RAG:} Pre-generation keyword pattern matching enforcing strict token bounds.
    \item \textbf{Decision Model + RAG (TypeSafe AI Jev):} Sub-millisecond ($<0.2\text{ ms}$) calibrated classification evaluating semantic intent, urgency, and token budget prior to decoding.
\end{enumerate}

\section{Empirical Results Summary}
\begin{table*}[t]
\centering
\small
\caption{Master Benchmark Performance Across 324 Runs (36 Scenarios $\times$ 9 Streams)}
\label{tab:master}
\begin{tabular}{llccccc}
\toprule
\textbf{Model Scale} & \textbf{Triage Regime} & \textbf{Median Latency ($p50$)} & \textbf{Tail ($p95$)} & \textbf{Recall (\%)} & \textbf{Tokens} & \textbf{Efficiency ($\eta$)} \\
\midrule
\textbf{Small Local (1.7B)} & Vanilla RAG (Control) & 905.2 ms [831, 979] & 1440.3 ms & 83.1\% & 137.5 & 0.647 \\
 & + Compiled Regex Trie & 567.5 ms [548, 587] & 694.0 ms & 60.7\% & 83.7 & 0.721 \\
 & \textbf{+ TypeSafe AI Jev} & \textbf{576.2 ms [556, 596]} & \textbf{689.5 ms} & \textbf{65.7\%} & \textbf{85.7} & \textbf{0.754} \\
\midrule
\textbf{Medium Local (7B)} & Vanilla RAG (Control) & 1711.6 ms [1551, 1872] & 2756.2 ms & 63.1\% & 86.5 & 0.804 \\
 & + Compiled Regex Trie & 1534.5 ms [1448, 1621] & 2058.0 ms & 63.8\% & 76.0 & 0.874 \\
 & \textbf{+ TypeSafe AI Jev} & \textbf{1512.4 ms [1425, 1599]} & \textbf{1992.5 ms} & \textbf{62.4\%} & \textbf{72.1} & \textbf{0.930} \\
\midrule
\textbf{Frontier Cloud (Gemini)} & Vanilla RAG (Control) & 2128.3 ms [1977, 2280] & 3180.4 ms & 73.5\% & 105.6 & 0.722 \\
 & + Compiled Regex Trie & 2028.4 ms [1891, 2166] & 2980.2 ms & 70.4\% & 109.1 & 0.590 \\
 & \textbf{+ TypeSafe AI Jev} & \textbf{2097.9 ms [1941, 2255]} & \textbf{3010.5 ms} & \textbf{73.2\%} & \textbf{108.3} & \textbf{0.691} \\
\bottomrule
\end{tabular}
\end{table*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.88\textwidth]{fig1_multidomain_pareto.png}
\caption{Figure 1: 9-Stream Multi-Domain Pareto Frontier with 95\% Confidence Interval Error Whiskers.}
\label{fig:pareto}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.85\textwidth]{fig4_tail_latency_cdf.png}
\caption{Figure 4: Cumulative Distribution Function (CDF) of Local Edge Latency ($p50, p95, p99$) on Qwen 1.7B.}
\label{fig:cdf}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.85\textwidth]{fig5_autoregressive_law_fit.png}
\caption{Figure 5: Physical Validation of the Autoregressive Law ($R^2 = 1.00$) on NVIDIA RTX 4070.}
\label{fig:fit}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=0.85\textwidth]{fig7_finops_cost_scaling.png}
\caption{Figure 7: FinOps Operating Expenditure vs. Monthly Query Volume (79.1\% Net Savings).}
\label{fig:finops}
\end{figure*}

\section*{Acknowledgments \& Funding Disclosure}
This research was conceived, executed, and independently funded by \textbf{Logic42 Lab} (\url{https://logic42.ai}). All local compute hardware, engineering labor, and Frontier Cloud API expenditures (Google AI Studio) were funded directly by Logic42 Lab. The author acknowledges \textbf{TypeSafe AI} for providing early developer platform access and an initial \$5 API credit used during preliminary pilot testing of the Jev SystemOne API. The empirical design, methodology, results analysis, and conclusions were conducted independently with zero editorial intervention or sponsorship from TypeSafe AI.

\section*{Artifact Availability}
All benchmark execution scripts, raw telemetry datasets, figure generators, and reproduction pipelines are open-sourced under the MIT License at: \url{https://github.com/asifmiqbal/beyond-vanilla-rag}.

\end{document}
"""

with open(os.path.join(latex_dir, "main.tex"), "w", encoding="utf-8") as f:
    f.write(latex_content)

# 4. Create tar.gz archive of the LaTeX source (Standard arXiv upload package)
tar_gz_path = os.path.join(SUB_DIR, "arxiv_latex_source.tar.gz")
with tarfile.open(tar_gz_path, "w:gz") as tar:
    for item in os.listdir(latex_dir):
        tar.add(os.path.join(latex_dir, item), arcname=item)
print(f"Created LaTeX arXiv package: {tar_gz_path}")

# 5. Create zip archive as well
zip_path = os.path.join(SUB_DIR, "arxiv_latex_source.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_f:
    for item in os.listdir(latex_dir):
        zip_f.write(os.path.join(latex_dir, item), arcname=item)
print(f"Created ZIP arXiv package: {zip_path}")

# 6. Generate Copy-Paste Text file for the arXiv web form
metadata_text = """================================================================================
ARXIV SUBMISSION METADATA (READY TO COPY-PASTE)
Paper: Beyond Vanilla RAG
Author: Asif Muhammad Iqbal (Founder, Logic42 Lab)
================================================================================

TITLE:
Beyond Vanilla RAG: How Pre-Generation Decision Models Reshape Latency, Token Economy, and Grounding in Local SLMs and Frontier LLMs

AUTHORS:
Asif Muhammad Iqbal

AUTHOR AFFILIATION:
Logic42 Lab

PRIMARY CATEGORY:
cs.AI (Artificial Intelligence)

CROSS-LIST CATEGORIES (Recommended):
cs.CL (Computation and Language)
cs.DC (Distributed, Parallel, and Cluster Computing)
cs.NI (Networking and Internet Architecture)

ABSTRACT:
Building enterprise customer support and incident triage systems with Retrieval-Augmented Generation (RAG) exposes a fundamental tension between generation latency, token expenditure, and grounding fidelity. While practitioners routinely deploy either unconstrained Vanilla RAG or rigid Regex-assisted pipelines, recent architectural advances introduce Decision Models---lightweight, sub-millisecond classification engines designed to evaluate intent, urgency, and token budgets before generative decoding begins. To investigate this, we present an exhaustive, multi-domain empirical benchmark across 324 full-factorial experimental conditions (4 Operational Domains x 3 Query Complexities x 3 Knowledge Base Densities x 3 Model Architectures x 3 Classification Regimes) evaluating Small Local (Qwen 1.7B), Medium Local (Qwen 2.5 7B, 4-bit Q4_K_M), and Frontier Cloud (Gemini 3.8 Flash) under three triage regimes: Vanilla RAG (Control), Compiled Regex Trie, and TypeSafe AI Jev (Calibrated Decision Model) on an NVIDIA GeForce RTX 4070 Mobile GPU (8GB VRAM) and Google AI Studio. Our empirical findings demonstrate: (1) Compiled Regex cuts SLM latency from 905.2 ms to 567.5 ms (37.3% speedup), but induces an acute grounding penalty, dropping recall from 83.1% to 60.7% (-22.4%); (2) The Jev Decision Model matches the sub-600 ms latency of Regex (576.2 ms vs. 567.5 ms, t = 8.138, p = 1.38e-09, Cohen's d = 1.356), while recovering recall to 65.7% (+5.0% over Regex overall, +18.0% in Contracts) and achieving benchmark-record token efficiency (eta = 0.930, p = 0.0133, Cohen's d = 0.435); (3) Decision models paired with the Zero-Fluff Invariant make cheap local 1.7B SLMs enterprise-viable, delivering verified technical dispatch codes in 526 ms to 576 ms at zero cloud API cost; and (4) Frontier cloud LLM latency is WAN-dominated (~1.5s transit), but Jev preserves reasoning fidelity (73.2% vs. 70.4% for Regex), enables instant operational dispatch (<1 ms), and empowers a hybrid router to safely offload 80% of routine traffic, achieving a 79.1% net enterprise FinOps cost reduction.

COMMENTS (Optional field in arXiv):
18 pages, 7 figures, 6 tables. Technical report by Logic42 Lab. Full benchmark dataset, interactive workbench, and open-source reproduction code available at https://github.com/asifmiqbal/beyond-vanilla-rag

REPORT NUMBER (Optional field in arXiv):
LOGIC42-TR-2026-01

ACM CLASSIFICATION (Optional):
I.2.7; C.2.4; C.4

DISTRIBUTION LICENSE (Selected during submission):
arXiv.org perpetual, non-exclusive license
(Or Creative Commons Attribution 4.0 International - CC BY 4.0)
================================================================================
"""

with open(os.path.join(SUB_DIR, "ARXIV_METADATA_COPYPASTE.txt"), "w", encoding="utf-8") as f:
    f.write(metadata_text)

# 7. Create SUBMISSION_INSTRUCTIONS.md
instructions_md = """# arXiv Submission Guide (Step-by-Step)

This folder contains all required files for submitting to arXiv.
**Note:** This entire folder is git-ignored and will not be pushed to GitHub.

---

## What Files Are in This Folder?

1. **`Beyond_Vanilla_RAG_v1.0.pdf`**: The standalone camera-ready PDF (2.0 MB).
2. **`arxiv_latex_source.tar.gz`**: Clean arXiv-ready LaTeX package (contains `main.tex` + 7 figures).
3. **`ARXIV_METADATA_COPYPASTE.txt`**: All form fields (Title, Abstract, Categories, Comments) formatted for copy-paste.

---

## How to Submit on arXiv (5-Minute Walkthrough)

### Step 1: Click "START NEW SUBMISSION"
On your arXiv dashboard (`arxiv.org/submit`), click the blue **START NEW SUBMISSION** button.

### Step 2: License & Agreement
1. Check the agreement boxes confirming you are the author.
2. Under **Select a License**, choose:
   - **`arXiv.org perpetual, non-exclusive license`** (Standard recommended by arXiv)
   - Or `CC BY 4.0` if you prefer Creative Commons.

### Step 3: Choose Primary Category
- Primary Category: **`Computer Science - Artificial Intelligence (cs.AI)`**
- Under Cross-lists, add:
  - **`cs.CL`** (Computation and Language)
  - **`cs.DC`** (Distributed, Parallel, and Cluster Computing)

### Step 4: Upload Files
You have two easy choices:
- **Option A (Recommended for arXiv):** Upload **`arxiv_latex_source.tar.gz`** (or `main.tex` + figures). arXiv's server will automatically compile it into a clean PDF.
- **Option B (Direct PDF):** Upload **`Beyond_Vanilla_RAG_v1.0.pdf`**. When asked *"Was this generated from TeX source?"*, select **No** (since it was generated via our Markdown/HTML rendering engine).

### Step 5: Copy & Paste Metadata
Open **`ARXIV_METADATA_COPYPASTE.txt`** and copy each field into the form:
- **Title**: *Beyond Vanilla RAG: How Pre-Generation Decision Models Reshape Latency, Token Economy, and Grounding in Local SLMs and Frontier LLMs*
- **Authors**: *Asif Muhammad Iqbal*
- **Abstract**: *(Copy the text block from the file)*
- **Comments**: `18 pages, 7 figures, 6 tables. Technical report by Logic42 Lab. Code: https://github.com/asifmiqbal/beyond-vanilla-rag`
- **Report Number**: `LOGIC42-TR-2026-01`

### Step 6: Preview & Submit
1. Click **View Article** to inspect the compiled preview.
2. If everything looks good, click **Submit Article**.
3. You will receive an arXiv submission identifier (e.g. `arXiv:submit/XXXXXXX`), and your paper will be announced publicly in the next scheduled mailings!
"""

with open(os.path.join(SUB_DIR, "SUBMISSION_INSTRUCTIONS.md"), "w", encoding="utf-8") as f:
    f.write(instructions_md)

print("\nSuccessfully assembled complete arXiv submission package in arxiv_submission/!")
