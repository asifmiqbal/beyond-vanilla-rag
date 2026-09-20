"""Compiles Docs/RESEARCH_PAPER.md into a publication-grade PDF using Headless Edge/Chrome.
Author: Asif Muhammad Iqbal, Founder, Logic42 Lab (logic42.ai)
"""

import os
import re
import base64
import shutil
import subprocess
import markdown

SOURCE_MD = "Docs/RESEARCH_PAPER.md"
OUTPUT_HTML = "Docs/paper_preview.html"
OUTPUT_PDF = "Beyond_Vanilla_RAG_v1.0.pdf"
DOCS_PDF = "Docs/Beyond_Vanilla_RAG_v1.0.pdf"
ARTIFACT_DIR = r"C:\Users\asifm\.gemini\antigravity\brain\c9ee5ec3-b7ae-4154-9727-b20a5cfb709c"

with open(SOURCE_MD, "r", encoding="utf-8") as f:
    text = f.read()

# Replace image paths with base64 data URIs so the PDF contains embedded images
def embed_images(match):
    alt_text = match.group(1)
    img_path = match.group(2)
    # resolve path
    if not os.path.isabs(img_path):
        candidate = os.path.join("Docs", img_path)
        if not os.path.exists(candidate):
            candidate = os.path.join("Docs", "figures", os.path.basename(img_path))
        if os.path.exists(candidate):
            img_path = candidate
    if os.path.exists(img_path):
        with open(img_path, "rb") as img_f:
            b64 = base64.b64encode(img_f.read()).decode("utf-8")
        ext = os.path.splitext(img_path)[1].lower().replace(".", "")
        if ext == "jpg": ext = "jpeg"
        return f'<div class="figure-container"><img src="data:image/{ext};base64,{b64}" alt="{alt_text}" class="paper-img" /><p class="figure-caption">{alt_text}</p></div>'
    return match.group(0)

# Convert markdown images: ![alt](path)
text_processed = re.sub(r'!\[(.*?)\]\((.*?)\)', embed_images, text)

# Convert Markdown tables and extensions
html_body = markdown.markdown(text_processed, extensions=['tables', 'fenced_code', 'toc'])

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Beyond Vanilla RAG - Logic42 Lab</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{
        delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}},
            {{left: '\\(', right: '\\)', display: false}},
            {{left: '\\[', right: '\\]', display: true}}
        ]
    }});"></script>
<style>
    @page {{
        size: letter;
        margin: 20mm 18mm 22mm 18mm;
        @bottom-right {{
            content: counter(page);
        }}
    }}
    body {{
        font-family: 'Times New Roman', Times, serif;
        font-size: 10.5pt;
        line-height: 1.45;
        color: #111;
        max-width: 100%;
        margin: 0 auto;
        padding: 0;
        background: #fff;
    }}
    h1 {{
        font-size: 19pt;
        line-height: 1.25;
        text-align: center;
        margin-top: 10pt;
        margin-bottom: 12pt;
        font-weight: bold;
    }}
    .author-block {{
        text-align: center;
        margin-bottom: 20pt;
        font-size: 10.5pt;
    }}
    .author-name {{
        font-size: 12pt;
        font-weight: bold;
    }}
    h2 {{
        font-size: 13pt;
        border-bottom: 1px solid #333;
        padding-bottom: 2pt;
        margin-top: 18pt;
        margin-bottom: 8pt;
        font-weight: bold;
    }}
    h3 {{
        font-size: 11pt;
        margin-top: 14pt;
        margin-bottom: 6pt;
        font-weight: bold;
    }}
    p {{
        text-align: justify;
        margin-bottom: 7pt;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 12pt 0;
        font-size: 8.5pt;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #444;
        padding: 4.5pt 5pt;
        text-align: left;
    }}
    th {{
        background-color: #f2f2f2;
        font-weight: bold;
    }}
    tr:nth-child(even) {{
        background-color: #fafafa;
    }}
    code {{
        font-family: 'Courier New', Courier, monospace;
        font-size: 8.5pt;
        background: #f4f4f4;
        padding: 1pt 3pt;
        border-radius: 2pt;
    }}
    pre {{
        background: #f7f7f7;
        border: 1px solid #ddd;
        padding: 7pt;
        font-size: 8pt;
        overflow-x: auto;
        page-break-inside: avoid;
    }}
    .figure-container {{
        text-align: center;
        margin: 14pt 0;
        page-break-inside: avoid;
    }}
    .paper-img {{
        max-width: 95%;
        height: auto;
        border: 1px solid #ccc;
    }}
    .figure-caption {{
        font-size: 9pt;
        font-style: italic;
        margin-top: 4pt;
        color: #333;
        text-align: center;
    }}
    blockquote {{
        border-left: 3px solid #08519c;
        margin: 8pt 0;
        padding-left: 10pt;
        color: #333;
        font-style: italic;
    }}
    hr {{
        border: 0;
        border-top: 1px solid #ccc;
        margin: 15pt 0;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"Generated HTML preview at {OUTPUT_HTML}")

# Try Edge or Chrome headless
browsers = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]

browser_exe = None
for b in browsers:
    if os.path.exists(b):
        browser_exe = b
        break

if not browser_exe:
    print("[ERROR] No headless browser found.")
    exit(1)

html_abs = os.path.abspath(OUTPUT_HTML)
pdf_abs = os.path.abspath(OUTPUT_PDF)

cmd = [
    browser_exe,
    "--headless",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_abs}",
    f"file:///{html_abs}"
]

print(f"Running headless render via: {browser_exe}")
res = subprocess.run(cmd, capture_output=True, text=True)
if res.returncode == 0 and os.path.exists(OUTPUT_PDF):
    size_mb = os.path.getsize(OUTPUT_PDF) / (1024 * 1024)
    print(f"[SUCCESS] Generated PDF: {OUTPUT_PDF} ({size_mb:.2f} MB)")
    # Copy to Docs and artifacts directory
    shutil.copy2(OUTPUT_PDF, DOCS_PDF)
    shutil.copy2(OUTPUT_PDF, os.path.join(ARTIFACT_DIR, OUTPUT_PDF))
    print(f"Copied to {DOCS_PDF} and artifact directory.")
else:
    print(f"[ERROR] PDF generation failed. Return code: {res.returncode}")
    print(res.stderr)
