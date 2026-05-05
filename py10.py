import os
import glob
import json
import time

from openai import OpenAI
import PyPDF2
from PyPDF2.errors import PdfReadError
import pandas as pd
from dotenv import load_dotenv

# ——— Load API key and configure Gemini via OpenAI-compatible endpoint ———
load_dotenv()
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ——— Configuration ———
DIR    = "directory"
OUTPUT = "papers.xlsx"

# Raw keys expected in DataFrame, ordered with bug_detection_related as second column
RAW_KEYS = [
    "paper_title",
    "bug_detection_related",
    "dataset_used",
    "programming_language",
    "bug_type",
    "dataset_or_tool_link",
    "dataset_scale",
    "approach / technical contribution",
    "llm_models",
    "analysis_method",
    "real_world_impact",
    "automation_level",
    "automation_comment",
    "benchmark_source",          # Renamed from benchmark_origin
    "benchmark_description",
    "code_scope_analyzed"
]

# ——— Helpers ———

def extract_text_from_pdf(path):
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
    except PdfReadError:
        print(f"[WARN] malformed PDF {os.path.basename(path)}")
        return ""

# ——— LLM-driven classification ———

def llm_classify(text, question, choices=None, freeform=False):
    if choices:
        prompt = (
            f"Based on the content of this paper, {question}\n"
            f"Choices: {', '.join(choices)}\n"
            f"Content:\n{text}\n"
            "Respond with exactly one of the choices."
        )
    else:
        prompt = (
            f"Based on the content of this paper, {question}\n"
            f"Content:\n{text}\n"
            "Respond concisely."
        )
    for _ in range(3):
        try:
            res = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=10000
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(15)
            else:
                raise
    return ""

# ——— Main pipeline ———

pdf_paths = sorted(glob.glob(os.path.join(DIR, "*.pdf")))
print(f"Found {len(pdf_paths)} PDFs in '{DIR}':")
for p in pdf_paths:
    print(" -", p)

records = []
for pdf_path in pdf_paths:
    title = os.path.basename(pdf_path)
    print(f"Processing {title}…")
    full_text = extract_text_from_pdf(pdf_path)
    record = {"paper_title": title}

    # Bug detection relevance (Yes/No) via LLM prompt only
    record["bug_detection_related"] = llm_classify(
        full_text,
        (
            "Carefully read the Abstract, Introduction, Methods, and Experiments sections of this paper. "
            "Determine whether the primary goal of the work is to develop or evaluate methods/tools for detecting software bugs. "
            "If it is, respond with exactly 'Yes'. If not, respond with exactly 'No'."
        ),
        ["Yes", "No"]
    )

    # More precise: examine Methods/Experiments for dataset names
    record["dataset_used"] = llm_classify(
        full_text,
        (
            "Carefully examine the Methods and Experiments sections of this paper and "
            "identify each dataset the authors used (by name). "
            "If no dataset was used, respond exactly 'None'. "
            "Otherwise, list the dataset names as a comma-separated list."
        ),
        freeform=True
    )

    # More precise: search entire document for dataset/tool URLs
    record["dataset_or_tool_link"] = llm_classify(
        full_text,
        (
            "Read every part of this paper—including Abstract, main text, Methods/Experiments, "
            "References, footnotes, and appendices—and identify **all** URLs (http or https) "
            "that point to datasets or tool repositories (e.g., GitHub, Zenodo, Data.gov). "
            "List them as a comma-separated list of full URLs. "
            "Assume at least one exists; only respond 'None' if you truly find zero."
        ),
        freeform=True
    )

    # Dataset scale via LLM freeform prompt
    record["dataset_scale"] = llm_classify(
        full_text,
        (
            "Thoroughly read the Methods and Experiments sections of this paper and find any quantitative description of the dataset’s size—"
            "for example, number of samples, number of projects, total files, or data volume. "
            "Respond with that number and its unit (for example: '10,000 images', '1.2M lines of code'). "
            "If multiple scales are given, list the primary one only. "
            "If no explicit size is stated, respond exactly 'Unknown'."
        ),
        freeform=True
    )

    # Other fields via LLM prompts
    record["programming_language"] = llm_classify(
        full_text,
        "What programming language(s) does the paper use? List comma-separated or 'None'.",
        freeform=True
    )

    # Expanded bug_type prompt to cover more categories and 'Other (specify)'
    record["bug_type"] = llm_classify(
        full_text,
        (
            "Read the entire content of this paper and identify which one of the following bug types it addresses. "
            "Choose from: buffer overflow, use-after-free, integer overflow, XSS, SQL injection, memory leak, "
            "race condition, logic bug, and Other (specify). "
            "If you select 'Other (specify)', append a brief explanation in parentheses. "
            "Respond with exactly one entry from this list (e.g., 'memory leak' or 'Other (specify: off-by-one)')."
        ),
        freeform=True
    )

    record["approach / technical contribution"] = llm_classify(
        full_text,
        (
            "Read the entire paper and, in exactly one sentence, summarize the core technical method they use and "
            "the main contribution it makes (e.g., new capability, improved accuracy, novel framework)."
        ),
        freeform=True
    )

    record["llm_models"] = llm_classify(
        full_text,
        "Which LLM models are used or mentioned? List comma-separated or 'None'.",
        freeform=True
    )

    record["analysis_method"] = llm_classify(
        full_text,
        (
            "Read the Abstract, Methods, and Experiments sections of this paper and select exactly one label "
            "that best describes the research approach:\n"
            "• Empirical study — data collection and statistical analysis (e.g., bug frequency, fix patterns)\n"
            "• Comparative analysis — comparing tools, models, or techniques\n"
            "• Case study — in-depth look at one or a few examples\n"
            "• Benchmark evaluation — running tests against a standard benchmark\n"
            "• Exploratory study — open-ended evaluation of model behavior\n"
            "• User study — human participants interacting with tools\n"
            "• Theoretical analysis — focus on formal models, frameworks, or logic\n"
            "• Tool implementation — main contribution is building a tool, not analyzing data\n"
            "• LLM prompting study — focused on designing, testing, and analyzing LLM prompts\n\n"
            "Respond with exactly one of these labels."
        ),
        choices=[
            "Empirical study",
            "Comparative analysis",
            "Case study",
            "Benchmark evaluation",
            "Exploratory study",
            "User study",
            "Theoretical analysis",
            "Tool implementation",
            "LLM prompting study"
        ]
    )

    record["real_world_impact"] = llm_classify(
        full_text,
        "Summarize any real-world impact described, or 'None'.",
        freeform=True
    )

    # Updated: richer automation level prompt (freeform)
    record["automation_level"] = llm_classify(
        full_text,
        (
            "Carefully read the entire paper and choose exactly one of these automation levels:\n"
            "• manual — entirely human effort (no automated tools)\n"
            "• semi-automated — human-guided LLM use or manual prompt writing\n"
            "• fully-automated — no human involvement once the system is set up\n"
            "• unclear — paper does not describe its automation level\n\n"
            "Respond with exactly that one word. "
            "If you pick 'unclear', add a very short parenthetical note (under 10 words) explaining why."
        ),
        freeform=True
    )
    record["automation_comment"] = ""

    # Renamed prompt usage: benchmark_source instead of benchmark_origin
    record["benchmark_source"] = llm_classify(
        full_text,
        (
            "Read the paper’s description of its benchmark and select exactly one of these origin labels:\n"
            "• manual — benchmarks manually constructed by the authors\n"
            "• real-world — collected from real software projects or datasets\n"
            "• synthetic — artificially generated benchmarks\n"
            "• both — a combination of real-world and synthetic data\n"
            "• unclear — no clear origin is described\n\n"
            "Respond with exactly one of: manual, real-world, synthetic, both, unclear."
        ),
        choices=["manual", "real-world", "synthetic", "both", "unclear"]
    )

    record["benchmark_description"] = llm_classify(
        full_text,
        (
            "Read the entire paper and, in one to two sentences, describe the benchmark dataset or suite used, "
            "including how it was collected or constructed and its intended purpose "
            "(for example: 'Collected from GitHub Python projects with known syntax bugs', "
            "or 'Manually written code snippets targeting resource leaks')."
        ),
        freeform=True
    )

    record["code_scope_analyzed"] = llm_classify(
        full_text,
        (
            "Read the paper and determine at which code scope(s) the analysis is performed. "
            "Choose from these categories:\n"
            "• Function-level\n"
            "• Module-level (e.g., drivers, libraries)\n"
            "• System-level (e.g., entire software system, cyber-physical system)\n"
            "• Code elements (e.g., loops, variables)\n"
            "• Other (if it doesn’t fit above)\n\n"
            "If the paper covers more than one, list them separated by semicolons (for example: "
            "Function-level; Code elements). Respond with category names only, no extra text."
        ),
        freeform=True
    )

    records.append(record)

# Build DataFrame and export

df = pd.DataFrame(records)
# Ensure all expected columns
for k in RAW_KEYS:
    if k not in df.columns:
        df[k] = ""

# Reorder to match desired column order
_df = df[RAW_KEYS]
_df.to_excel(OUTPUT, index=False)
print(f"Written to {OUTPUT}")