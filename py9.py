import os
import glob
import json
import re
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

# Raw keys expected from the LLM
RAW_KEYS = [
    "dataset_used","programming_language","benchmark_usage","bug_type",
    "dataset_or_tool_link","dataset_scale","llm_usage_strategy","llm_models",
    "analysis_type","real_world_impact","automation_level","automation_comment","bug_location",
]

# Specific bug categories
BUG_CATEGORIES = {
    "Buffer Overflow": ["buffer overflow","hboff","so","hobf"],
    "Use-After-Free": ["use-after-free","uaf"],
    "Integer Bugs": ["integer overflow","integer underflow","dbz"],
    "Null Pointer Deref": ["null pointer","npd"],
    "SQL Injection": ["sql injection","injection"],
}

# Detection relevance patterns
DETECTION_PATTERNS = [
    r"\bbug detection\b",
    r"\bvulnerability detection\b",
    r"\bdetect(ion|ing)? (bugs?|vulnerabilities?)\b",
    r"\bidentify(ing)? (bugs?|vulnerabilities?)\b",
]

# Known datasets fallback
KNOWN_DATASETS = [
    "ImageNet","CIFAR-10","CIFAR-100","MNIST",
    "SQuAD","CoNLL","Penn Treebank","MS COCO",
    "GLUE","SuperGLUE"
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

# Debug wrapper
def debug_extract(path):
    text = extract_text_from_pdf(path)
    print(f"\nFound text in {os.path.basename(path)} (first 500 chars):\n" +
          text[:500].replace("\n","␤") + "\n")
    return text

def detect_datasets_from_text(text):
    found = set()
    for name in KNOWN_DATASETS:
        if re.search(rf"\b{name}\b", text, flags=re.IGNORECASE):
            found.add(name)
    return list(found)

# ——— LLM-driven classification ———

def classify_paper(text, filename):
    prompt = (
        "You are an assistant that reads the full content of an academic paper and outputs a JSON with these keys:\n"
        "dataset_used, programming_language, benchmark_usage, bug_type, dataset_or_tool_link, dataset_scale,\n"
        "llm_usage_strategy, llm_models, analysis_method, real_world_impact, automation_level, bug_location.\n"
        f"Filename: {filename}\nContent:\n{text}\n"
    )
    for attempt in range(3):
        try:
            res = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=1000
            )
            content = res.choices[0].message.content.strip()
            break
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                wait = 15
                print(f"[WARN] classify_paper rate-limit, sleeping {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)
            else:
                raise
    else:
        print(f"[ERROR] classify_paper failed for {filename}")
        content = ""

    print(f"RAW classify_paper for {filename}:\n{content}\n")
    try:
        return json.loads(content)
    except:
        start, end = content.find("{"), content.rfind("}")
        if start < end:
            try:
                return json.loads(content[start:end+1])
            except:
                pass
        return {k: ([] if k in ["dataset_used","llm_models","real_world_impact","bug_location"] else "") for k in RAW_KEYS}


def norm_list(x):
    return "; ".join(x) if isinstance(x, list) else (x or "")

# ——— Rule-based classification ———

def compute_analysis_method(row):
    t = row.get("full_text","").lower()
    if re.search(r"empirical", t): return "Empirical study"
    if re.search(r"comparative|we compare", t): return "Comparative analysis"
    if re.search(r"case study", t): return "Case study"
    if re.search(r"benchmark", t): return "Benchmark evaluation"
    if re.search(r"exploratory", t): return "Exploratory study"
    return "Others"

def compute_automation_level(row):
    t = row.get("full_text","").lower()
    if re.search(r"no human in the loop|fully-automated", t): return ("fully-automated", "")
    if re.search(r"semi-automated", t): return ("semi-automated", "")
    if re.search(r"manual", t): return ("manual", "")
    return ("unclear", "")

def compute_llm_integration(row):
    t = row.get("full_text","").lower()
    parts = []
    if "static analysis" in t: parts.append("LLM+static")
    if "dynamic analysis" in t: parts.append("LLM+dynamic")
    return parts[0] if len(parts)==1 else ("Hybrid" if parts else "LLM-only")

def compute_benchmark_origin(row):
    t = row.get("full_text","").lower()
    if "real-world" in t: return "real-world"
    if "synthetic" in t: return "synthetic"
    return "unclear"

def classify_scope(raw):
    s = raw.lower()
    scopes = []
    if "function" in s: scopes.append("Function-level")
    if "module" in s: scopes.append("Module-level")
    return "; " .join(scopes) if scopes else "Other"

# ——— Benchmark description with retry ———

def compute_benchmark_description(row):
    text = row.get("full_text","")
    prompt = f"Describe the benchmark used in this paper:\n{text}\n"
    for i in range(3):
        try:
            res = client.chat.completions.create(
                model="gemini-2.0-flash",
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=200
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(15)
                continue
            else:
                raise
    return ""

# ——— Main pipeline ———

# List PDFs
pdf_paths = sorted(glob.glob(os.path.join(DIR, "*.pdf")))
print(f"Found {len(pdf_paths)} PDFs in '{DIR}':")
for p in pdf_paths: print(" -", p)

records = []
for pdf_path in pdf_paths:
    title = os.path.basename(pdf_path)
    print(f"Processing {title}…")
    txt = debug_extract(pdf_path)
    info = classify_paper(txt, title)
    if not info.get("dataset_used"):
        info["dataset_used"] = detect_datasets_from_text(txt)
    info.update({"paper_title": title, "full_text": txt})
    records.append(info)

# Build DataFrame and export
df = pd.DataFrame(records)
for k in RAW_KEYS + ['paper_title','full_text']:
    if k not in df:
        df[k] = [] if k in ['dataset_used','llm_models','real_world_impact','bug_location'] else ""

# Rename column
df.rename(columns={"llm_usage_strategy":"approach / technical contribution"}, inplace=True)

# Derive new columns
df['bug_detection_related'] = df['full_text'].apply(
    lambda t: "Yes" if any(re.search(p, t, flags=re.IGNORECASE) for p in DETECTION_PATTERNS) else "No"
)
df['analysis_method']       = df.apply(compute_analysis_method, axis=1)
levels, comments           = zip(*df.apply(compute_automation_level, axis=1))
df['automation_level']      = levels
df['automation_comment']    = comments
df['llm_integration_type']  = df.apply(compute_llm_integration, axis=1)
df['benchmark_origin']      = df.apply(compute_benchmark_origin, axis=1)
df['benchmark_description'] = df.apply(compute_benchmark_description, axis=1)
df['code_scope_analyzed']   = df['bug_location'].apply(norm_list).apply(classify_scope)

# Cleanup unwanted columns
cols_to_drop = ['analysis_type','bug_location','benchmark_usage','full_text']
df.drop(columns=[c for c in cols_to_drop if c in df], inplace=True)

# ——— REORDER so paper_title is first and bug_detection_related is second ———
ordered_cols = ['paper_title', 'bug_detection_related'] + [c for c in df.columns if c not in ['paper_title','bug_detection_related']]
df = df[ordered_cols]

# Write out
df.to_excel(OUTPUT, index=False)
print(f"Written to {OUTPUT}")
