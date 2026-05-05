import os
import glob
import json
import re

from openai import OpenAI
import PyPDF2
from PyPDF2.errors import PdfReadError
import pandas as pd
from dotenv import load_dotenv

# ——— Load API key ———
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ——— Configuration ———
DIR    = "directory"
OUTPUT = "papers.xlsx"

# Raw keys expected from the LLM
RAW_KEYS = [
    "dataset_used","programming_language","benchmark_usage","bug_type",
    "dataset_or_tool_link","dataset_scale","llm_usage_strategy","llm_models",
    "analysis_type","real_world_impact","automation_level","automation_comment","bug_location",
]

# Specific bug categories (for primary_bug_type)
BUG_CATEGORIES = {
    "Buffer Overflow": ["buffer overflow","hboff","so","hobf"],
    "Use-After-Free": ["use-after-free","uaf"],
    "Integer Bugs": ["integer overflow","integer underflow","dbz"],
    "Null Pointer Deref": ["null pointer","npd"],
    "SQL Injection": ["sql injection","injection"],
}

# Patterns for detection relevance
DETECTION_PATTERNS = [
    r"\bbug detection\b",
    r"\bvulnerability detection\b",
    r"\bdetect(ion|ing)? (bugs?|vulnerabilities?)\b",
    r"\bidentify(ing)? (bugs?|vulnerabilities?)\b",
    r"\bfault localization\b",
    r"\bbug localization\b",
]

# Known datasets fallback list
KNOWN_DATASETS = [
    "ImageNet", "CIFAR-10", "CIFAR-100", "MNIST",
    "SQuAD", "CoNLL", "Penn Treebank", "MS COCO",
    "GLUE", "SuperGLUE"
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

def detect_datasets_from_text(text):
    found = set()
    for name in KNOWN_DATASETS:
        if re.search(rf"\b{name}\b", text, flags=re.IGNORECASE):
            found.add(name)
    return list(found)

def classify_paper(text, filename):
    prompt = (
        "You are an assistant that reads the full content of an academic paper\n"
        "and outputs a JSON object with exactly these keys (no extras):\n"
        "1. dataset_used (list of dataset names),\n"
        "2. programming_language (string),\n"
        "3. benchmark_usage (string),\n"
        "4. bug_type (string),\n"
        "5. dataset_or_tool_link (string),\n"
        "6. dataset_scale (string),\n"
        "7. llm_usage_strategy (string),\n"
        "8. llm_models (list),\n"
        "9. analysis_method (string) — choose one of: Empirical study, Comparative analysis, Case study, "
          "Benchmark evaluation, Exploratory study, User study, Theoretical analysis, Tool implementation, "
          "LLM prompting study, Others;\n"
        "10. real_world_impact (list),\n"
        "11. automation_level (string),\n"
        "12. bug_location (list).\n\n"
        "Definitions for analysis_method values:\n"
        "- Empirical study: Data collection and statistical analysis (e.g., bug frequency, fix patterns)\n"
        "- Comparative analysis: Comparing tools, models, or techniques\n"
        "- Case study: In-depth look at one or a few examples\n"
        "- Benchmark evaluation: Running tests against a standard benchmark\n"
        "- Exploratory study: Open-ended evaluation of model behavior\n"
        "- User study: Human participants interacting with tools\n"
        "- Theoretical analysis: Focus on formal models, frameworks, or logic\n"
        "- Tool implementation: Main contribution is building a tool, not analyzing data\n"
        "- LLM prompting study: Designing, testing, and analyzing LLM prompts\n"
        "- Others: If none of the above fits.\n\n"
        f"Paper filename: {filename}\n\n"
        f"Paper content:\n{text}\n"
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        max_tokens=1000
    )
    content = res.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # try to salvage JSON
        start, end = content.find("{"), content.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(content[start:end+1])
            except:
                pass
        # fallback empty
        return {k: ([] if k in ["dataset_used","llm_models","real_world_impact","bug_location"] else "")
                for k in RAW_KEYS}

def norm_list(x):
    return "; ".join(x) if isinstance(x, list) else (x or "")

def standardize_bug_type(raw):
    s = (raw or "").lower()
    for cat, kws in BUG_CATEGORIES.items():
        if any(kw in s for kw in kws):
            return cat
    m = re.search(r"uaf|npd|so|hboff|hobf|dbz", s)
    if m:
        return {
            "uaf":"Use-After-Free",
            "npd":"Null Pointer Deref",
            "so":"Buffer Overflow",
            "hboff":"Buffer Overflow",
            "hobf":"Buffer Overflow",
            "dbz":"Integer Bugs"
        }[m.group()]
    if re.search(r"vulnerab", s):
        return "Security vulnerability"
    return "General" if s else "Unknown"

def is_detection_related(text):
    for pat in DETECTION_PATTERNS:
        if re.search(pat, text or "", flags=re.IGNORECASE):
            return True
    return False

# ——— Classification Functions ———

def compute_automation_level(row):
    text = (row.get("full_text","") or "").lower()
    # Rule-based first
    if re.search(r"end[- ]to[- ]end|no human in the loop|fully[- ]automated", text):
        return "fully-automated", ""
    if re.search(r"tool helps|requires human input|human[- ]in[- ]the[- ]loop|semi[- ]automated", text):
        return "semi-automated", ""
    if re.search(r"entirely by humans|manual", text):
        return "manual", ""
    # Ask the LLM for unclear cases
    prompt = (
        "You are an assistant classifying the automation level of a bug-detection method.\n"
        "Possible labels: manual, semi-automated, fully-automated, unclear.\n"
        "- manual: entirely human effort (no tool)\n"
        "- semi-automated: human-guided LLM use or manual prompt writing\n"
        "- fully-automated: no human involvement once set up\n"
        "- unclear: paper truly doesn't describe automation level.\n\n"
        f"Paper content:\n{text}\n\n"
        "Respond in JSON with keys 'automation_level' and 'comment' (comment can be empty)."
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        max_tokens=200
    )
    try:
        result = json.loads(res.choices[0].message.content)
        return (
            result.get("automation_level","unclear"),
            result.get("comment","")
        )
    except:
        return "unclear", "Could not parse LLM response"

def compute_analysis_method(row):
    text = (row.get("full_text","") or "").lower()
    if re.search(r"user study|participants|survey|interview", text):
        return "User study"
    if re.search(r"theoretical|formal model|proof|framework", text):
        return "Theoretical analysis"
    if re.search(r"implement(ed)? (a )?tool|we (present|introduce) (a )?tool", text):
        return "Tool implementation"
    if re.search(r"prompt(ing)? study|design(ing)? prompts|prompt (effectiveness|evaluation)", text):
        return "LLM prompting study"
    if re.search(r"benchmark (evaluation|study)", text):
        return "Benchmark evaluation"
    if re.search(r"comparative (study|analysis)|we compare", text):
        return "Comparative analysis"
    if re.search(r"case study", text):
        return "Case study"
    if re.search(r"exploratory (study|investigation)", text):
        return "Exploratory study"
    if re.search(r"empirical (study|evaluation|analysis)|statistical|data (collection|analysis)", text):
        return "Empirical study"
    return "Others"

def compute_llm_integration(row):
    text = (row.get("full_text","") or "").lower()
    parts = []
    if "symbolic execution" in text:
        parts.append("LLM + symbolic execution")
    if "static analysis" in text:
        parts.append("LLM + static analysis")
    if "dynamic analysis" in text or "fuzzing" in text:
        parts.append("LLM + dynamic analysis")
    if not parts and "llm" in (row.get("llm_usage_strategy","") or "").lower():
        return "LLM-only"
    if len(parts)==1:
        return parts[0]
    if len(parts)>1:
        return "Hybrid"
    return "LLM-only"

def compute_benchmark_origin(row):
    text = (row.get("full_text","") or "").lower()
    manual = bool(re.search(r"manually|human[- ]written", text))
    real = bool(re.search(r"real[- ]world", text))
    synthetic = bool(re.search(r"synthetic|generated", text))
    if manual and real:
        return "both"
    if manual:
        return "manual"
    if real:
        return "real-world"
    if synthetic:
        return "synthetic"
    return "unclear"

def compute_benchmark_description(row):
    text = row.get("full_text","") or ""
    prompt = (
        "You are an assistant that reads the content of an academic paper and writes 1–2 sentences "
        "describing the benchmark used.\n\n"
        f"Paper content:\n{text}\n\n"
        "Benchmark description:"
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        max_tokens=200
    )
    return res.choices[0].message.content.strip()

def classify_scope(raw):
    s = (raw or "").lower()
    scopes = []
    if any(k in s for k in ["function","method"]):
        scopes.append("Function-level")
    if any(k in s for k in ["module","library","driver"]):
        scopes.append("Module-level")
    if "system" in s or "software" in s:
        scopes.append("System-level")
    if any(k in s for k in ["loop","variable","api","code"]):
        scopes.append("Code elements")
    return "; ".join(scopes) if scopes else "Other"

# ——— Main pipeline ———

records = []
for pdf_path in glob.glob(os.path.join(DIR, "*.pdf")):
    title = os.path.basename(pdf_path)
    print(f"Processing {title}…")
    txt = extract_text_from_pdf(pdf_path)
    info = classify_paper(txt, title)
    if not info.get("dataset_used"):
        info["dataset_used"] = detect_datasets_from_text(txt)
    info.update({"paper_title": title, "full_text": txt})
    records.append(info)

# Build DataFrame
_df = pd.DataFrame(records)
for k in RAW_KEYS + ["paper_title","full_text"]:
    if k not in _df.columns:
        _df[k] = ([] if k in ["dataset_used","llm_models","real_world_impact","bug_location"] else "")

# Rename llm_usage_strategy
_df.rename(columns={"llm_usage_strategy":"approach / technical contribution"}, inplace=True)

# Apply transformations
_df["bug_detection_related"] = _df["full_text"].apply(lambda t: "Yes" if is_detection_related(t) else "No")
_df["analysis_method"]       = _df.apply(compute_analysis_method, axis=1)
_df[["automation_level","automation_comment"]] = pd.DataFrame(
    _df.apply(compute_automation_level, axis=1).tolist(),
    index=_df.index
)
_df["llm_integration_type"]  = _df.apply(compute_llm_integration, axis=1)
_df["benchmark_origin"]      = _df.apply(compute_benchmark_origin, axis=1)
_df["benchmark_description"] = _df.apply(compute_benchmark_description, axis=1)
_df["code_scope_analyzed"]   = _df["bug_location"].apply(norm_list).apply(classify_scope)

# Cleanup and reorder
df = _df.copy()
df.drop(columns=[
    "analysis_type","bug_location","benchmark_usage","full_text","benchmark_characteristics"
], errors="ignore", inplace=True)
df["dataset_scale"] = df["dataset_scale"].apply(norm_list).str.strip()

# detection-first ordering
df = pd.concat([
    df[df["bug_detection_related"]=="Yes"],
    df[df["bug_detection_related"]=="No"]
], ignore_index=True)

fixed_cols = [
    "paper_title","code_scope_analyzed","bug_detection_related",
    "analysis_method","automation_level","automation_comment",
    "llm_integration_type","benchmark_origin","benchmark_description",
    "dataset_scale"
]
other_cols = [c for c in df.columns if c not in fixed_cols]
df = df[fixed_cols + other_cols]

# Export to Excel with summaries
with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="All_Papers", index=False)
    for col in df.columns:
        safe = re.sub(r"[\\/:*?[\]]","_", col)[:31]
        tmp = df.copy()
        tmp[col] = tmp[col].apply(norm_list)
        summary = tmp.groupby(col)["paper_title"].apply(lambda names: "; ".join(names)).reset_index(name="paper_titles")
        summary.to_excel(writer, sheet_name=safe, index=False)

print(f"Written cleaned results to {OUTPUT}")