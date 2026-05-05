import os
import glob
import json

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

# Fields we expect back
EXPECTED_KEYS = [
    "dataset_used",
    "programming_language",
    "benchmark_usage",
    "bug_type",
    "dataset_or_tool_link",
    "dataset_scale",
    "llm_usage_strategy",
    "llm_models",
    "analysis_type",
    "real_world_impact",
    "automation_level",
    "bug_location",  # will be renamed
]

# ——— Helper functions ———

def extract_text_from_pdf(path):
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
    except PdfReadError:
        print(f"Warning: Skipping malformed PDF {os.path.basename(path)}")
        return ""

def classify_paper(text, filename):
    prompt = (
        "You are an assistant that reads the full content of an academic paper "
        "and outputs a JSON object with exactly these keys (no extras):\n"
        "1. dataset_used (list),\n"
        "2. programming_language (string),\n"
        "3. benchmark_usage (string),\n"
        "4. bug_type (string),\n"
        "5. dataset_or_tool_link (string),\n"
        "6. dataset_scale (string),\n"
        "7. llm_usage_strategy (string),\n"
        "8. llm_models (list),\n"
        "9. analysis_type (string),\n"
        "10. real_world_impact (list),\n"
        "11. automation_level (string),\n"
        "12. bug_location (list).\n\n"
        f"Paper filename: {filename}\n\nPaper content:\n{text}"
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        max_tokens=1000,
    )
    content = res.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end   = content.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end+1])
            except json.JSONDecodeError:
                pass
        # fallback to empty defaults
        return {k: ([] if k in ["dataset_used","llm_models","real_world_impact","bug_location"] else "") 
                for k in EXPECTED_KEYS}

# Scope standardization
def classify_scope(loc):
    s = loc.lower() if isinstance(loc, str) else ""
    scopes = []
    if any(x in s for x in ["function", "method"]):
        scopes.append("Function-level")
    if any(x in s for x in ["module", "library", "driver"]):
        scopes.append("Module-level")
    if "system" in s or "software" in s:
        scopes.append("System-level")
    if any(x in s for x in ["loop", "variable", "api", "data structure"]):
        scopes.append("Code elements")
    return "; ".join(dict.fromkeys(scopes)) if scopes else "Other"

# Split analysis_type
def classify_analysis_method(at):
    t = at.lower() if isinstance(at, str) else ""
    if "case study" in t: return "Case study"
    if "empirical" in t: return "Empirical study"
    if "benchmark" in t: return "Benchmark evaluation"
    if "comparative" in t: return "Comparative analysis"
    if "exploratory" in t: return "Exploratory study"
    return "Other"

def classify_analysis_technique(at):
    t = at.lower() if isinstance(at, str) else ""
    techniques = []
    if "static" in t:      techniques.append("Static analysis")
    if "dynamic" in t or "fuzz" in t: techniques.append("Dynamic analysis")
    if "symbolic" in t:    techniques.append("Symbolic execution")
    if "llm" in t or "prompt" in t:   techniques.append("LLM prompting")
    return "; ".join(dict.fromkeys(techniques)) if techniques else "Other"

# Normalize automation_level
def classify_automation_level(al):
    t = al.lower() if isinstance(al, str) else ""
    if "fully" in t: return "fully-automated"
    if "semi" in t:  return "semi-automated"
    if "manual" in t: return "manual"
    return "unclear"

# LLM integration type
def classify_llm_integration(row):
    tech = classify_analysis_technique(row.get("analysis_type", ""))
    combo = []
    if "static analysis" in tech:    combo.append("LLM + static analysis")
    if "dynamic analysis" in tech:   combo.append("LLM + dynamic analysis")
    if "symbolic execution" in tech: combo.append("LLM + symbolic execution")
    if not combo and ("llm" in tech.lower()): return "LLM-only"
    return "Hybrid" if len(combo)>1 else (combo[0] if combo else "LLM-only")

# Flatten helper
def norm_str(x):
    return "; ".join(x) if isinstance(x, list) else (x or "")

# ——— Main pipeline ———
records = []
for path in glob.glob(os.path.join(DIR, "*.pdf")):
    fn = os.path.basename(path)
    print(f"Processing {fn}…")
    text = extract_text_from_pdf(path)
    entry = classify_paper(text, fn)
    entry["paper_title"] = fn
    records.append(entry)

# Build DF
df = pd.DataFrame(records)

# Pad missing columns
for k in EXPECTED_KEYS:
    if k not in df.columns:
        df[k] = [] if k in ["dataset_used","llm_models","real_world_impact","bug_location"] else ""

# Rename & compute new fields
df["code_scope_analyzed"]    = df["bug_location"].apply(norm_str).apply(classify_scope)
df["analysis_method"]        = df["analysis_type"].apply(classify_analysis_method)
df["analysis_technique"]     = df["analysis_type"].apply(classify_analysis_technique)
df["automation_level"]       = df["automation_level"].apply(classify_automation_level)
df["llm_integration_type"]   = df.apply(classify_llm_integration, axis=1)

# Drop old & reorder
df = df.drop(columns=["bug_location", "analysis_type"])
cols = ["paper_title"] + [c for c in df.columns if c!="paper_title"]
df = df[cols]

# Export to Excel
with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    # Full details
    df.to_excel(writer, sheet_name="All_Papers", index=False)

    for col in cols:
        if col == "paper_title":
            # list unique titles
            summary = pd.DataFrame({ "paper_title": df["paper_title"].unique() })
        else:
            grp = df.copy()
            grp[col] = grp[col].apply(norm_str)
            summary = (
                grp.groupby(col)["paper_title"]
                   .apply(lambda names: "; ".join(names))
                   .reset_index()
            )
        summary.to_excel(writer, sheet_name=col[:31], index=False)

print(f"Saved to {OUTPUT}")