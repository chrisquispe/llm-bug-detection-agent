import os
import glob
import json
from openai import OpenAI
import PyPDF2
from PyPDF2.errors import PdfReadError
import pandas as pd
from dotenv import load_dotenv

# ——— Load API key from .env ———
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ——— Configuration ———
DIR    = "directory"
OUTPUT = "papers.xlsx"

# Fields we expect from the model
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
]

# Additional grouping parameters
BUG_CLASSES = {
    "Buffer Overflow": ["buffer overflow", "overflow"],
    "Use-After-Free": ["use-after-free", "use after free"],
    "SQL Injection": ["sql injection", "injection"],
    # add more as needed
}
BASIC_STRATEGIES = {"zero-shot", "few-shot", "chain-of-thought", "cot"}

# ——— Helpers ———
def extract_text_from_pdf(path: str) -> str:
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
    except PdfReadError:
        print(f"Warning: Skipping malformed PDF {os.path.basename(path)}")
        return ""

def classify_paper(text: str, filename: str) -> dict:
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
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1000,
    )
    content = res.choices[0].message.content.strip()

    # Parse JSON, fallback if malformed
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end   = content.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(content[start:end+1])
            except json.JSONDecodeError:
                pass
        print(f"Warning: malformed JSON for {filename}, using empty defaults")
        return {
            key: [] if key.endswith("_used") or key.endswith("_models") else ""
            for key in EXPECTED_KEYS
        }

def categorize_language(lang: str) -> str:
    l = (lang or "").lower()
    if "c++" in l or ("c" in l and "/" not in l and "," not in l):
        return "C/C++"
    if "java" in l:
        return "Java"
    if "python" in l:
        return "Python"
    if "solidity" in l or "smart contract" in l:
        return "Solidity"
    return "Other"

def classify_bug(bug: str) -> list[str]:
    b = (bug or "").lower()
    found = [cls for cls, keys in BUG_CLASSES.items() if any(k in b for k in keys)]
    return found or ["General"]

def tag_enhancement(strategy: str) -> str:
    s = (strategy or "").lower()
    if any(bs in s for bs in BASIC_STRATEGIES):
        return "LLM-only"
    return "LLM + new technique"

def normalize_to_str(x):
    return ", ".join(x) if isinstance(x, list) else (x or "")

# ——— Main pipeline ———
records = []
for path in glob.glob(os.path.join(DIR, "*.pdf")):
    fname = os.path.basename(path)
    print(f"Processing {fname}…")
    text = extract_text_from_pdf(path)
    info = classify_paper(text, fname)
    info["file_name"] = fname
    records.append(info)

# ——— Build DataFrame ———
df = pd.DataFrame(records)

# ——— Pad missing columns ———
for key in EXPECTED_KEYS:
    if key not in df.columns:
        df[key] = []

# ——— Compute extra groupings ———
df["language_category"] = df["programming_language"].apply(categorize_language)
df["bug_class"]         = df["bug_type"].apply(lambda bt: classify_bug(normalize_to_str(bt)))
df["llm_enhancement"]   = df["llm_usage_strategy"].apply(tag_enhancement)

# ——— Export to Excel ———
all_sheets = EXPECTED_KEYS + ["language_category", "bug_class", "llm_enhancement"]
with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="All_Papers", index=False)
    for col in all_sheets:
        grp = df.copy()
        grp[col] = grp[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        summary = (
            grp.groupby(col)["file_name"]
               .apply(lambda names: "; ".join(names))
               .reset_index()
        )
        summary.to_excel(writer, sheet_name=col[:31], index=False)

print(f"Analysis complete. Results saved to {OUTPUT}")