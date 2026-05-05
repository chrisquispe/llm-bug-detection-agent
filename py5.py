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

# These are the fields we expect from the model
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

# ——— Helpers ———
def extract_text_from_pdf(path: str) -> str:
    """
    Safely read all text from a PDF, or return empty string on error.
    """
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except PdfReadError:
        print(f"Warning: Skipping malformed PDF {os.path.basename(path)}")
        return ""

def classify_paper(text: str, filename: str) -> dict:
    """
    Send the full text to the LLM and parse its JSON reply.
    """
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

    # Parse JSON, with a fallback to extract {...}
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
        # Return defaults so we never raise
        return {
            key: [] if key.endswith("_used") or key.endswith("_models") else ""
            for key in EXPECTED_KEYS
        }

# ——— Main pipeline ———
records = []
pdf_paths = glob.glob(os.path.join(DIR, "*.pdf"))

for path in pdf_paths:
    fname = os.path.basename(path)
    print(f"Processing {fname}…")
    text = extract_text_from_pdf(path)

    info = classify_paper(text, fname)
    info["file_name"] = fname
    records.append(info)

# ——— Build DataFrame ———
df = pd.DataFrame(records)

# ——— Pad missing columns so grouping won't KeyError ———
for key in EXPECTED_KEYS:
    if key not in df.columns:
        # default to empty list for list fields, else empty string
        df[key] = []  

# ——— Export to Excel ———
with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    # Full details sheet
    df.to_excel(writer, sheet_name="All_Papers", index=False)

    # One summary sheet per field
    for col in EXPECTED_KEYS:
        grp = df.copy()
        grp[col] = grp[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        summary = (
            grp.groupby(col)["file_name"]
               .apply(lambda names: "; ".join(names))
               .reset_index()
        )
        summary.to_excel(writer, sheet_name=col[:31], index=False)

print(f"Analysis complete. Results saved to {OUTPUT}")