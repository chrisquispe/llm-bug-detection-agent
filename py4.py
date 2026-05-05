import os
import glob
import json
from openai import OpenAI
import PyPDF2
from PyPDF2.errors import PdfReadError
import pandas as pd

# CONFIG
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DIR    = "directory"
OUTPUT = "papers.xlsx"

# The keys we expect back from classify_paper
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

def extract_text_from_pdf(path):
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

def classify_paper(text, filename):
    """
    Call OpenAI to get a JSON classification. Raises on connection/API errors.
    """
    prompt = (
        "You are an assistant that reads the full content of an academic paper "
        "and outputs a JSON object with the following keys:\n"
        "1. dataset_used: list of dataset names if multiple; empty list if none established or new dataset.\n"
        "2. programming_language: primary language(s) used or targeted.\n"
        "3. benchmark_usage: 'established' or 'new'.\n"
        "4. bug_type: Like memory bugs or logic bugs. Mention the specifics about the bugs included.\n"
        "5. dataset_or_tool_link: URL if provided, else empty string.\n"
        "6. dataset_scale: descriptive summary (e.g., '500 bugs', '200MB total', etc.).\n"
        "7. llm_usage_strategy: 'fine-tuning', 'zero-shot', 'few-shot', 'CoT', etc.\n"
        "8. llm_models: list of LLM names and versions.\n"
        "9. analysis_type: 'static' or 'dynamic' or 'both'.\n"
        "10. real_world_impact: list of CVE IDs or notes on validation, or empty if none.\n"
        "11. automation_level: 'fully automated', 'human-in-the-loop', etc.\n"
        "12. bug_location: list where bugs were found.\n"
        "Read the paper text below. Always output valid JSON. Do not include any additional keys.\n\n"
        f"Paper filename: {filename}\nPaper content:\n{text}\n"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1000
    )
    content = response.choices[0].message.content.strip()
    return json.loads(content)

# Main processing
records = []
pdf_paths = glob.glob(os.path.join(DIR, "*.pdf"))
for path in pdf_paths:
    fname = os.path.basename(path)
    print(f"Processing {fname}…")
    text = extract_text_from_pdf(path)

    try:
        info = classify_paper(text, fname)
    except Exception as e:
        print(f"Failed to classify {fname}: {e}")
        # Build an empty stub so we still record this paper
        info = {
            key: [] if key.endswith("_used") or key.endswith("_models") else ""
            for key in EXPECTED_KEYS
        }

    # Always include the filename
    info["file_name"] = fname
    records.append(info)

# Build DataFrame
df = pd.DataFrame(records)

# Write to Excel with grouping sheets
with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    # Detailed sheet
    df.to_excel(writer, sheet_name="All_Papers", index=False)

    # Summary sheets
    for col in EXPECTED_KEYS:
        grp = df.copy()
        # Flatten lists to comma-joined strings
        grp[col] = grp[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        summary = grp.groupby(col)["file_name"] \
                     .apply(lambda names: "; ".join(names)) \
                     .reset_index()
        sheet_name = col[:31]  # Excel sheet name limit
        summary.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Analysis complete. Results saved to {OUTPUT}")