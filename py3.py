import os
import glob
import json
from openai import OpenAI
import PyPDF2
import pandas as pd

# CONFIG
client = OpenAI(api_key="")
DIR = "directory"
OUTPUT = "papers.xlsx"

# extract all text from a PDF
def extract_text_from_pdf(path):
    text_pages = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    return "\n".join(text_pages)

# call openai to classify a paper
def classify_paper(text, filename):
    prompt = (
        "You are an assistant that reads the full content of an academic paper and outputs a JSON object with the following keys:\n"
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
        f"Read the paper text below. Always output valid JSON. Do not include any additional keys.\n\n"
        f"Paper filename: {filename}\nPaper content:\n{text}\n"
    )
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=1000)
    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # if the model output isn't valid json, wrap with a fallback
        # attempt to locate the first and last braces
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            return json.loads(content[start:end+1])
        else:
            raise

# main processing
records = []
pdf_paths = glob.glob(os.path.join(DIR, "*.pdf"))
for path in pdf_paths:
    fname = os.path.basename(path)
    print(f"Processing {fname}...")
    text = extract_text_from_pdf(path)
    try:
        info = classify_paper(text, fname)
        info['file_name'] = fname
        records.append(info)
    except Exception as e:
        print(f"Failed to classify {fname}: {e}")

# build dataframe
df = pd.DataFrame(records)

# write to excel with grouping sheets
with pd.ExcelWriter(OUTPUT, engine='openpyxl') as writer:
    # sheet of detailed results
    df.to_excel(writer, sheet_name='All_Papers', index=False)

    # for each grouping criterion, produce a summary sheet
    for col in [
        'dataset_used', 'programming_language', 'benchmark_usage', 'bug_type',
        'dataset_scale', 'llm_usage_strategy', 'llm_models', 'analysis_type',
        'real_world_impact', 'automation_level'
    ]:
        # some columns may be lists; convert to string for grouping
        grp = df.copy()
        grp[col] = grp[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        summary = grp.groupby(col)['file_name'].apply(lambda names: "; ".join(names)).reset_index()
        # clean sheet name to max 31 chars
        sheet = col[:31]
        summary.to_excel(writer, sheet_name=sheet, index=False)

print(f"Analysis complete. Results saved to {OUTPUT}")