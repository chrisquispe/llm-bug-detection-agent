import json
import re

# Load original output
with open("output.json", "r") as f:
    raw_data = json.load(f)

# Helper function to extract answers from markdown-style text
def extract_answer(block, question_number):
    match = re.search(rf"{question_number}\.\s+\*\*(.*?)\*\*.*?\n\s*-\s*(.*?)\n", block, re.DOTALL)
    return match.group(2).strip() if match else ""

cleaned_data = []

for entry in raw_data:
    response = entry.get("response", "")

    bug_type_raw = extract_answer(response, 1)
    lang_raw = extract_answer(response, 2)
    benchmark_raw = extract_answer(response, 3)
    repo_link = extract_answer(response, 4)
    public_link = extract_answer(response, 5)
    prompt_strategy = extract_answer(response, 6)

    cleaned_data.append({
        "file_name": entry.get("file_name"),
        "bug_type": bug_type_raw,
        "bug_origin": "Real-world" if "real-world" in bug_type_raw.lower() else "Synthetic" if "synthetic" in bug_type_raw.lower() else "Unknown",
        "programming_language": lang_raw,
        "benchmarks_or_datasets": benchmark_raw,
        "dataset_tool_repo": repo_link,
        "publicly_available": "Yes" if "yes" in public_link.lower() else "No",
        "llm_prompt_strategy": prompt_strategy
    })

# Save to new JSON
with open("cleaned_output.json", "w") as f:
    json.dump(cleaned_data, f, indent=2)

print("Cleaned output saved to 'cleaned_output.json'")
