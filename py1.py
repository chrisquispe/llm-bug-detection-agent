import os
import requests
import xml.etree.ElementTree as ET
import time
from tqdm import tqdm
import json

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CONFIG
#client = OpenAI(api_key="")
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
DIR = "directory"
OUTPUT = "output.json"
MODEL = "gpt-4o-mini"

# process pdfs with grobid
def process_pdf_with_grobid(pdf_path):
    with open(pdf_path, 'rb') as f:
        files = {'input': f}
        response = requests.post(GROBID_URL, files=files)
        if response.status_code == 200:
            return response.text
        else:
            return None

# extract text sections from grobid XML
def extract_text_from_xml(xml_data):
    root = ET.fromstring(xml_data)
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}
    text = ""
    for sec in root.findall('.//tei:body//tei:div', namespaces):
        for p in sec.findall('.//tei:p', namespaces):
            text += ET.tostring(p, encoding='unicode', method='text') + "\n"
    return text.strip()

# ask LLM questions
def ask_questions(title, context):
    prompt = f"""
From the paper titled \"{title}\", answer the following:

1. What is the bug type? Are they synthetic or real-world?
2. What programming language is used?
3. What benchmarks or datasets are used?
4. Is a dataset/tool repository provided? If yes, include the link.
5. Is the tool/benchmark publicly available? If yes, include the link.
6. Any LLM prompt strategy used?

Context:
{context}
"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for extracting structured information from research papers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# main loop
def run_pipeline():
    time.sleep(2)
    results = []
    for file in tqdm(os.listdir(DIR)):
        if not file.endswith(".pdf"):
            continue
        pdf_path = os.path.join(DIR, file)
        title = file.replace(".pdf", "")

        xml = process_pdf_with_grobid(pdf_path)
        if not xml:
            results.append({"file_name": file, "response": "Grobid processing failed"})
            continue

        text = extract_text_from_xml(xml)
        answer = ask_questions(title, text)

        results.append({"file_name": file, "response": answer})

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {OUTPUT}")

run_pipeline()
