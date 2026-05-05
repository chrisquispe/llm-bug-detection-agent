# LLM Bug Detection Research Agent

An AI-powered research tool built during undergraduate research at Virginia Tech that automates the analysis and classification of academic papers on LLM-based bug detection.

## What it does
- Extracts text from 100+ research PDFs using PyPDF2
- Uses Gemini 2.0 Flash (via OpenAI-compatible API) to classify each paper across 14 dimensions
- Classifies papers by bug type, automation level, LLM models used, benchmark source, dataset scale, and more
- Outputs results to a structured Excel spreadsheet for analysis
- Reduces manual review time by ~50%

## Tech Stack
- Python
- Gemini 2.0 Flash (Google AI) via OpenAI-compatible endpoint
- PyPDF2, Pandas, OpenAI SDK, python-dotenv

## Setup
1. Clone the repo
2. Install dependencies: `pip install openai PyPDF2 pandas python-dotenv openpyxl`
3. Create a `.env` file with your Gemini API key: `GEMINI_API_KEY=your_key_here`
4. Place research PDFs in a `directory/` folder
5. Run: `python Ai_bot.py`

## Output
Generates a `papers.xlsx` file with 14 columns per paper including bug type, dataset used, approach, automation level, and more.

## Context
Built as part of undergraduate AI research at Virginia Tech (May–August 2025).
