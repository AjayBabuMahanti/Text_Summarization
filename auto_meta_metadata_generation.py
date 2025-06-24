# AutoMeta: Best-in-Class AI Metadata Generator (Notebook)

# 🔧 Install Required Packages
!pip install --quiet PyMuPDF python-docx pdf2image pytesseract \
                  transformers keybert streamlit pyyaml sentence-transformers

# 📦 Import Libraries
import fitz  # PyMuPDF
import docx
import pytesseract
from pdf2image import convert_from_path
from transformers import pipeline
from keybert import KeyBERT
import json
import yaml
import os
from pathlib import Path

# 🧠 Load Summarization and Keyword Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
kw_model = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")

# 📄 Document Extraction Functions

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc).strip()

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join(para.text for para in doc.paragraphs).strip()

def extract_text_via_ocr(path):
    images = convert_from_path(path)
    return "\n".join(pytesseract.image_to_string(img) for img in images).strip()

# 📌 Metadata Generation Function

def generate_metadata(text, doc_type="PDF"):
    short_text = text[:1024] if len(text) > 1024 else text
    summary_result = summarizer(short_text, max_length=130, min_length=30, do_sample=False)
    summary = summary_result[0]['summary_text']
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=8)

    metadata = {
        "title": summary.split('.')[0],
        "summary": summary,
        "keywords": [kw[0] for kw in keywords],
        "document_type": doc_type,
        "word_count": len(text.split())
    }
    return metadata

# 🧪 Smart File Handler

def process_file(file_path):
    ext = Path(file_path).suffix.lower()
    text, doc_type = "", "Unknown"

    if ext == ".pdf":
        text = extract_text_from_pdf(file_path)
        if not text:
            text = extract_text_via_ocr(file_path)
            doc_type = "Scanned PDF"
        else:
            doc_type = "PDF"

    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
        doc_type = "DOCX"

    elif ext == ".txt":
        text = Path(file_path).read_text(encoding="utf-8")
        doc_type = "TXT"

    return text, doc_type

# 🚀 Run Sample (Replace with your document)
file_path = "sample.pdf"  # ← Replace with your document path

if os.path.exists(file_path):
    raw_text, doc_type = process_file(file_path)
    if raw_text:
        metadata = generate_metadata(raw_text, doc_type)

        print("\n📑 JSON Metadata Output:\n")
        print(json.dumps(metadata, indent=4))

        print("\n📋 YAML Metadata Output:\n")
        print(yaml.dump(metadata, sort_keys=False))
    else:
        print("❌ No text found in the document.")
else:
    print("⚠️ File not found. Please update the 'file_path' variable.")
