
import streamlit as st
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import yaml

from transformers import pipeline
from keybert import KeyBERT
import fitz
import docx
import pytesseract
from pdf2image import convert_from_path

@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    kw_model = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")
    return summarizer, kw_model

summarizer, kw_model = load_models()

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc).strip()

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs).strip()

def extract_text_via_ocr(path):
    images = convert_from_path(path)
    return "\n".join(pytesseract.image_to_string(img) for img in images).strip()

def generate_metadata(text, doc_type):
    short_text = text[:1024] if len(text) > 1024 else text
    summary = summarizer(short_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=8)
    return {
        "title": summary.split('.')[0],
        "summary": summary,
        "keywords": [kw[0] for kw in keywords],
        "document_type": doc_type,
        "word_count": len(text.split())
    }

st.set_page_config(page_title="AutoMeta Metadata Generator", layout="centered")
st.title("📄 AutoMeta: Smart Metadata Generator")
st.write("Upload any document (PDF, DOCX, TXT) and get structured metadata using AI.")

uploaded = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])

if uploaded:
    with NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp_file:
        tmp_file.write(uploaded.read())
        tmp_path = tmp_file.name

    ext = Path(uploaded.name).suffix.lower()
    text, doc_type = "", "Unknown"

    if ext == ".pdf":
        text = extract_text_from_pdf(tmp_path)
        if not text:
            text = extract_text_via_ocr(tmp_path)
            doc_type = "Scanned PDF"
        else:
            doc_type = "PDF"
    elif ext == ".docx":
        text = extract_text_from_docx(tmp_path)
        doc_type = "DOCX"
    elif ext == ".txt":
        text = Path(tmp_path).read_text(encoding="utf-8")
        doc_type = "TXT"

    if text:
        st.success("✅ Document processed successfully.")
        metadata = generate_metadata(text, doc_type)
        st.subheader("📑 JSON Metadata")
        st.json(metadata)
        st.subheader("📋 YAML Metadata")
        st.code(yaml.dump(metadata, sort_keys=False), language="yaml")
    else:
        st.error("❌ Could not extract text from the file.")
