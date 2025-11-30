import re
import io
from PyPDF2 import PdfReader
import docx
import os

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s]{7,}\d)')

def extract_text_from_pdf(data):
    reader = PdfReader(io.BytesIO(data))
    text = []
    for page in reader.pages:
        if page.extract_text():
            text.append(page.extract_text())
    return "\n".join(text)

def extract_text_from_docx(data):
    tmp = "temp_resume.docx"
    with open(tmp, "wb") as f:
        f.write(data)
    doc = docx.Document(tmp)
    os.remove(tmp)

    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    elif name.endswith(".docx") or name.endswith(".doc"):
        return extract_text_from_docx(data)
    else:
        try:
            return data.decode("utf-8")
        except:
            return str(data)

def find_email(text):
    match = EMAIL_RE.search(text)
    return match.group(0) if match else None

def extract_years_of_experience(text):
    match = re.search(r'(\d+)\+?\s*(years|year|yrs)', text, re.I)
    if match:
        return int(match.group(1))

    # Fallback: detect date ranges
    ranges = re.findall(r'(\b19\d{2}|\b20\d{2})\s*[-–]\s*(\b19\d{2}|\b20\d{2})', text)
    total_years = 0
    for start, end in ranges:
        try:
            total_years += max(0, int(end) - int(start))
        except:
            pass

    return total_years

def basic_skill_extract(text, top_n=30):
    tokens = re.findall(r"[A-Za-z+#\.\-]+", text.lower())
    from collections import Counter
    freq = Counter(tokens)
    return [token for token, _ in freq.most_common(top_n)]
