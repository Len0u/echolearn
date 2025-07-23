import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        print(f"✅ Extracted text from PDF ({pdf_path})")
        return text
    except Exception as e:
        print(f"❌ Error extracting PDF text: {e}")
        return ""