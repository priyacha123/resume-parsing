import pypdf 
import docx

def extract_text_from_pdf(file_obj):
    reader = pypdf.PdfReader(file_obj)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_obj):
    doc = docx.Document(file_obj)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_resume_text(file_field, filename):
    ext = filename.lower().rsplit('.', 1)[-1]
    file_field.seek(0)  # Ensure the file pointer is at the beginning
    if ext == 'pdf':
        return extract_text_from_pdf(file_field)
    elif ext == 'docx':
        return extract_text_from_docx(file_field)
    else:
        raise ValueError(f"Unsupported file type.")

