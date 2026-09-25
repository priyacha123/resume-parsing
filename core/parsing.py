import io
import re
import pypdf
import docx

def sanitize_extracted_text(text: str) -> str:
    """Removes null bytes and cleans up excessive whitespace while preserving line structure."""
    if not text:
        return ""
    # Remove PostgreSQL-incompatible NUL characters
    text = text.replace('\x00', '')
    # Normalize carriage returns
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Replace non-breaking spaces
    text = text.replace('\xa0', ' ')
    # Remove lines with just strange control characters
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_text_from_pdf(file_obj) -> str:
    """Extracts text from all pages of a PDF file, handling seek and layout spaces."""
    try:
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        
        # Read into BytesIO to ensure a seekable stream for pypdf
        content = file_obj.read() if hasattr(file_obj, 'read') else file_obj
        if isinstance(content, (bytes, bytearray)):
            stream = io.BytesIO(content)
        else:
            stream = file_obj
            stream.seek(0)

        reader = pypdf.PdfReader(stream)
        if reader.is_encrypted:
            try:
                reader.decrypt('')
            except Exception:
                raise ValueError("PDF is password-protected and cannot be parsed.")

        page_texts = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_texts.append(page_text.strip())

        full_text = "\n\n".join(page_texts)
        return sanitize_extracted_text(full_text)
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Failed to read PDF file: {str(e)}")


def extract_text_from_docx(file_obj) -> str:
    """Extracts text from paragraphs AND tables in DOCX files."""
    try:
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        
        content = file_obj.read() if hasattr(file_obj, 'read') else file_obj
        if isinstance(content, (bytes, bytearray)):
            stream = io.BytesIO(content)
        else:
            stream = file_obj
            stream.seek(0)

        doc = docx.Document(stream)
        text_elements = []

        # Extract standard paragraphs
        for p in doc.paragraphs:
            clean_p = p.text.strip()
            if clean_p:
                text_elements.append(clean_p)

        # Extract table content (resumes heavily use tables for columns/layouts)
        for table in doc.tables:
            for row in table.rows:
                row_cells = []
                last_cell_text = ""
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    # Skip duplicate merged cells
                    if cell_text and cell_text != last_cell_text:
                        row_cells.append(cell_text)
                        last_cell_text = cell_text
                if row_cells:
                    text_elements.append(" | ".join(row_cells))

        full_text = "\n".join(text_elements)
        return sanitize_extracted_text(full_text)
    except Exception as e:
        raise ValueError(f"Failed to read DOCX file: {str(e)}")


def extract_text_from_txt(file_obj) -> str:
    """Extracts text from a plain text file."""
    try:
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
        content = file_obj.read() if hasattr(file_obj, 'read') else file_obj
        if isinstance(content, bytes):
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1', errors='ignore')
        else:
            text = str(content)
        return sanitize_extracted_text(text)
    except Exception as e:
        raise ValueError(f"Failed to read TXT file: {str(e)}")


def extract_resume_text(file_field, filename: str) -> str:
    """
    Extracts text from an uploaded resume file (PDF, DOCX, or TXT).
    Validates that non-empty, readable text was extracted.
    """
    if not filename or '.' not in filename:
        raise ValueError("Invalid file: missing file extension.")

    ext = filename.lower().rsplit('.', 1)[-1]
    
    if hasattr(file_field, 'seek'):
        file_field.seek(0)

    if ext == 'pdf':
        text = extract_text_from_pdf(file_field)
    elif ext in ('docx', 'doc'):
        text = extract_text_from_docx(file_field)
    elif ext == 'txt':
        text = extract_text_from_txt(file_field)
    else:
        raise ValueError(f"Unsupported file format '.{ext}'. Supported formats: PDF, DOCX, TXT.")

    if not text or len(text.strip()) < 20:
        raise ValueError(
            "Could not extract sufficient text from this file. "
            "If this is a scanned document (image-only), please use an OCR tool or upload a text-based document."
        )

    return text
