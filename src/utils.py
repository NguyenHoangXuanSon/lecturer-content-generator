import os
import docx
import pandas as pd
import io
import re
import string
import pickle
import fitz 
import pytesseract
from io import BytesIO
from PIL import Image 
import psycopg2
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.db_connection import get_connection


def get_file_type(input_file):
    
    """Xác định loại đầu vào: đường dẫn tệp (str) hoặc đối tượng tệp."""

    if isinstance(input_file, str) and os.path.exists(input_file):
        return "path"
    elif hasattr(input_file, "filename"):
        return "file"
    else: 
        return ""

def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\[(.*?)\]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\.\.\.|…", "", text)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[\U00010000-\U0010FFFF]", "", text)
    return text


async def get_file_text(input_file):
    """
    Trích xuất văn bản từ một tệp đầu vào, có thể là đường dẫn
    hoặc một đối tượng tệp từ FastAPI.
    """
    file_type = get_file_type(input_file)
    file_extension = ""
    file_content = None

    if file_type == "path":
        file_extension = os.path.splitext(input_file)[1].lower()
        with open(input_file, 'rb') as f:
            file_content = f.read()
    elif file_type == "file":
        file_extension = os.path.splitext(input_file.filename)[1].lower()
        try:
            # Đọc nội dung bất đồng bộ từ UploadFile
            file_content = await input_file.read()
        except Exception as e:
            print(f"Error reading file content: {e}")
            return ""
    else:
        return ""

    if file_extension == ".pdf":
        current_file_text = get_pdf_text(file_content)
    elif file_extension == ".docx":
        current_file_text = get_docx_text(file_content)
    elif file_extension == ".csv":
        current_file_text = get_csv_text(file_content)
    else:
        current_file_text = ""
        
    return current_file_text


def get_pdf_text(file_content, lang="eng"):
    """
    Trích xuất văn bản từ tệp PDF, bao gồm cả OCR cho hình ảnh.
    """
    text_parts = []
    
    # Mở tệp từ luồng bytes
    doc = fitz.open(stream=file_content, filetype="pdf")

    for page in doc:
        page_text = page.get_text() or ""
        cleaned_text = clean_text(page_text)
        if cleaned_text:
            text_parts.append(cleaned_text)

        # OCR cho ảnh trong trang
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(pil_image, lang=lang)
            cleaned_text_ocr = clean_text(ocr_text)
            if cleaned_text_ocr:
                text_parts.append(cleaned_text_ocr)

    return "\n".join(text_parts)

def get_docx_text(file_content):

    """Trích xuất văn bản từ tệp DOCX."""

    all_paragraphs_text = [] 
    document = docx.Document(io.BytesIO(file_content))
    for paragraph in document.paragraphs:
        if paragraph.text: 
            all_paragraphs_text.append(paragraph.text)
    return ' '.join(all_paragraphs_text)

def get_csv_text(file_content):
    """Trích xuất văn bản từ tệp CSV."""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        text = df.to_string(index=False) 
        return text
    except Exception as e:
        print(f"Lỗi khi đọc tệp CSV: {e}")
        return ""

def get_text_chunks(all_text):
    """Chia văn bản thành các chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(all_text)

def save_embeddings_to_db(text_chunks):
    embeddings_model = HuggingFaceEmbeddings()

    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        insert_script = "INSERT INTO knowledge_base (text, embedding) VALUES (%s, %s)"
        values_to_insert = []
        for text in text_chunks:
            vector = embeddings_model.embed_query(text)
            vector_bytes = pickle.dumps(vector)
            values_to_insert.append((text, psycopg2.Binary(vector_bytes)))
        cur.executemany(insert_script, values_to_insert)
        conn.commit()
        print(f"Đã lưu {len(text_chunks)} embeddings vào database thành công.")

    except Exception as error:
        print("Lỗi khi kết nối với PostgreSQL")
        if conn:
            conn.rollback()

    finally:
        conn.close()
        cur.close()
