import os
import docx
import pandas as pd
import io
import re
import string
import fitz 
import pytesseract
from io import BytesIO
from PIL import Image 
import json
import re
import psycopg2
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.db_connection import get_connection
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, Field
from typing import List
from src.config import GEMINI_API_KEY
import google.generativeai as genai
from fastapi import HTTPException

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".csv"}

def get_file_type(input_file):
    
    """Identify the input type is file path or file object """

    if isinstance(input_file, str) and os.path.exists(input_file):
        return "path"
    elif hasattr(input_file, "filename"):
        return "file"
    else: 
        return ""
    
def validate_extension(input_file):
    file_type = get_file_type(input_file)
    file_extension = ""
    
    if file_type == "path":
        file_extension = os.path.splitext(input_file)[1].lower()
    elif file_type == "file":
        file_extension = os.path.splitext(input_file.filename)[1].lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"The file extension is not supported: {file_extension}. Please using allowed file extension: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    return file_extension
        

def clean_text(text):

    """Clean text from reader"""

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
    Exact text from an input file, which can be either a file path or a FastAPI file object
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
    Exact test from PDFs, include text and images
    """
    text_parts = []
    
    doc = fitz.open(stream=file_content, filetype="pdf")

    for page in doc:
        page_text = page.get_text() or ""
        cleaned_text = clean_text(page_text)
        if cleaned_text:
            text_parts.append(cleaned_text)

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

    """Exact test from DOCX."""

    all_paragraphs_text = [] 
    document = docx.Document(io.BytesIO(file_content))
    for paragraph in document.paragraphs:
        if paragraph.text: 
            all_paragraphs_text.append(paragraph.text)
    return ' '.join(all_paragraphs_text)

def get_csv_text(file_content):

    """Exact test from CSV."""

    try:
        df = pd.read_csv(io.BytesIO(file_content))
        text = df.to_string(index=False) 
        return text
    except Exception as e:
        print(f"Lỗi khi đọc tệp CSV: {e}")
        return ""

def get_text_chunks(all_text):

    """Chunk the text"""

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(all_text)

def save_embeddings_to_db(text_chunks):

    """ Save the embeddings to the database """

    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        insert_script = "INSERT INTO knowledge_base (text, embedding) VALUES (%s, %s)"
        values_to_insert = []

        for text in text_chunks:
            vector = embeddings_model.embed_query(text)
            values_to_insert.append((text, vector))

        cur.executemany(insert_script, values_to_insert)
        conn.commit()

        print(f"Đã lưu {len(text_chunks)} embeddings vào database thành công.")

    except Exception as error:
        print("Lỗi", error)
        if conn:
            conn.rollback()

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def get_query_embedding(user_query: str)->list[float]:

    """Exact user query into embedding vector"""

    return embeddings_model.embed_query(user_query)

class MainPoint(BaseModel):
    title: str = Field(..., description="Title for the mainpoint")
    content: str = Field(..., description="Detailed explanation of the slide")
    example: str = Field(..., description="Example or illustration for this slide")

class GameContent(BaseModel):
    title: str
    description: str
    instruction: str

class LectureContent(BaseModel):
    title: str
    intro: str
    main_points: List[MainPoint]
    game: GameContent  
    conclusion: str


def to_pgvector(vector: list[float]) -> str:

    """Convert a Python list into a string formatted for PostgreSQL """

    return "[" + ",".join(str(round(x, 6)) for x in vector) + "]"

def retrive_data_from_db(user_query: str, top_k):

    """Find top_k vectors that most relevant to the prompt from user"""
    
    query_embedding = get_query_embedding(user_query)

    conn = None
    cur = None 
    result = []

    try: 
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        retrieval_script = """
        SELECT id, text, 1 - (embedding <=> %s) AS cosine_similarity
        FROM knowledge_base
        ORDER BY embedding <=> %s
        LIMIT %s;
        """
        #cur.execute(retrival_script, (query_embedding, query_embedding, top_k))
        pg_vector = to_pgvector(query_embedding)
        cur.execute(retrieval_script, (pg_vector, pg_vector, top_k))

        result= cur.fetchall()

    except Exception as error:
        print("Error", error)
        if conn:
            conn.rollback()
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
    return result

import json
import re

def safe_json_loads(generated_text: str):

    """Convert output text from Gemini into safe Json"""
    if not generated_text:
        print("Chuỗi đầu vào rỗng")

    cleaned_text = re.sub(r"^```[a-zA-Z]*\n?", "", generated_text.strip())
    cleaned_text = re.sub(r"```$", "", cleaned_text.strip())

    try:
        cleaned_text_json = json.loads(cleaned_text)
        return cleaned_text_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Lỗi khi parse JSON: {e}\nNội dung sau khi làm sạch:\n{cleaned_text}")


def generate_lecture_content(user_query: str, top_k=5):

    """Generate the lecture from user query and top_k most relevant vectors"""
    docs = retrive_data_from_db(user_query, top_k)
    context = "\n\n".join([doc["text"] for doc in docs])

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
            Bạn là một trợ lý giảng dạy. 
            Dựa trên nội dung sau, hãy tạo một **Bài giảng hoàn chỉnh bằng tiếng Việt**.

            Nội dung tham khảo:
            {context}

            Yêu cầu xuất kết quả dưới dạng **JSON hợp lệ**, theo đúng cấu trúc sau:

            {{
            "title": "Tiêu đề bài giảng",
            "intro": "Phần giới thiệu ngắn gọn, hấp dẫn",
            "main_points": [
                {{
                "title": "Tiêu đề ý chính",
                "content": "Giải thích chi tiết cho ý chính",
                "example": "Ví dụ minh họa cụ thể"
                }}
            ],
            "game": {{
                "title": "Tên trò chơi",
                "description": "Mô tả chi tiết trò chơi",
                "instruction": "Cách thực hiện"
            }},
            "conclusion": "Phần kết luận cô đọng, súc tích"
            }}
            """

    response = model.generate_content(contents=prompt)
    generated_text = response.text

    if not generated_text:
        print("Chuỗi trả về rỗng")
        return None
    
    try:
        lecture_data = safe_json_loads(generated_text)
        return lecture_data
    except ValueError as e:
        print(e)
        print("Original text")
        print(generated_text)
        return None
