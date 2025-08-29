import os
from PyPDF2 import PdfReader
import docx
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import fitz 
import pytesseract
from pdf2image import convert_from_path
from io import BytesIO


def get_file_type(input_file):

    if isinstance(input_file, str) and os.path.exists(input_file):
        return "path"
    elif hasattr(input_file, "name"):
        return "file"
    else: 
        return ""

def clean_text(text):
    return " ". join(text.split())


def get_file_text(input_file):
    all_texts_list = []

    file_type = get_file_type(input_file)
    if file_type == "path":
        split_tuple = os.path.splitext(input_file)
        file_extension = split_tuple[1].lower()

        if file_extension == ".pdf":
            current_file_text = get_pdf_text(input_file)
        elif file_extension == ".docx":
            current_file_text = get_docx_text(input_file)
        elif file_extension == ".csv":
            current_file_text = get_csv_text(input_file)

        if current_file_text: 
            all_texts_list.append(current_file_text)

    return "\n\n".join(all_texts_list)
        


def get_pdf_text(input_file):
    text_parts = []

    if isinstance(input_file, str) and os.path.exists(input_file):
        with open(input_file, "rb") as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                page_content = page.extract_text()
                if page_content:
                    text_parts.append(clean_text(page_content))

    elif hasattr(input_file, "read"): 
        pdf_reader = PdfReader(input_file)
        for page in pdf_reader.pages:
            page_content = page.extract_text()
            if page_content:
                text_parts.append(clean_text(page_content))
    else:
        print("Input must be a PDF file path or a file object")

    return "\n".join(text_parts)


def get_docx_text(doc_file):
    all_paragraphs_text = [] 
    document = docx.Document(doc_file)
    for paragraph in document.paragraphs:
        if paragraph.text: 
            all_paragraphs_text.append(paragraph.text)
    return ' '.join(all_paragraphs_text)


def get_csv_text(file):
    try:
        df = pd.read_csv(file)
        text = df.to_string(index=False) 
        return text
    except Exception as e:
        print(f"Lỗi khi đọc tệp CSV: {e}")
        return ""


def get_text_chunks(all_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(all_text)


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base
