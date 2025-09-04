import os
import sys
# Thêm đường dẫn thư mục gốc của dự án vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from src.utils import get_file_text, get_text_chunks, save_embeddings_to_db
from src.db_connection import create_not_existed_table

app = FastAPI()

@app.on_event("startup")
def startup_event():
    """Tạo bảng database khi ứng dụng khởi động."""
    create_not_existed_table()

@app.get("/")
def health_check():
    """Kiểm tra trạng thái của API"""
    return {"status": "successful"}

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    """
    Xử lý tệp được tải lên, trích xuất văn bản, tạo chunks
    và lưu embeddings vào cơ sở dữ liệu.
    """
    try:
        text = await get_file_text(file)
        
        if not text: 
            return {"status": "error", "detail": "Không đọc được dữ liệu từ tệp."}

        text_chunks = await run_in_threadpool(get_text_chunks, text)
        if not text_chunks:
            return {"status": "error", "detail": "Không thể chia văn bản thành các chunks. Tệp có thể rỗng."}
        
        await run_in_threadpool(save_embeddings_to_db, text_chunks)
        
        return {"status": "success", "chunks_count": len(text_chunks)}

    except Exception as e:
        return {"status": "error", "detail": f"Đã xảy ra lỗi: {str(e)}"}
