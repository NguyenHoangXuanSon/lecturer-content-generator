# app.py

import os
import traceback
import json
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool
from typing import Optional, List
from src.utils import (
    get_file_text, get_text_chunks, save_embeddings_to_db,
    generate_lecture_content, retrive_data_from_db,
    LectureContent, MainPoint
)
from src.db_connection import create_not_existed_table, get_connection
from src.utils import validate_extension
app = FastAPI()

class UploadResponse(BaseModel):
    status: str = Field(..., description="Trạng thái của yêu cầu")
    lecture: Optional[LectureContent] = Field(None, description="Nội dung bài giảng được tạo ra")
    detail: Optional[str] = Field(None, description="Chi tiết lỗi nếu yêu cầu thất bại")

@app.post("/validate-extension/")
async def validate_file_extension(file: UploadFile = File(...)):
    file_extention = validate_extension(file)
    return {"status": "successfull", "extension": file_extention}

@app.post("/uploadfile/", response_model=UploadResponse)
async def upload_file_with_prompt(file: UploadFile = File(...), user_query: str = Form(...)):
    try:
        text = await get_file_text(file)
        if not text:
            return {"status": "error", "detail": "Không đọc được dữ liệu từ tệp."}
        
        text_chunks = await run_in_threadpool(get_text_chunks, text)
        if not text_chunks:
            return {"status": "error", "detail": "Không thể chia văn bản thành các chunks."}
        
        await run_in_threadpool(create_not_existed_table)
        await run_in_threadpool(save_embeddings_to_db, text_chunks)

        docs = await run_in_threadpool(retrive_data_from_db, user_query, 5)

        if not docs:
            return {"status": "error", "detail": "Không tìm thấy dữ liệu phù hợp trong DB."}

        lecture_content = await run_in_threadpool(generate_lecture_content, user_query)
        
        if not lecture_content:
            return {"status": "error", "detail": "Không thể tạo nội dung bài giảng. Kiểm tra log để biết thêm chi tiết."}

        return {"status": "success", "lecture": lecture_content}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "detail": str(e)}
