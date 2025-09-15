import psycopg2
import pickle
from src.config import HOST_NAME, DB_NAME, USER_NAME, PWD, PORT_ID
import os

def get_connection():
    return psycopg2.connect(
        host=HOST_NAME,
        database=DB_NAME,
        user=USER_NAME,
        password=PWD,
        port=PORT_ID
    )

def create_not_existed_table():
    """ Create a new table if not existed"""
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        create_script = '''
            CREATE TABLE IF NOT EXISTS knowledge_base(
                id          SERIAL PRIMARY KEY,
                text        TEXT NOT NULL,
                embedding   vector(384) NOT NULL
            );'''
        
        cur.execute(create_script)
        conn.commit()
        print("Đã kiểm tra và đảm bảo bảng 'knowledge_base' tồn tại.")

    except Exception as error:
        print(f"Lỗi khi tạo bảng: {error}")
        if conn:
            conn.rollback()
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
