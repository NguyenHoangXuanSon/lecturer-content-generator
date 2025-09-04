import psycopg2
import pickle
from src.config import hostname, database, username, pwd, port_id

def get_connection():
    return psycopg2.connect(
        host=hostname,
        database=database,
        user=username,
        password=pwd,
        port=port_id
    )

def create_not_existed_table():

    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        create_script = '''
            CREATE TABLE IF NOT EXISTS knowledge_base(
                id          SERIAL PRIMARY KEY,
                text        TEXT NOT NULL,
                embedding   BYTEA NOT NULL
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
