import sqlite3
from datetime import datetime

DB_FILE = "taeyang.db"

def init_db():
    """DB 테이블 생성 (메시지 기록 & 기억 저장소)"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 1. 메시지 저장 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            sender TEXT,
            content TEXT,
            created_at TIMESTAMP
        )
    """)

    # 2. 상대방별 장기 기억 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            memory_text TEXT,
            created_at TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

def save_message(user_id: str, sender: str, content: str):
    """메시지 1건 저장"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (user_id, sender, content, created_at) VALUES (?, ?, ?, ?)",
        (user_id, sender, content, datetime.now())
    )
    conn.commit()
    conn.close()

def get_recent_messages(user_id: str, limit: int = 6):
    """최근 대화 N개 불러오기"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sender, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    rows.reverse()
    return rows

def save_memory(user_id: str, memory_text: str):
    """중요 기억 추가"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO memories (user_id, memory_text, created_at) VALUES (?, ?, ?)",
        (user_id, memory_text, datetime.now())
    )
    conn.commit()
    conn.close()

def get_memories(user_id: str):
    """해당 유저에 대한 모든 기억 불러오기"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT memory_text FROM memories WHERE user_id = ?",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

# DB 초기 세팅 실행
init_db()