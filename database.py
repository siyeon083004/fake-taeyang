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

    # 3. 말투 학습용 예시 문장 테이블 (실제 카톡 대화에서 추출한 이태양 발화)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS style_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT UNIQUE
        )
    """)

    conn.commit()
    conn.close()


def import_style_samples(filepath: str = "style_samples.txt"):
    """
    style_samples.txt(한 줄에 문장 하나씩)를 읽어서 style_samples 테이블에 채워넣는다.
    이미 데이터가 들어있으면 다시 넣지 않는다 (서버 재시작할 때마다 중복 삽입 방지).
    """
    import os
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM style_samples")
    count = cursor.fetchone()[0]

    if count > 0:
        conn.close()
        return count

    if not os.path.exists(filepath):
        conn.close()
        return 0

    with open(filepath, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    cursor.executemany(
        "INSERT OR IGNORE INTO style_samples (message) VALUES (?)",
        [(line,) for line in lines]
    )
    conn.commit()
    conn.close()
    return len(lines)


def get_random_style_samples(n: int = 12):
    """말투 예시 문장을 랜덤으로 n개 뽑아온다."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT message FROM style_samples ORDER BY RANDOM() LIMIT ?", (n,))
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_relevant_style_samples(user_input: str, n: int = 12):
    """
    지금 들어온 메시지와 겹치는 단어가 있는 말투 샘플을 우선으로 가져오고,
    부족한 만큼은 랜덤으로 채운다.
    """
    import re as _re
    words = [w for w in _re.split(r"\s+", user_input) if len(w) >= 2][:5]

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    relevant = []
    if words:
        like_clauses = " OR ".join(["message LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words]
        cursor.execute(
            f"SELECT message FROM style_samples WHERE {like_clauses} ORDER BY RANDOM() LIMIT ?",
            params + [n // 2]
        )
        relevant = [r[0] for r in cursor.fetchall()]

    remaining = n - len(relevant)
    if remaining > 0:
        cursor.execute("SELECT message FROM style_samples ORDER BY RANDOM() LIMIT ?", (remaining,))
        relevant += [r[0] for r in cursor.fetchall()]

    conn.close()
    return relevant


def search_knowledge(query: str, limit: int = 6):
    """
    사용자 질문 속 주요 키워드를 바탕으로 과거 대화 데이터에서 배경 사실/정보를 검색한다.
    """
    import re as _re
    stop_words = {"이태양", "짭태양", "누구", "뭐야", "어때", "진짜", "어디", "무슨", "누구야", "알아"}
    words = [w for w in _re.findall(r'[가-힣a-zA-Z0-9]{2,}', query) if w not in stop_words]
    if not words:
        return []

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    conditions = " OR ".join(["message LIKE ?" for _ in words])
    params = [f"%{w}%" for w in words]

    cursor.execute(f"""
        SELECT message FROM style_samples 
        WHERE {conditions} 
        ORDER BY RANDOM() 
        LIMIT ?
    """, (*params, limit))

    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def save_style_sample(text: str):
    """이태양이 실제로 오늘 한 말을 말투 학습 데이터로 추가한다 (자동/수동 공용)."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO style_samples (message) VALUES (?)", (text,))
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
    """해당 유저에 대한 모든 기억 불러오기 (텍스트 리스트)"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT memory_text FROM memories WHERE user_id = ?",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_memories_with_id(user_id: str):
    """해당 유저의 기억 번호(ID)와 내용 함께 불러오기"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, memory_text FROM memories WHERE user_id = ? ORDER BY id ASC",
        (user_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_memory_by_id(user_id: str, memory_id: int):
    """특정 번호의 기억 삭제"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM memories WHERE user_id = ? AND id = ?",
        (user_id, memory_id)
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


init_db()
