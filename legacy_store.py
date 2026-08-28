import sqlite3
from datetime import datetime

DB_FILE = "taeyang.db"


def init_db():
    """DB 테이블 생성"""

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # 메시지 기록
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            sender TEXT,
            content TEXT,
            created_at TIMESTAMP
        )
    """)

    # 장기 기억
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            memory_text TEXT,
            created_at TIMESTAMP
        )
    """)

    # 말투 샘플
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS style_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT UNIQUE
        )
    """)

    conn.commit()
    conn.close()


def import_style_samples(filepath: str = "style_samples.txt"):
    """style_samples.txt를 말투 학습 데이터로 가져온다."""

    import os

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM style_samples"
    )

    count = cursor.fetchone()[0]

    if count > 0:
        conn.close()
        return count

    if not os.path.exists(filepath):
        conn.close()
        return 0

    with open(filepath, encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]

    cursor.executemany(
        """
        INSERT OR IGNORE INTO style_samples (message)
        VALUES (?)
        """,
        [(line,) for line in lines]
    )

    conn.commit()
    conn.close()

    return len(lines)


def get_random_style_samples(n: int = 12):
    """랜덤 말투 샘플."""

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT message
        FROM style_samples
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n,)
    )

    rows = cursor.fetchall()

    conn.close()

    return [row[0] for row in rows]


def get_relevant_style_samples(
    user_input: str,
    n: int = 12,
):
    """
    현재 입력과 겹치는 단어가 있는 말투 샘플을
    우선적으로 가져오고 부족하면 랜덤으로 채운다.
    """

    import re as _re

    words = [
        word
        for word in _re.split(
            r"\s+",
            user_input
        )
        if len(word) >= 2
    ][:5]

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    relevant = []

    if words:
        like_clauses = " OR ".join(
            ["message LIKE ?"] * len(words)
        )

        params = [
            f"%{word}%"
            for word in words
        ]

        cursor.execute(
            f"""
            SELECT message
            FROM style_samples
            WHERE {like_clauses}
            ORDER BY RANDOM()
            LIMIT ?
            """,
            params + [n // 2]
        )

        relevant = [
            row[0]
            for row in cursor.fetchall()
        ]

    remaining = n - len(relevant)

    if remaining > 0:
        cursor.execute(
            """
            SELECT message
            FROM style_samples
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (remaining,)
        )

        relevant += [
            row[0]
            for row in cursor.fetchall()
        ]

    conn.close()

    return relevant


def save_style_sample(text: str):
    """말투 샘플 저장."""

    if not text or not text.strip():
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR IGNORE INTO style_samples (message)
        VALUES (?)
        """,
        (text.strip(),)
    )

    conn.commit()
    conn.close()


def save_message(
    user_id: str,
    sender: str,
    content: str,
):
    """메시지 1건 저장."""

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO messages
        (user_id, sender, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            sender,
            content,
            datetime.now(),
        )
    )

    conn.commit()
    conn.close()


def get_recent_messages(
    user_id: str,
    limit: int = 6,
):
    """최근 대화 N개."""

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT sender, content
        FROM messages
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            user_id,
            limit,
        )
    )

    rows = cursor.fetchall()

    conn.close()

    rows.reverse()

    return rows


def save_memory(
    user_id: str,
    memory_text: str,
):
    """중요 기억 추가."""

    if not memory_text or not memory_text.strip():
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO memories
        (user_id, memory_text, created_at)
        VALUES (?, ?, ?)
        """,
        (
            user_id,
            memory_text.strip(),
            datetime.now(),
        )
    )

    conn.commit()
    conn.close()


def get_memories(user_id: str):
    """해당 유저의 기억."""

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT memory_text
        FROM memories
        WHERE user_id = ?
        ORDER BY id ASC
        """,
        (user_id,)
    )

    rows = cursor.fetchall()

    conn.close()

    return [
        row[0]
        for row in rows
    ]


def get_memories_with_id(user_id: str):
    """기억 ID + 내용."""

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id, memory_text
        FROM memories
        WHERE user_id = ?
        ORDER BY id ASC
        """,
        (user_id,)
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


def delete_memory_by_id(
    user_id: str,
    memory_id: int,
):
    """특정 기억 삭제."""

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM memories
        WHERE user_id = ?
        AND id = ?
        """,
        (
            user_id,
            memory_id,
        )
    )

    affected = cursor.rowcount

    conn.commit()
    conn.close()

    return affected > 0


init_db()
