"""
DB 엔진 / 세션 설정.
기존 taeyang.db 파일명을 그대로 재사용한다 (같은 SQLite 파일에 새 테이블만 추가됨).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DB_PATH = "taeyang.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# SQLite + FastAPI(멀티스레드 워커) 조합에서는 check_same_thread=False 필요
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()


def get_db():
    """FastAPI Depends용 세션 제너레이터"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """models.py에 정의된 모든 테이블 생성 (이미 있으면 무시)"""
    import models  # noqa: F401  (모델을 Base 메타데이터에 등록시키기 위한 import)
    Base.metadata.create_all(bind=engine)
