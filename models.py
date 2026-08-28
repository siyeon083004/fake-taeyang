"""
장산범 Persona Engine v1 - DB 모델

설계 문서 5장(DB 설계) 기준.
카테고리/소스/상태 값은 파이썬 상수 문자열로만 관리한다 (Enum으로 강제하면
나중에 카테고리 추가할 때 migration이 귀찮아지므로, 자유 문자열 + 상수 참고용).
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Text, Float, ForeignKey, DateTime, JSON
)
from sqlalchemy.orm import relationship as orm_relationship

from database import Base


def now_utc():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 상수 (문서 1-4, 1-5, 4장 기준)
# ---------------------------------------------------------------------------

class Source:
    DIRECT_CORRECTION = "DIRECT_CORRECTION"
    DIRECT_STATEMENT = "DIRECT_STATEMENT"
    OBSERVED = "OBSERVED"
    INFORMANT = "INFORMANT"
    INFERRED = "INFERRED"
    INITIAL_SEED = "INITIAL_SEED"

    # 우선순위 숫자가 낮을수록 신뢰도 높음 (충돌 판정에 사용)
    PRIORITY = {
        DIRECT_CORRECTION: 0,
        DIRECT_STATEMENT: 1,
        OBSERVED: 2,
        INFORMANT: 3,
        INFERRED: 4,
        INITIAL_SEED: 4,
    }


class ItemStatus:
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


class PersonaCategory:
    IDENTITY = "IDENTITY"
    LIFESTYLE = "LIFESTYLE"
    PERSONALITY = "PERSONALITY"
    EMOTION = "EMOTION"
    SPEECH = "SPEECH"
    BEHAVIOR = "BEHAVIOR"
    HUMOR = "HUMOR"
    RELATIONSHIP = "RELATIONSHIP"
    PREFERENCE = "PREFERENCE"
    OPINION = "OPINION"


class MemoryType:
    FACT = "FACT"
    EPISODE = "EPISODE"
    ANECDOTE = "ANECDOTE"
    EVENT = "EVENT"
    CONVERSATION = "CONVERSATION"
    RELATIONSHIP_MEMORY = "RELATIONSHIP_MEMORY"
    TEMPORARY = "TEMPORARY"


class LearningType:
    NEW_INFORMATION = "NEW_INFORMATION"
    CORRECTION = "CORRECTION"
    REINFORCEMENT = "REINFORCEMENT"
    CONFLICT = "CONFLICT"
    RECLASSIFICATION = "RECLASSIFICATION"


class LearningEventStatus:
    OPEN = "OPEN"
    DISCUSSING = "DISCUSSING"
    REVISED = "REVISED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# 5-1. personas
# ---------------------------------------------------------------------------

class Persona(Base):
    __tablename__ = "personas"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)          # 예: "태양"
    nickname = Column(String)                       # 예: "짭태양"
    description = Column(Text)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=now_utc)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc)

    items = orm_relationship("PersonaItem", back_populates="persona")
    memories = orm_relationship("Memory", back_populates="persona")
    conversations = orm_relationship("Conversation", back_populates="persona")
    relationships = orm_relationship("PersonaRelationship", back_populates="persona")


# ---------------------------------------------------------------------------
# 17. 관계형 ID가 없는 카카오톡 사용자 처리
#     - identities: 관측된 표시 이름(닉네임)들을 canonical target_key로 묶는다.
#       예: "챠" / "한이현" / "Mo" 전부 -> target_key="cha"
# ---------------------------------------------------------------------------

class Identity(Base):
    __tablename__ = "identities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    target_key = Column(String, nullable=False, index=True)   # canonical id, 예: "cha", "self"
    platform = Column(String, default="kakaotalk")
    display_name = Column(String, nullable=False, index=True)  # 관측된 이름 (닉네임 변경 이력 전부 보관)
    is_primary = Column(Integer, default=0)  # 1이면 현재 대표 표시 이름
    created_at = Column(DateTime, default=now_utc)


# ---------------------------------------------------------------------------
# 5-2. conversations (원본 대화, 삭제하지 않음)
# ---------------------------------------------------------------------------

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    persona_id = Column(Integer, ForeignKey("personas.id"), nullable=False)
    session_id = Column(String)          # 1:1 학습 세션 구분용
    room_id = Column(String)             # 단톡방 구분용
    speaker_id = Column(String)          # identities.target_key
    speaker_name = Column(String)        # 그 시점에 관측된 표시 이름 원본
    message = Column(Text, nullable=False)
    message_type = Column(String, default="text")
    created_at = Column(DateTime, default=now_utc)

    persona = orm_relationship("Persona", back_populates="conversations")


# ---------------------------------------------------------------------------
# 5-3. persona_items
# ---------------------------------------------------------------------------

class PersonaItem(Base):
    __tablename__ = "persona_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    persona_id = Column(Integer, ForeignKey("personas.id"), nullable=False)

    category = Column(String, nullable=False)      # PersonaCategory
    subcategory = Column(String)

    content = Column(Text, nullable=False)
    context = Column(Text)

    # BEHAVIOR 전용 구조 (문서 7장) - 해당 없는 카테고리는 전부 NULL
    trigger = Column(Text)
    interpretation = Column(Text)
    response_strategy = Column(Text)
    tone = Column(Text)
    follow_up = Column(Text)

    # RELATIONSHIP 카테고리 항목일 때 어떤 상대에 대한 특성인지 (없으면 전역 특성)
    target_key = Column(String, nullable=True)

    source = Column(String, nullable=False)          # Source
    source_conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)

    status = Column(String, default=ItemStatus.CANDIDATE)

    confidence = Column(Float, default=0.5)
    importance = Column(Float, default=0.5)
    evidence_count = Column(Integer, default=1)

    embedding = Column(JSON, nullable=True)   # float 리스트를 JSON으로 저장 (v1: SQLite, 추후 교체 가능)

    created_at = Column(DateTime, default=now_utc)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc)
    last_confirmed_at = Column(DateTime, nullable=True)

    persona = orm_relationship("Persona", back_populates="items")


# ---------------------------------------------------------------------------
# 11. relationships (Persona -> 특정 상대별 관계 데이터)
# ---------------------------------------------------------------------------

class PersonaRelationship(Base):
    __tablename__ = "relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    persona_id = Column(Integer, ForeignKey("personas.id"), nullable=False)
    target_key = Column(String, nullable=False, index=True)   # identities.target_key

    relation_type = Column(String)       # 예: "친구", "본인"
    intimacy = Column(String)            # LOW / MEDIUM / HIGH
    trust = Column(String)
    interaction_style = Column(Text)
    teasing_style = Column(Text)
    nickname = Column(String)            # 페르소나가 상대를 부르는 호칭
    shared_history = Column(Text)
    sensitive_topics = Column(Text)
    current_state = Column(Text)

    created_at = Column(DateTime, default=now_utc)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc)

    persona = orm_relationship("Persona", back_populates="relationships")


# ---------------------------------------------------------------------------
# 14. memories
# ---------------------------------------------------------------------------

class Memory(Base):
    # 주의: 기존(legacy) DB에 이미 "memories"라는 테이블(평문 저장용)이 존재하므로
    # 이름 충돌을 피하기 위해 새 스키마는 "persona_memories"를 사용한다.
    __tablename__ = "persona_memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    persona_id = Column(Integer, ForeignKey("personas.id"), nullable=False)

    memory_type = Column(String, nullable=False)   # MemoryType

    content = Column(Text, nullable=False)
    context = Column(Text)

    people_involved = Column(JSON, nullable=True)   # target_key 리스트
    event_time = Column(DateTime, nullable=True)

    source = Column(String, nullable=False)
    source_conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)

    importance = Column(Float, default=0.5)
    confidence = Column(Float, default=0.5)
    evidence_count = Column(Integer, default=1)

    status = Column(String, default=ItemStatus.CANDIDATE)

    embedding = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=now_utc)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc)
    last_referenced_at = Column(DateTime, nullable=True)

    persona = orm_relationship("Persona", back_populates="memories")


# ---------------------------------------------------------------------------
# 15. learning_events
# ---------------------------------------------------------------------------

class LearningEvent(Base):
    __tablename__ = "learning_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    persona_id = Column(Integer, ForeignKey("personas.id"), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)

    learning_type = Column(String, nullable=False)   # LearningType
    target_category = Column(String)
    target_subcategory = Column(String)

    old_value = Column(Text)
    proposed_value = Column(Text)

    reason = Column(Text)

    source = Column(String, nullable=False)
    confidence = Column(Float, default=0.5)

    status = Column(String, default=LearningEventStatus.OPEN)

    created_at = Column(DateTime, default=now_utc)
    resolved_at = Column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# 16. persona_history
# ---------------------------------------------------------------------------

class PersonaHistory(Base):
    __tablename__ = "persona_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    persona_item_id = Column(Integer, ForeignKey("persona_items.id"), nullable=False)

    previous_content = Column(Text)
    new_content = Column(Text)

    change_reason = Column(Text)
    learning_event_id = Column(Integer, ForeignKey("learning_events.id"), nullable=True)

    created_at = Column(DateTime, default=now_utc)
