"""
장산범 Persona Engine v2 - DB 모델

핵심 구조:
- Person: 봇이 알고 있는 '인간/인물' 자체
- Identity: 실제 카톡 sender와 Person을 연결하는 별칭/표시 이름
- PersonAlias: 채팅방에 없는 인물의 이름/별칭
- PersonaItem / Memory: Person의 이름과 독립된 페르소나 기억
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
# 상수
# ---------------------------------------------------------------------------

class Source:
    DIRECT_CORRECTION = "DIRECT_CORRECTION"
    DIRECT_STATEMENT = "DIRECT_STATEMENT"
    OBSERVED = "OBSERVED"
    INFORMANT = "INFORMANT"
    INFERRED = "INFERRED"
    INITIAL_SEED = "INITIAL_SEED"

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
# Persona
# ---------------------------------------------------------------------------

class Persona(Base):
    __tablename__ = "personas"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    nickname = Column(String)
    description = Column(Text)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=now_utc)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc)

    items = orm_relationship("PersonaItem", back_populates="persona")
    memories = orm_relationship("Memory", back_populates="persona")
    conversations = orm_relationship("Conversation", back_populates="persona")
    relationships = orm_relationship("PersonaRelationship", back_populates="persona")


# ---------------------------------------------------------------------------
# Person
#
# '실제 인간/인물' 자체.
# 카톡 표시 이름과 분리한다.
#
# 예:
# person_key = "self"
# canonical_name = "본인"
#
# person_key = "person_001"
# canonical_name = "백호"
# ---------------------------------------------------------------------------

class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_key = Column(String, nullable=False, unique=True, index=True)
    canonical_name = Column(String, nullable=False, index=True)

    person_type = Column(String, default="person")
    status = Column(String, default="active")

    # 채팅방에서 실제로 관측된 적 있는지
    observed_in_chat = Column(Integer, default=0)

    # 봇이 확실히 아는 인물인지
    confirmed = Column(Integer, default=0)

    notes = Column(Text)

    created_at = Column(DateTime, default=now_utc)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc)

    aliases = orm_relationship(
        "PersonAlias",
        back_populates="person",
        cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# PersonAlias
#
# 백호 = 배코 같은 별칭.
#
# 채팅방에 실제로 없는 사람도 여기에 등록 가능.
# ---------------------------------------------------------------------------

class PersonAlias(Base):
    __tablename__ = "person_aliases"

    id = Column(Integer, primary_key=True, autoincrement=True)

    person_id = Column(
        Integer,
        ForeignKey("persons.id"),
        nullable=False,
        index=True
    )

    alias = Column(String, nullable=False, unique=True, index=True)

    source = Column(String, default=Source.DIRECT_STATEMENT)
    confidence = Column(Float, default=1.0)

    created_at = Column(DateTime, default=now_utc)

    person = orm_relationship("Person", back_populates="aliases")


# ---------------------------------------------------------------------------
# Identity
#
# 실제 카톡 sender 표시 이름 -> Person
#
# 중요:
# Identity 삭제 = 이름 연결 삭제
# 대화/기억 삭제가 아니다.
# ---------------------------------------------------------------------------

class Identity(Base):
    __tablename__ = "identities"

    id = Column(Integer, primary_key=True, autoincrement=True)

    person_id = Column(
        Integer,
        ForeignKey("persons.id"),
        nullable=False,
        index=True
    )

    target_key = Column(
        String,
        nullable=False,
        index=True
    )

    platform = Column(String, default="kakaotalk")

    display_name = Column(
        String,
        nullable=False,
        index=True
    )

    is_primary = Column(Integer, default=0)

    created_at = Column(DateTime, default=now_utc)

    person = orm_relationship("Person")


# ---------------------------------------------------------------------------
# conversations
# ---------------------------------------------------------------------------

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    persona_id = Column(
        Integer,
        ForeignKey("personas.id"),
        nullable=False
    )

    session_id = Column(String)
    room_id = Column(String)

    speaker_id = Column(String)
    speaker_name = Column(String)

    message = Column(Text, nullable=False)
    message_type = Column(String, default="text")

    created_at = Column(DateTime, default=now_utc)

    persona = orm_relationship("Persona", back_populates="conversations")


# ---------------------------------------------------------------------------
# persona_items
# ---------------------------------------------------------------------------

class PersonaItem(Base):
    __tablename__ = "persona_items"

    id = Column(Integer, primary_key=True, autoincrement=True)

    persona_id = Column(
        Integer,
        ForeignKey("personas.id"),
        nullable=False
    )

    category = Column(String, nullable=False)
    subcategory = Column(String)

    content = Column(Text, nullable=False)
    context = Column(Text)

    trigger = Column(Text)
    interpretation = Column(Text)
    response_strategy = Column(Text)
    tone = Column(Text)
    follow_up = Column(Text)

    target_key = Column(String, nullable=True)

    source = Column(String, nullable=False)

    source_conversation_id = Column(
        Integer,
        ForeignKey("conversations.id"),
        nullable=True
    )

    status = Column(
        String,
        default=ItemStatus.CANDIDATE
    )

    confidence = Column(Float, default=0.5)
    importance = Column(Float, default=0.5)
    evidence_count = Column(Integer, default=1)

    embedding = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=now_utc)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc)
    last_confirmed_at = Column(DateTime, nullable=True)

    persona = orm_relationship(
        "Persona",
        back_populates="items"
    )


# ---------------------------------------------------------------------------
# relationships
# ---------------------------------------------------------------------------

class PersonaRelationship(Base):
    __tablename__ = "relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)

    persona_id = Column(
        Integer,
        ForeignKey("personas.id"),
        nullable=False
    )

    target_key = Column(String, nullable=False, index=True)

    relation_type = Column(String)
    intimacy = Column(String)
    trust = Column(String)

    interaction_style = Column(Text)
    teasing_style = Column(Text)

    nickname = Column(String)
    shared_history = Column(Text)

    sensitive_topics = Column(Text)
    current_state = Column(Text)

    created_at = Column(DateTime, default=now_utc)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc)

    persona = orm_relationship(
        "Persona",
        back_populates="relationships"
    )


# ---------------------------------------------------------------------------
# persona_memories
# ---------------------------------------------------------------------------

class Memory(Base):
    __tablename__ = "persona_memories"

    id = Column(Integer, primary_key=True, autoincrement=True)

    persona_id = Column(
        Integer,
        ForeignKey("personas.id"),
        nullable=False
    )

    memory_type = Column(String, nullable=False)

    content = Column(Text, nullable=False)
    context = Column(Text)

    people_involved = Column(JSON, nullable=True)

    event_time = Column(DateTime, nullable=True)

    source = Column(String, nullable=False)

    source_conversation_id = Column(
        Integer,
        ForeignKey("conversations.id"),
        nullable=True
    )

    importance = Column(Float, default=0.5)
    confidence = Column(Float, default=0.5)
    evidence_count = Column(Integer, default=1)

    status = Column(
        String,
        default=ItemStatus.CANDIDATE
    )

    embedding = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=now_utc)
    updated_at = Column(DateTime, default=now_utc, onupdate=now_utc)

    last_referenced_at = Column(DateTime, nullable=True)

    persona = orm_relationship(
        "Persona",
        back_populates="memories"
    )


# ---------------------------------------------------------------------------
# learning_events
# ---------------------------------------------------------------------------

class LearningEvent(Base):
    __tablename__ = "learning_events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    persona_id = Column(
        Integer,
        ForeignKey("personas.id"),
        nullable=False
    )

    conversation_id = Column(
        Integer,
        ForeignKey("conversations.id"),
        nullable=True
    )

    learning_type = Column(String, nullable=False)

    target_category = Column(String)
    target_subcategory = Column(String)

    old_value = Column(Text)
    proposed_value = Column(Text)

    reason = Column(Text)

    source = Column(String, nullable=False)

    confidence = Column(Float, default=0.5)

    status = Column(
        String,
        default=LearningEventStatus.OPEN
    )

    created_at = Column(DateTime, default=now_utc)
    resolved_at = Column(DateTime, nullable=True)


# ---------------------------------------------------------------------------
# persona_history
# ---------------------------------------------------------------------------

class PersonaHistory(Base):
    __tablename__ = "persona_history"

    id = Column(Integer, primary_key=True, autoincrement=True)

    persona_item_id = Column(
        Integer,
        ForeignKey("persona_items.id"),
        nullable=False
    )

    previous_content = Column(Text)
    new_content = Column(Text)

    change_reason = Column(Text)

    learning_event_id = Column(
        Integer,
        ForeignKey("learning_events.id"),
        nullable=True
    )

    created_at = Column(DateTime, default=now_utc)
