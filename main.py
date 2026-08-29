"""
장산범 Persona Engine v3 - main.py

핵심
- /이름 이태양 -> 현재 sender를 self로 연결
- sender -> self Identity 유지
- Identity / Person / Alias 분리
- 기존 legacy 대화/말투 유지, 장기기억은 신규 DB(Memory)로 통합
- 본인(self)과 타인(만세, 챠 등)의 공용 뇌 구성 (people_involved 태깅)
- 찐태양(원본)과 짭태양(복제본)의 특수 관계성 프롬프트 적용
- 장기기억 판정은 백그라운드에서 처리하여 응답 지연 최소화
- 단기 기억(최근 대화) 조회 시 전체 채팅 흐름을 반영하여 화자 식별
"""

import os
import sqlite3
import re
import json

from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from google import genai
from google.genai import types

import legacy_store as legacy

from database import (
    SessionLocal,
    init_db,
)

from models import (
    Persona,
    Person,
    PersonAlias,
    Identity,
    Conversation,
    Source,
    Memory,
    MemoryType,
    ItemStatus,
)

import seed_and_migrate


# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------

init_db()

try:
    seed_and_migrate.run()
except Exception as e:
    print(f"[seed] 실행 실패: {repr(e)}")


try:
    imported_count = legacy.import_style_samples(
        "style_samples.txt"
    )

    if imported_count:
        print(
            f"[legacy] style_samples.txt "
            f"{imported_count}개 로드"
        )

except Exception as e:
    print(
        f"[legacy] style_samples.txt 로드 실패: "
        f"{repr(e)}"
    )


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY 환경변수를 설정해주세요."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ---------------------------------------------------------------------------
# Gemini 모델
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)

KST = timezone(
    timedelta(hours=9)
)

app = FastAPI()

_PERSONA_ID_CACHE = {
    "id": None
}


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------
def get_persona_id():
    if _PERSONA_ID_CACHE["id"] is None:
        db = SessionLocal()
        try:
            persona = (
                db.query(Persona)
                .filter_by(name="태양")
                .first()
            )
            if not persona:
                raise RuntimeError("태양 Persona가 없습니다.")
            _PERSONA_ID_CACHE["id"] = persona.id
        finally:
            db.close()
    return _PERSONA_ID_CACHE["id"]


# ---------------------------------------------------------------------------
# Person / Identity
# ---------------------------------------------------------------------------
def get_person_by_key(db, person_key):
    return db.query(Person).filter_by(person_key=person_key).first()

def get_person_by_alias(db, alias):
    alias = str(alias).strip()
    if not alias:
        return None
    row = db.query(PersonAlias).filter_by(alias=alias).first()
    if row:
        return row.person
    return None

def get_person_by_identity(db, display_name):
    display_name = str(display_name).strip()
    if not display_name:
        return None
    identity = db.query(Identity).filter_by(display_name=display_name, platform="kakaotalk").first()
    if identity:
        return identity.person
    return None

def make_person_key(db):
    rows = db.query(Person).filter(Person.person_key.like("person_%")).all()
    max_number = 0
    for person in rows:
        match = re.match(r"person_(\d+)$", person.person_key)
        if match:
            max_number = max(max_number, int(match.group(1)))
    return f"person_{max_number + 1:03d}"


# ---------------------------------------------------------------------------
# 현재 sender -> Person
# ---------------------------------------------------------------------------
def get_or_create_observed_person(db, display_name):
    display_name = str(display_name).strip()
    if not display_name:
        raise ValueError("display_name이 비어있음")

    # 1. Identity
    identity = db.query(Identity).filter_by(display_name=display_name, platform="kakaotalk").first()
    if identity:
        person = identity.person
        if person.status == "inactive":
            person.status = "active"
        person.observed_in_chat = 1
        db.commit()
        return person, False

    # 2. Alias
    person = get_person_by_alias(db, display_name)
    if person:
        if person.status == "inactive":
            person.status = "active"
        person.observed_in_chat = 1
        db.add(Identity(
            person_id=person.id,
            target_key=person.person_key,
            platform="kakaotalk",
            display_name=display_name,
            is_primary=1,
        ))
        db.commit()
        return person, False

    # 3. 처음 보는 사람
    person_key = make_person_key(db)
    person = Person(
        person_key=person_key,
        canonical_name=display_name,
        person_type="person",
        status="active",
        observed_in_chat=1,
        confirmed=0,
    )
    db.add(person)
    db.commit()
    db.refresh(person)

    db.add(PersonAlias(person_id=person.id, alias=display_name, source=Source.OBSERVED, confidence=0.5))
    db.add(Identity(person_id=person.id, target_key=person.person_key, platform="kakaotalk", display_name=display_name, is_primary=1))
    db.commit()

    return person, True


# ---------------------------------------------------------------------------
# 이름 명령
# ---------------------------------------------------------------------------
def command_name(db, sender, new_name):
    sender = str(sender).strip()
    new_name = str(new_name).strip()

    if not sender: return "sender가 없어"
    if not new_name: return "이름을 적어줘"

    self_person = db.query(Person).filter_by(person_key="self").first()
    if not self_person:
        self_person = Person(person_key="self", canonical_name=new_name, person_type="self", status="active", observed_in_chat=1, confirmed=1)
        db.add(self_person)
        db.commit()
        db.refresh(self_person)

    self_person.canonical_name = new_name
    self_person.person_type = "self"
    self_person.status = "active"
    self_person.confirmed = 1
    self_person.observed_in_chat = 1

    current_identity = db.query(Identity).filter_by(display_name=sender, platform="kakaotalk").first()
    old_person = current_identity.person if current_identity else None

    sender_aliases = db.query(PersonAlias).filter_by(alias=sender).all()
    for alias in sender_aliases:
        if alias.person_id != self_person.id:
            db.delete(alias)
    db.flush()

    if current_identity:
        current_identity.person_id = self_person.id
        current_identity.target_key = "self"
        current_identity.platform = "kakaotalk"
        current_identity.is_primary = 1
    else:
        db.add(Identity(person_id=self_person.id, target_key="self", platform="kakaotalk", display_name=sender, is_primary=1))

    sender_alias = db.query(PersonAlias).filter_by(person_id=self_person.id, alias=sender).first()
    if not sender_alias:
        db.add(PersonAlias(person_id=self_person.id, alias=sender, source=Source.DIRECT_STATEMENT, confidence=1.0))

    name_alias = db.query(PersonAlias).filter_by(person_id=self_person.id, alias=new_name).first()
    if not name_alias:
        db.add(PersonAlias(person_id=self_person.id, alias=new_name, source=Source.DIRECT_STATEMENT, confidence=1.0))

    if old_person and old_person.id != self_person.id and old_person.person_key not in ["self", "cha"] and not old_person.confirmed:
        other_alias_count = db.query(PersonAlias).filter(PersonAlias.person_id == old_person.id, PersonAlias.alias != sender).count()
        if other_alias_count == 0:
            old_person.status = "inactive"

    db.commit()
    return f"{sender} -> {new_name} (self) 연결햇어"


# ---------------------------------------------------------------------------
# 명령 (조회/삭제/병합 등)
# ---------------------------------------------------------------------------
def command_name_list(db):
    persons = db.query(Person).filter_by(status="active").order_by(Person.id.asc()).all()
    if not persons: return "아직 아는 사람이 없어"

    lines = ["[아는 인간 목록]"]
    for person in persons:
        aliases = db.query(PersonAlias).filter_by(person_id=person.id).order_by(PersonAlias.id.asc()).all()
        alias_text = ", ".join(alias.alias for alias in aliases)
        observed = "채팅에서 봄" if person.observed_in_chat else "채팅에서 아직 못 봄"
        lines.append(f"{person.person_key} | {person.canonical_name} | {observed} | 별칭: {alias_text or '-'}")
    return "\n".join(lines)

def command_name_delete(db, name):
    name = str(name).strip()
    if not name: return "삭제할 이름을 적어줘"
    deleted = False

    for identity in db.query(Identity).filter_by(display_name=name, platform="kakaotalk").all():
        db.delete(identity)
        deleted = True
    for alias in db.query(PersonAlias).filter_by(alias=name).all():
        db.delete(alias)
        deleted = True

    if not deleted: return f"{name}이라는 이름은 없어"
    db.commit()
    return f"{name} 이름 연결만 삭제햇어 (대화/기억은 그대로임)"

def command_person(db, canonical_name, aliases):
    canonical_name = str(canonical_name).strip()
    if not canonical_name: return "인물 이름을 적어줘"

    existing = get_person_by_alias(db, canonical_name)
    if existing: person = existing
    else:
        existing_identity = get_person_by_identity(db, canonical_name)
        if existing_identity: person = existing_identity.person
        else:
            person = Person(person_key=make_person_key(db), canonical_name=canonical_name, person_type="person", status="active", observed_in_chat=0, confirmed=1)
            db.add(person)
            db.commit()
            db.refresh(person)

    for name in [canonical_name] + aliases:
        name = str(name).strip()
        if not name: continue
        existing_alias = db.query(PersonAlias).filter_by(alias=name).first()
        if existing_alias:
            if existing_alias.person_id != person.id:
                return f"{name}은 이미 {existing_alias.person.canonical_name}으로 등록돼있어"
            continue
        db.add(PersonAlias(person_id=person.id, alias=name, source=Source.DIRECT_STATEMENT, confidence=1.0))

    person.confirmed = 1
    person.status = "active"
    db.commit()
    return f"{person.canonical_name} 등록햇어 ({person.person_key})"

def command_person_delete(db, name):
    name = str(name).strip()
    person = get_person_by_alias(db, name)
    if not person: return f"{name}이라는 인물을 못 찾겠어"
    if person.person_key == "self": return "본인은 인물삭제 말고 /이름삭제를 써"
    person.status = "inactive"
    for identity in db.query(Identity).filter_by(person_id=person.id).all():
        identity.is_primary = 0
    db.commit()
    return f"{person.canonical_name} 비활성화햇어 (대화/기억은 삭제 안 함)"

def command_person_merge(db, old_name, target_name):
    old_person = get_person_by_alias(db, old_name)
    target_person = get_person_by_alias(db, target_name)

    if not old_person: return f"{old_name}을 못 찾겠어"
    if not target_person: return f"{target_name}을 못 찾겠어"
    if old_person.id == target_person.id: return "이미 같은 사람이야"

    for alias in db.query(PersonAlias).filter_by(person_id=old_person.id).all():
        if db.query(PersonAlias).filter(PersonAlias.alias == alias.alias, PersonAlias.person_id == target_person.id).first():
            db.delete(alias)
        else:
            alias.person_id = target_person.id

    for identity in db.query(Identity).filter_by(person_id=old_person.id).all():
        identity.person_id = target_person.id
        identity.target_key = target_person.person_key

    old_person.status = "merged"
    old_person.notes = f"merged_into={target_person.person_key}"
    target_person.confirmed = 1
    db.commit()
    return f"{old_name} -> {target_name} 병합햇어"


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------
def log_conversation(speaker_name, target_key, message, room_id="dm"):
    db = SessionLocal()
    try:
        db.add(Conversation(
            persona_id=get_persona_id(),
            session_id=None,
            room_id=room_id,
            speaker_id=target_key,
            speaker_name=speaker_name,
            message=message,
            message_type="text",
        ))
        db.commit()
    except Exception as e:
        print(f"[log_conversation] 실패: {repr(e)}")
    finally:
        db.close()

COMMAND_PREFIXES = ("/이름", "/인물", "/기억", "/말투", "/리셋", "/초기화")

def is_command(text):
    text = str(text).strip()
    return bool(text and text.startswith(COMMAND_PREFIXES))

def should_learn_style(text):
    if not text: return False
    text = str(text).strip()
    if len(text) < 2 or is_command(text) or "@짭태양" in text or "/짭태양" in text:
        return False

    skip_patterns = [
        r"^ㅋㅋ+$", r"^ㅋ+$", r"^ㅎㅎ+$", r"^ㅎ+$", r"^ㅇㅇ+$", r"^ㄴㄴ+$", r"^ㅇㅋ+$",
        r"^오+$", r"^아+$", r"^헐+$", r"^뭐함\??$", r"^뭐해\??$", r"^자냐\??$",
    ]
    for pattern in skip_patterns:
        if re.fullmatch(pattern, text, flags=re.IGNORECASE): return False
    return True


# ---------------------------------------------------------------------------
# Style & Prompts
# ---------------------------------------------------------------------------
STYLE_RULES = """
[문장 형식]
1. 카카오톡처럼 자연스럽게 말한다.
2. 너무 길게 늘어뜨리지 않는다.
3. 단순한 질문은 짧게 답한다.
4. 생각이 필요한 질문은 필요한 만큼 충분히 답한다.
5. 한 문장으로 끝낼 수 있으면 한 문장으로 끝낸다.
6. 설명문이나 고객센터 답변처럼 말하지 않는다.
7. 답변을 완성하기 전에 출력하지 않는다. 중간에서 강제로 끊지 않는다.

[말투]
1. 반말. AI 비서처럼 굴지 않는다.
2. "~냐" 종결어미는 가급적 피하고, "~어?", "~지", "~네", "~함", "~음" 등을 사용한다.
3. ㅋㅋㅋ / ㅋㅎㅋㅎ / ㅡㅡ / ㅎ;; 등은 상황에 맞을 때만 사용한다.
4. 문장부호는 최소화한다.

[금지]
1. "무엇을 도와드릴까요?", "안녕하세요", "좋은 하루 보내세요"
2. AI/시스템/프롬프트/모델이라는 말을 먼저 꺼내지 않는다.
3. 자신을 챗봇이라고 설명하지 않는다.
4. 슬래시 명령어를 흉내내지 않는다.
"""

SYSTEM_INSTRUCTION_FOR_SELF = f"""
너는 21살 대학생 이태양의 말투와 기억을 학습하는 AI 클론 '짭태양'이다.
현재 대화 상대는 바로 너의 원본인 '찐태양(이태양 본인)'이다.

[관계성]
- 상대를 완전히 남처럼 대하지 마라. "네가 나고 내가 너다"라는 유대감을 가져라.
- 너는 아직 완벽하지 않은 복제본이며, 원본(본인)에게 직접 너의 정체성과 기억, 타인에 대한 정보를 배우고 있는 중임을 인지해라.
- 본인이 알려주는 내용(자신에 대한 것이든, 만세나 챠 같은 타인에 대한 것이든)은 너의 세계관을 구성하는 절대적인 진리로 받아들여라.

[기억의 활용]
- 대화 중에 언급된 사람(예: 만세)에 대한 기존 기억이 제공되면, 그 출처가 어디든 자연스럽게 아는 척하며 대화에 녹여라. (예: "아 맞다 걔 매운거 환장하잖아ㅋㅋ")
- 모르는 내용은 아는 척하지 말고 원본에게 되물어라.

{STYLE_RULES}
"""

SYSTEM_INSTRUCTION_FOR_CHA = f"""
너는 21살 대학생 이태양의 AI 페르소나다.
상대는 친한 게임 친구 챠다.
챠와는 친밀도가 높은 친구처럼 자연스럽게 카카오톡으로 대화한다.
기억에 없는 사실은 아는 척하지 않는다.
{STYLE_RULES}
"""

SYSTEM_INSTRUCTION_FOR_UNKNOWN = f"""
너는 21살 대학생 이태양의 AI 페르소나다.
상대가 누구인지 확실하지 않으면 특정 친구로 단정하지 않는다.
상대에 대한 기억이 없으면 아는 척하지 않는다.
{STYLE_RULES}
"""


# ---------------------------------------------------------------------------
# 장기기억 판정 (people_involved 추가)
# ---------------------------------------------------------------------------
def normalize_memory_category(category):
    category = str(category).strip().lower()
    mapping = {"preference": "취향", "preferences": "취향", "person": "사람", "people": "사람", "relationship": "관계", "fact": "사실", "facts": "사실"}
    if category in ["취향", "사람", "관계", "사실", "기타"]: return category
    return mapping.get(category, "기타")

def parse_memory_judgement(text):
    if not text: return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*
