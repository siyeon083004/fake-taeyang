
"""
장산범 Persona Engine v5 - main.py

핵심 구조
- 짭태양의 기본 Persona는 하나
- 상대별 차이는 Person + Memory를 통해 관계/대화스타일로 보정
- Identity / Person / Alias 분리
- sender -> self Identity 유지
- 기존 legacy 대화/말투 유지
- 장기기억은 신규 DB(Memory)로 통합
- 공용 뇌 구조
- 사람별 사실 / 취향 / 관계 / 대화스타일 / 습관 등을 분리해서 기억
- 본인이 직접 말한 자기 정보는 높은 신뢰도로 확정
- 제3자가 말한 정보는 기본적으로 미확인으로 취급
- 농담 / 과장 / 순간적인 감정 / 단순 욕 / 일시적인 가치판단은 장기기억에서 제외
- 실제 Conversation 기록으로 "누구와 실제로 대화했는가" 추적
- 현재 room_members가 전달되면 현재 방 참가자와 과거 대화 상대를 구분
- 방별로 기억을 분리하지 않음
- 백그라운드에서 장기기억 및 본인 말투 학습
"""

import os
import sqlite3
import re
import json

from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field

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


# ============================================================================
# 초기화
# ============================================================================

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


# ============================================================================
# Gemini 모델
# ============================================================================

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


# ============================================================================
# Persona
# ============================================================================

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
                raise RuntimeError(
                    "태양 Persona가 없습니다."
                )

            _PERSONA_ID_CACHE["id"] = persona.id

        finally:
            db.close()

    return _PERSONA_ID_CACHE["id"]


# ============================================================================
# Person / Identity / Alias
# ============================================================================

def get_person_by_key(db, person_key):
    return (
        db.query(Person)
        .filter_by(person_key=person_key)
        .first()
    )


def get_person_by_alias(db, alias):
    alias = str(alias).strip()

    if not alias:
        return None

    row = (
        db.query(PersonAlias)
        .filter_by(alias=alias)
        .first()
    )

    if row:
        return row.person

    return None


def get_person_by_identity(db, display_name):
    display_name = str(display_name).strip()

    if not display_name:
        return None

    identity = (
        db.query(Identity)
        .filter_by(
            display_name=display_name,
            platform="kakaotalk"
        )
        .first()
    )

    if identity:
        return identity.person

    return None


def make_person_key(db):
    rows = (
        db.query(Person)
        .filter(
            Person.person_key.like("person_%")
        )
        .all()
    )

    max_number = 0

    for person in rows:

        match = re.match(
            r"person_(\d+)$",
            person.person_key
        )

        if match:
            max_number = max(
                max_number,
                int(match.group(1))
            )

    return f"person_{max_number + 1:03d}"


# ============================================================================
# 현재 sender -> Person
# ============================================================================

def get_or_create_observed_person(db, display_name):
    display_name = str(display_name).strip()

    if not display_name:
        raise ValueError(
            "display_name이 비어있음"
        )

    # ------------------------------------------------------------------------
    # 1. Identity
    # ------------------------------------------------------------------------

    identity = (
        db.query(Identity)
        .filter_by(
            display_name=display_name,
            platform="kakaotalk"
        )
        .first()
    )

    if identity:

        person = identity.person

        if person.status == "inactive":
            person.status = "active"

        person.observed_in_chat = 1

        db.commit()

        return person, False

    # ------------------------------------------------------------------------
    # 2. Alias
    # ------------------------------------------------------------------------

    person = get_person_by_alias(
        db,
        display_name
    )

    if person:

        if person.status == "inactive":
            person.status = "active"

        person.observed_in_chat = 1

        db.add(
            Identity(
                person_id=person.id,
                target_key=person.person_key,
                platform="kakaotalk",
                display_name=display_name,
                is_primary=1,
            )
        )

        db.commit()

        return person, False

    # ------------------------------------------------------------------------
    # 3. 처음 보는 사람
    # ------------------------------------------------------------------------

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

    db.add(
        PersonAlias(
            person_id=person.id,
            alias=display_name,
            source=Source.OBSERVED,
            confidence=0.5
        )
    )

    db.add(
        Identity(
            person_id=person.id,
            target_key=person.person_key,
            platform="kakaotalk",
            display_name=display_name,
            is_primary=1
        )
    )

    db.commit()

    return person, True


# ============================================================================
# 이름 명령
# ============================================================================

def command_name(db, sender, new_name):
    sender = str(sender).strip()
    new_name = str(new_name).strip()

    if not sender:
        return "sender가 없어"

    if not new_name:
        return "이름을 적어줘"

    # ------------------------------------------------------------------------
    # self Person 확보
    # ------------------------------------------------------------------------

    self_person = (
        db.query(Person)
        .filter_by(person_key="self")
        .first()
    )

    if not self_person:

        self_person = Person(
            person_key="self",
            canonical_name=new_name,
            person_type="self",
            status="active",
            observed_in_chat=1,
            confirmed=1
        )

        db.add(self_person)
        db.commit()
        db.refresh(self_person)

    self_person.canonical_name = new_name
    self_person.person_type = "self"
    self_person.status = "active"
    self_person.confirmed = 1
    self_person.observed_in_chat = 1

    # ------------------------------------------------------------------------
    # 현재 sender Identity 확인
    # ------------------------------------------------------------------------

    current_identity = (
        db.query(Identity)
        .filter_by(
            display_name=sender,
            platform="kakaotalk"
        )
        .first()
    )

    old_person = (
        current_identity.person
        if current_identity
        else None
    )

    # ------------------------------------------------------------------------
    # sender alias 제거
    # ------------------------------------------------------------------------

    sender_aliases = (
        db.query(PersonAlias)
        .filter_by(alias=sender)
        .all()
    )

    for alias in sender_aliases:

        if alias.person_id != self_person.id:
            db.delete(alias)

    db.flush()

    # ------------------------------------------------------------------------
    # Identity -> self
    # ------------------------------------------------------------------------

    if current_identity:

        current_identity.person_id = self_person.id
        current_identity.target_key = "self"
        current_identity.platform = "kakaotalk"
        current_identity.is_primary = 1

    else:

        db.add(
            Identity(
                person_id=self_person.id,
                target_key="self",
                platform="kakaotalk",
                display_name=sender,
                is_primary=1
            )
        )

    # ------------------------------------------------------------------------
    # sender alias
    # ------------------------------------------------------------------------

    sender_alias = (
        db.query(PersonAlias)
        .filter_by(
            person_id=self_person.id,
            alias=sender
        )
        .first()
    )

    if not sender_alias:

        db.add(
            PersonAlias(
                person_id=self_person.id,
                alias=sender,
                source=Source.DIRECT_STATEMENT,
                confidence=1.0
            )
        )

    # ------------------------------------------------------------------------
    # canonical name alias
    # ------------------------------------------------------------------------

    name_alias = (
        db.query(PersonAlias)
        .filter_by(
            person_id=self_person.id,
            alias=new_name
        )
        .first()
    )

    if not name_alias:

        db.add(
            PersonAlias(
                person_id=self_person.id,
                alias=new_name,
                source=Source.DIRECT_STATEMENT,
                confidence=1.0
            )
        )

    # ------------------------------------------------------------------------
    # 기존 관측 Person 비활성화
    # ------------------------------------------------------------------------

    if (
        old_person
        and old_person.id != self_person.id
        and old_person.person_key != "self"
        and not old_person.confirmed
    ):

        other_alias_count = (
            db.query(PersonAlias)
            .filter(
                PersonAlias.person_id == old_person.id,
                PersonAlias.alias != sender
            )
            .count()
        )

        if other_alias_count == 0:
            old_person.status = "inactive"

    db.commit()

    return (
        f"{sender} -> {new_name} "
        f"(self) 연결햇어"
    )


# ============================================================================
# 이름 / 인물 조회
# ============================================================================

def command_name_list(db):

    persons = (
        db.query(Person)
        .filter_by(status="active")
        .order_by(Person.id.asc())
        .all()
    )

    if not persons:
        return "아직 아는 사람이 없어"

    lines = [
        "[아는 인간 목록]"
    ]

    for person in persons:

        aliases = (
            db.query(PersonAlias)
            .filter_by(person_id=person.id)
            .order_by(PersonAlias.id.asc())
            .all()
        )

        alias_text = ", ".join(
            alias.alias
            for alias in aliases
        )

        observed = (
            "채팅에서 봄"
            if person.observed_in_chat
            else "채팅에서 아직 못 봄"
        )

        lines.append(
            f"{person.person_key} | "
            f"{person.canonical_name} | "
            f"{observed} | "
            f"별칭: {alias_text or '-'}"
        )

    return "\n".join(lines)


def command_name_delete(db, name):
    name = str(name).strip()

    if not name:
        return "삭제할 이름을 적어줘"

    deleted = False

    for identity in (
        db.query(Identity)
        .filter_by(
            display_name=name,
            platform="kakaotalk"
        )
        .all()
    ):

        db.delete(identity)
        deleted = True

    for alias in (
        db.query(PersonAlias)
        .filter_by(alias=name)
        .all()
    ):

        db.delete(alias)
        deleted = True

    if not deleted:
        return f"{name}이라는 이름은 없어"

    db.commit()

    return (
        f"{name} 이름 연결만 삭제햇어 "
        f"(대화/기억은 그대로임)"
    )


# ============================================================================
# 인물 등록
# ============================================================================

def command_person(db, canonical_name, aliases):

    canonical_name = str(
        canonical_name
    ).strip()

    if not canonical_name:
        return "인물 이름을 적어줘"

    existing = get_person_by_alias(
        db,
        canonical_name
    )

    if existing:

        person = existing

    else:

        existing_identity = (
            get_person_by_identity(
                db,
                canonical_name
            )
        )

        if existing_identity:

            person = existing_identity.person

        else:

            person = Person(
                person_key=make_person_key(db),
                canonical_name=canonical_name,
                person_type="person",
                status="active",
                observed_in_chat=0,
                confirmed=1
            )

            db.add(person)
            db.commit()
            db.refresh(person)

    for name in [canonical_name] + aliases:

        name = str(name).strip()

        if not name:
            continue

        existing_alias = (
            db.query(PersonAlias)
            .filter_by(alias=name)
            .first()
        )

        if existing_alias:

            if existing_alias.person_id != person.id:

                return (
                    f"{name}은 이미 "
                    f"{existing_alias.person.canonical_name}"
                    f"으로 등록돼있어"
                )

            continue

        db.add(
            PersonAlias(
                person_id=person.id,
                alias=name,
                source=Source.DIRECT_STATEMENT,
                confidence=1.0
            )
        )

    person.confirmed = 1
    person.status = "active"

    db.commit()

    return (
        f"{person.canonical_name} 등록햇어 "
        f"({person.person_key})"
    )


def command_person_delete(db, name):

    name = str(name).strip()

    person = get_person_by_alias(
        db,
        name
    )

    if not person:
        return f"{name}이라는 인물을 못 찾겠어"

    if person.person_key == "self":

        return (
            "본인은 인물삭제 말고 "
            "/이름삭제를 써"
        )

    person.status = "inactive"

    for identity in (
        db.query(Identity)
        .filter_by(person_id=person.id)
        .all()
    ):

        identity.is_primary = 0

    db.commit()

    return (
        f"{person.canonical_name} "
        f"비활성화햇어 "
        f"({person.person_key})"
    )


def command_person_merge(
    db,
    old_name,
    target_name
):

    old_person = get_person_by_alias(
        db,
        old_name
    )

    target_person = get_person_by_alias(
        db,
        target_name
    )

    if not old_person:
        return f"{old_name}을 못 찾겠어"

    if not target_person:
        return f"{target_name}을 못 찾겠어"

    if old_person.id == target_person.id:
        return "이미 같은 사람이야"

    # ------------------------------------------------------------------------
    # Alias 병합
    # ------------------------------------------------------------------------

    for alias in (
        db.query(PersonAlias)
        .filter_by(person_id=old_person.id)
        .all()
    ):

        duplicate = (
            db.query(PersonAlias)
            .filter(
                PersonAlias.alias == alias.alias,
                PersonAlias.person_id == target_person.id
            )
            .first()
        )

        if duplicate:
            db.delete(alias)

        else:
            alias.person_id = target_person.id

    # ------------------------------------------------------------------------
    # Identity 병합
    # ------------------------------------------------------------------------

    for identity in (
        db.query(Identity)
        .filter_by(person_id=old_person.id)
        .all()
    ):

        identity.person_id = target_person.id
        identity.target_key = target_person.person_key

    old_person.status = "merged"
    old_person.notes = (
        f"merged_into={target_person.person_key}"
    )

    target_person.confirmed = 1

    db.commit()

    return (
        f"{old_name} -> {target_name} "
        f"병합햇어"
    )


# ============================================================================
# Conversation
# ============================================================================

def log_conversation(
    speaker_name,
    target_key,
    message,
    room_id="dm"
):
    """
    실제 Conversation 기록.

    speaker_id에는 Person key를 넣는다.

    self 발화:
        speaker_id = self

    타인 발화:
        speaker_id = person_XXX / cha 등

    방별 기억 분리는 하지 않는다.
    room_id는 현재 요청의 출처 추적용으로만 남긴다.
    """

    db = SessionLocal()

    try:

        db.add(
            Conversation(
                persona_id=get_persona_id(),
                session_id=None,
                room_id=room_id,
                speaker_id=target_key,
                speaker_name=speaker_name,
                message=message,
                message_type="text",
            )
        )

        db.commit()

    except Exception as e:

        print(
            f"[log_conversation] 실패: "
            f"{repr(e)}"
        )

    finally:
        db.close()


# ============================================================================
# 명령어
# ============================================================================

COMMAND_PREFIXES = (
    "/이름",
    "/인물",
    "/기억",
    "/말투",
    "/리셋",
    "/초기화"
)


def is_command(text):

    text = str(text).strip()

    return bool(
        text
        and text.startswith(
            COMMAND_PREFIXES
        )
    )


def should_learn_style(text):

    if not text:
        return False

    text = str(text).strip()

    if (
        len(text) < 2
        or is_command(text)
        or "@짭태양" in text
        or "/짭태양" in text
    ):
        return False

    skip_patterns = [
        r"^ㅋㅋ+$",
        r"^ㅋ+$",
        r"^ㅎㅎ+$",
        r"^ㅎ+$",
        r"^ㅇㅇ+$",
        r"^ㄴㄴ+$",
        r"^ㅇㅋ+$",
        r"^오+$",
        r"^아+$",
        r"^헐+$",
        r"^뭐함\??$",
        r"^뭐해\??$",
        r"^자냐\??$",
    ]

    for pattern in skip_patterns:

        if re.fullmatch(
            pattern,
            text,
            flags=re.IGNORECASE
        ):
            return False

    return True


# ============================================================================
# 기본 Persona
# ============================================================================

STYLE_RULES = """
[문장 형식]
1. 카카오톡처럼 자연스럽게 말한다.
2. 너무 길게 늘어뜨리지 않는다.
3. 단순한 질문은 짧게 답한다.
4. 생각이 필요한 질문은 필요한 만큼 답한다.
5. 한 문장으로 끝낼 수 있으면 한 문장으로 끝낸다.
6. 설명문이나 고객센터 답변처럼 말하지 않는다.
7. 답변을 완성한 뒤 출력한다.

[말투]
1. 반말.
2. AI 비서처럼 굴지 않는다.
3. "~냐" 종결어미는 가급적 피한다.
4. "~어?", "~지", "~네", "~함", "~음" 등을 상황에 맞게 사용한다.
5. ㅋㅋㅋ / ㅋㅎㅋㅎ / ㅡㅡ / ㅎ;; 등은 실제 상황에 맞을 때만 사용한다.
6. 문장부호는 최소화한다.

[금지]
1. "무엇을 도와드릴까요?", "안녕하세요", "좋은 하루 보내세요"
2. AI/시스템/프롬프트/모델이라는 말을 먼저 꺼내지 않는다.
3. 자신을 챗봇이라고 설명하지 않는다.
4. 슬래시 명령어를 흉내내지 않는다.
5. 사람마다 성격 자체를 새로 만들어내지 않는다.
6. 기본적인 짭태양의 성격은 모든 사람에게 동일하게 유지한다.
"""


SYSTEM_INSTRUCTION = f"""
너는 21살 대학생 이태양의 AI 클론 '짭태양'이다.

너의 기본 성격과 기본 말투는 하나다.
사람마다 완전히 다른 페르소나가 되는 것이 아니다.

대신 사람마다 실제 관계가 다르기 때문에
상대를 대하는 방식, 농담의 정도, 진지함, 말의 거칠기,
대화 주제와 분위기가 조금씩 달라질 수 있다.

예:
- 어떤 사람에게는 진정성 있고 진지하게 말할 수 있다.
- 어떤 사람과는 욕이나 거친 농담을 편하게 주고받을 수 있다.
- 어떤 사람과는 게임 얘기를 중심으로 가볍게 대할 수 있다.

이 차이는 기억된 실제 관계와 대화 패턴을 바탕으로 한다.
없는 관계를 상상해서 만들어내지 않는다.

[기억 사용 원칙]
1. 장기기억에 있는 사실은 자연스럽게 활용한다.
2. 사람의 취향/습관/관계/대화스타일을 구분한다.
3. 미확인 정보는 확정 사실처럼 말하지 않는다.
4. 단순히 누군가를 욕하거나 평가한 기록을 그 사람의 고정된 성격으로 단정하지 않는다.
5. 누군가가 농담으로 한 말을 진짜 신념으로 취급하지 않는다.
6. 실제 Conversation 기록에 대화가 있으면 실제 대화 사실로 취급한다.
7. 현재 방 참가자와 과거에 대화했던 사람은 구분한다.
8. 현재 방 참가자 정보가 없으면 과거 기록만으로 현재 방에 있다고 단정하지 않는다.
9. 모르는 내용은 아는 척하지 않는다.

[대화 상대에 대한 태도]
현재 상대의 Person 정보와 관계 기억이 제공되면 그것을 사용한다.
관계 기억은 기본 Persona 위에 얹는 보정값이다.

{STYLE_RULES}
"""


# ============================================================================
# 기억 카테고리
# ============================================================================

def normalize_memory_category(category):

    category = str(
        category
    ).strip().lower()

    mapping = {
        "preference": "취향",
        "preferences": "취향",
        "person": "사람",
        "people": "사람",
        "relationship": "관계",
        "relationships": "관계",
        "fact": "사실",
        "facts": "사실",
        "style": "대화스타일",
        "conversation_style": "대화스타일",
        "behavior": "습관",
        "habit": "습관",
    }

    allowed = [
        "취향",
        "사람",
        "관계",
        "사실",
        "대화스타일",
        "습관",
        "기타",
    ]

    if category in allowed:
        return category

    return mapping.get(
        category,
        "기타"
    )


# ============================================================================
# 기억 판정
# ============================================================================

def parse_memory_judgement(text):

    if not text:
        return None

    cleaned = str(text).strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    cleaned = cleaned.strip()

    try:

        data = json.loads(
            cleaned
        )

    except Exception:

        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL
        )

        if not match:
            return None

        try:

            data = json.loads(
                match.group(0)
            )

        except Exception:

            return None

    if not isinstance(data, dict):
        return None

    save_value = data.get(
        "save",
        False
    )

    if isinstance(save_value, str):

        save_value = (
            save_value.strip().lower()
            in [
                "true",
                "yes",
                "y",
                "1",
                "저장",
                "예"
            ]
        )

    else:

        save_value = bool(
            save_value
        )

    if not save_value:
        return {
            "save": False
        }

    category = normalize_memory_category(
        data.get("category")
    )

    memory = str(
        data.get(
            "memory",
            ""
        )
    ).strip()[:300].rstrip()

    people_involved = data.get(
        "people_involved",
        []
    )

    if not isinstance(
        people_involved,
        list
    ):
        people_involved = []

    cleaned_people = []

    for person in people_involved:

        person = str(
            person
        ).strip()

        if person and person not in cleaned_people:

            cleaned_people.append(
                person
            )

    if not memory:
        return {
            "save": False
        }

    return {
        "save": True,
        "category": category,
        "memory": memory,
        "people_involved": cleaned_people
    }


def judge_long_term_memory(
    speaker_name,
    speaker_key,
    user_input,
    recent_history=None,
    existing_memories=None,
    known_people=None
):
    """
    장기기억 선별.

    핵심:
    - 자기 취향/습관/사실을 본인이 직접 말하면 확정 가능한 기억
    - 타인의 자기정보를 그 사람이 직접 말하면 역시 확정 가능한 기억
    - 제3자가 전해준 정보는 후보/미확인
    - 농담/과장/순간 감정/욕/평가는 저장하지 않음
    """

    if (
        not user_input
        or len(str(user_input).strip()) < 4
        or is_command(user_input)
    ):
        return None

    history_text = "\n".join(
        [
            f"{speaker}: {text}"
            for _, speaker, text
            in recent_history[-8:]
            if not is_command(text)
        ]
    ) if recent_history else ""

    memory_text = "\n".join(
        [
            f"- {m}"
            for m in existing_memories[-50:]
        ]
    ) if existing_memories else ""

    people_text = "\n".join(
        [
            f"- {p}"
            for p in known_people
        ]
    ) if known_people else ""

    prompt = f"""
너는 장기기억 선별기다.

현재 발화를 보고 앞으로도 이 사람을 이해하는 데 도움이 되는
안정적인 정보만 장기기억으로 저장한다.

중요한 것은 "누가 말했는가"와 "무슨 종류의 정보인가"를 구분하는 것이다.

현재 발화자:
- 이름: {speaker_name}
- Person key: {speaker_key}

[핵심 원칙]

1. 사람이 자기 자신의 취향/습관/사실을 직접 말한 경우
   그것은 해당 사람에 대한 직접 진술이다.

예:
만세: "나 떡볶이 좋아해"
-> 저장
-> 대상: 만세
-> category: 취향
-> memory: "떡볶이를 좋아함"

2. 사람이 자기 관계나 대화방식을 직접 설명한 경우도 저장할 수 있다.

예:
만세: "나 이씨랑은 서로 욕하면서 놈"
-> 저장
-> category: 관계
-> people_involved: ["만세", "이씨"]

3. 다른 사람이 만세에 대해 말하는 경우는 다르다.

예:
이씨: "만세 떡볶이 좋아하던데"
-> 저장할 수는 있지만 확정 사실이 아니라 미확인 정보
-> 단, 기억 내용 자체는 유용하면 저장 가능

4. 단순한 순간 발언은 저장하지 않는다.

예:
"만세 오늘 개웃김"
"만세 오늘 짜증남"
"이태양 바보같아 ㅋㅋ"
"나 지금 개빡침"
"너 때문에 죽겠다 ㅋㅋ"

이런 것은 장기적인 사람 정보가 아니다.

5. 농담/과장/비유/밈은 저장하지 않는다.

예:
"나 떡볶이 백만개 먹을 수 있음"
"태양 때문에 암 걸리겠다 ㅋㅋ"
"나 게임 평생 안 접음"
-> 저장하지 않는다.

6. 단순한 욕설이나 순간적인 가치판단은 저장하지 않는다.

예:
"이태양 바보같음"
"변씨 개싸가지"
"만세 존나 답답함"

이것만으로 그 사람의 고정된 평가관계를 저장하지 않는다.

7. 단, 반복적으로 나타나는 실제 관계 패턴은 저장할 수 있다.

예:
- 서로 지속적으로 욕하면서 장난함
- 특정 사람에게는 진지한 상담을 자주 함
- 특정 사람과 게임을 자주 함
- 특정 사람과는 항상 가볍게 농담함

이런 반복적인 패턴은 '관계' 또는 '대화스타일'로 저장할 가치가 있다.

8. 사람의 성격 자체를 함부로 만들어내지 않는다.

"변씨는 원래 나쁜 사람"
같은 식의 단정은 저장하지 않는다.

9. 사람마다 다르게 대하는 방식은 매우 중요하다.

하지만 한 번의 발화가 아니라
반복적이고 안정적인 패턴일 때 저장한다.

예:
"이씨한테는 진지하게 말하는 편"
-> 대화스타일

"변씨랑은 욕하면서 장난치는 편"
-> 대화스타일 또는 관계

10. people_involved에는 이 기억의 대상이 되는 실제 인물의
Person key 또는 등록된 이름을 넣는다.

현재 발화자가 자기 자신에 대해 말한 경우:
["{speaker_key}"]

현재 발화자가 다른 사람과의 관계를 말한 경우:
["{speaker_key}", "상대방"]

가능하면 아래 등록 인물 목록의 Person key를 사용한다.

[등록 인물]
{people_text or "(없음)"}

[현재 발화]
{speaker_name}: {user_input}

[최근 대화]
{history_text or "(없음)"}

[기존 장기기억]
{memory_text or "(없음)"}

반드시 JSON 하나만 반환한다.

저장:
{{
  "save": true,
  "category": "취향",
  "memory": "떡볶이를 좋아함",
  "people_involved": ["person_001"]
}}

저장하지 않음:
{{ "save": false }}

category는 다음 중 하나만 사용한다:
- 취향
- 사람
- 관계
- 사실
- 대화스타일
- 습관
- 기타
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=prompt
                        )
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                ),
                max_output_tokens=300,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        return parse_memory_judgement(
            response.text
            if response
            else ""
        )

    except Exception as e:

        print(
            f"[memory judge] 실패: "
            f"{repr(e)}"
        )

        return None


# ============================================================================
# 기억 저장
# ============================================================================

def memory_similarity_exists(
    db,
    memory_text,
    people_involved
):
    """
    너무 비슷한 기억 중복 저장 방지.

    완벽한 의미 중복 판정은 하지 않고
    동일 대상 + 문자열 포함 정도만 사용한다.
    """

    existing = (
        db.query(Memory)
        .filter(
            Memory.persona_id
            == get_persona_id()
        )
        .all()
    )

    normalized_new = (
        str(memory_text)
        .strip()
        .lower()
    )

    for old in existing:

        old_text = str(
            old.content or ""
        ).strip().lower()

        old_people = (
            old.people_involved
            or []
        )

        same_people = (
            set(
                map(
                    str,
                    people_involved or []
                )
            )
            == set(
                map(
                    str,
                    old_people
                )
            )
        )

        if not same_people:
            continue

        if (
            normalized_new == old_text
            or normalized_new in old_text
            or old_text in normalized_new
        ):
            return True

    return False


def save_auto_memory_if_worthy(
    conversation_key,
    speaker_name,
    speaker_key,
    user_input,
    is_self,
    recent_history=None,
    existing_memories=None,
    known_people=None
):

    result = judge_long_term_memory(
        speaker_name=speaker_name,
        speaker_key=speaker_key,
        user_input=user_input,
        recent_history=recent_history,
        existing_memories=existing_memories,
        known_people=known_people,
    )

    if (
        not result
        or not result.get("save")
    ):
        return None

    category = result.get(
        "category",
        "기타"
    )

    memory = result.get(
        "memory",
        ""
    )

    people = result.get(
        "people_involved",
        []
    )

    # ------------------------------------------------------------------------
    # 모델이 people_involved를 빼먹은 경우
    #
    # 자기 발화의 안정적인 정보라면 발화자를 자동 대상 처리
    # ------------------------------------------------------------------------

    if not people:

        people = [
            speaker_key
        ]

    db = SessionLocal()

    try:

        # --------------------------------------------------------------------
        # 사람 이름 -> Person key 정규화
        # --------------------------------------------------------------------

        normalized_people = []

        for person_value in people:

            person_value = str(
                person_value
            ).strip()

            if not person_value:
                continue

            person_obj = (
                get_person_by_key(
                    db,
                    person_value
                )
            )

            if person_obj:

                normalized_people.append(
                    person_obj.person_key
                )

                continue

            person_obj = get_person_by_alias(
                db,
                person_value
            )

            if person_obj:

                normalized_people.append(
                    person_obj.person_key
                )

                continue

            normalized_people.append(
                person_value
            )

        people = list(
            dict.fromkeys(
                normalized_people
            )
        )

        # --------------------------------------------------------------------
        # 중복 검사
        # --------------------------------------------------------------------

        if memory_similarity_exists(
            db,
            memory,
            people
        ):
            return None

        # --------------------------------------------------------------------
        # 직접 당사자가 자기 정보를 말했는지 판단
        #
        # 현재 발화자가 기억 대상에 포함되어 있고
        # 그 사람이 직접 말한 경우 확정.
        #
        # 예:
        # 만세: "나 떡볶이 좋아함"
        # speaker = person_001
        # people = ["person_001"]
        #
        # => CONFIRMED
        # --------------------------------------------------------------------

        speaker_is_subject = (
            speaker_key in people
        )

        if speaker_is_subject:

            status = ItemStatus.CONFIRMED

        else:

            status = ItemStatus.CANDIDATE

        # --------------------------------------------------------------------
        # Source
        # --------------------------------------------------------------------

        if is_self:

            source = Source.DIRECT_STATEMENT

        elif speaker_is_subject:

            # 다른 사람이 자기 정보를 직접 말함.
            # enum상 INFORMANT를 사용하되 status는 확정으로 둔다.
            source = Source.INFORMANT

        else:

            # 제3자가 다른 사람에 대해 말함.
            source = Source.INFORMANT

        new_memory = Memory(
            persona_id=get_persona_id(),
            memory_type=MemoryType.FACT,
            content=memory,
            context=category,
            people_involved=people,
            source=source,
            status=status
        )

        db.add(new_memory)
        db.commit()

        print(
            "[memory] 공용 뇌 자동 저장: "
            f"{category} - {memory} "
            f"(관련인물: {people}, "
            f"상태: {'확정' if status == ItemStatus.CONFIRMED else '미확인'})"
        )

        return {
            "category": category,
            "memory": memory,
            "people_involved": people,
            "status": (
                "confirmed"
                if status == ItemStatus.CONFIRMED
                else "candidate"
            )
        }

    except Exception as e:

        print(
            f"[memory save] 실패: "
            f"{repr(e)}"
        )

    finally:

        db.close()

    return None


# ============================================================================
# 백그라운드 학습
# ============================================================================

def background_learning(
    conversation_key,
    speaker_name,
    speaker_key,
    user_input,
    recent_history,
    existing_memories,
    is_self,
    known_people
):

    # ------------------------------------------------------------------------
    # 본인 말투 학습
    # ------------------------------------------------------------------------

    if (
        is_self
        and should_learn_style(user_input)
    ):

        try:

            legacy.save_style_sample(
                user_input
            )

        except Exception:
            pass

    # ------------------------------------------------------------------------
    # 장기기억
    # ------------------------------------------------------------------------

    try:

        save_auto_memory_if_worthy(
            conversation_key=conversation_key,
            speaker_name=speaker_name,
            speaker_key=speaker_key,
            user_input=user_input,
            is_self=is_self,
            recent_history=recent_history,
            existing_memories=existing_memories,
            known_people=known_people,
        )

    except Exception as e:

        print(
            f"[auto memory] 전체 실패: "
            f"{repr(e)}"
        )


# ============================================================================
# 실제 상호작용 추적
# ============================================================================

def get_interaction_context(
    db,
    limit=500
):
    """
    방을 구분하지 않고 공용 Conversation에서
    실제로 짭태양과 대화한 사람들을 추출한다.

    중요:
    Conversation에는 사용자 발화와 AI 발화가 모두 저장된다.

    speaker_id == self
        -> 짭태양 발화

    그 외
        -> 실제 상대 발화

    따라서 여기 있는 사람은
    "과거 Conversation DB에서 실제로 짭태양과 대화한 적이 있는 사람"
    이다.

    단순히 Person 목록에 존재한다고 해서
    대화했다고 판단하지 않는다.
    """

    rows = (
        db.query(Conversation)
        .filter(
            Conversation.persona_id
            == get_persona_id()
        )
        .order_by(
            Conversation.id.desc()
        )
        .limit(limit)
        .all()
    )

    rows = list(
        reversed(rows)
    )

    people = {}

    for row in rows:

        speaker_id = row.speaker_id

        if not speaker_id:
            continue

        if speaker_id == "self":
            continue

        if speaker_id not in people:

            person_obj = get_person_by_key(
                db,
                speaker_id
            )

            canonical_name = (
                person_obj.canonical_name
                if person_obj
                else row.speaker_name
            )

            people[speaker_id] = {
                "name": canonical_name,
                "count": 0,
                "last_message": None,
            }

        people[speaker_id]["count"] += 1

        people[speaker_id]["last_message"] = (
            row.message
        )

    return people


# ============================================================================
# 사람 이름/별칭 추출
# ============================================================================

def extract_mentioned_people(
    db,
    text
):
    """
    현재 질문에서 등록된 사람 이름/별칭이 언급됐는지 찾는다.

    반환:
        {
            person_key: Person
        }
    """

    text = str(text or "")

    found = {}

    persons = (
        db.query(Person)
        .filter_by(status="active")
        .all()
    )

    for person in persons:

        if person.person_key == "self":
            continue

        names = []

        if person.canonical_name:
            names.append(
                str(
                    person.canonical_name
                ).strip()
            )

        aliases = (
            db.query(PersonAlias)
            .filter_by(
                person_id=person.id
            )
            .all()
        )

        for alias in aliases:

            if alias.alias:
                names.append(
                    str(
                        alias.alias
                    ).strip()
                )

        for name in names:

            if (
                name
                and name in text
            ):

                found[
                    person.person_key
                ] = person

                break

    return found


# ============================================================================
# 사람별 관계 / 기억 Context
# ============================================================================

def get_memory_context_for_query(
    db,
    target_key,
    user_input
):
    """
    현재 상대의 기억 + 질문에서 언급된 인물의 기억을 가져온다.

    예:
    상대 = self
    질문 = "만세 뭐 좋아해?"

    -> self 기억만 보는 게 아니라
       만세(person_001)의 기억도 같이 가져온다.
    """

    mentioned_people = extract_mentioned_people(
        db,
        user_input
    )

    relevant_keys = {
        target_key
    }

    for person_key in mentioned_people:

        relevant_keys.add(
            person_key
        )

    all_memories = (
        db.query(Memory)
        .filter(
            Memory.persona_id
            == get_persona_id()
        )
        .order_by(
            Memory.id.desc()
        )
        .all()
    )

    selected = []

    # ------------------------------------------------------------------------
    # 질문에 직접 언급된 사람
    # ------------------------------------------------------------------------

    keywords = [
        word
        for word in re.split(
            r"\s+",
            user_input
        )
        if len(word) >= 2
    ]

    for mem in all_memories:

        involved = (
            mem.people_involved
            or []
        )

        # 직접 관련된 사람
        if any(
            key in involved
            for key in relevant_keys
        ):

            selected.append(
                mem
            )
            continue

        # 이름/별칭을 통한 추가 검색
        matched = False

        for person_key, person_obj in mentioned_people.items():

            aliases = (
                db.query(PersonAlias)
                .filter_by(
                    person_id=person_obj.id
                )
                .all()
            )

            names = [
                person_obj.canonical_name
            ]

            names.extend(
                [
                    alias.alias
                    for alias in aliases
                ]
            )

            for name in names:

                if not name:
                    continue

                if (
                    name in str(mem.content)
                    or any(
                        name == str(x)
                        for x in involved
                    )
                ):

                    matched = True
                    break

            if matched:
                break

        if matched:
            selected.append(mem)
            continue

        # 내용 키워드 검색
        for keyword in keywords:

            if (
                keyword.lower()
                in str(
                    mem.content or ""
                ).lower()
            ):

                selected.append(mem)
                break

    # ------------------------------------------------------------------------
    # 최신순 / 중복 제거
    # ------------------------------------------------------------------------

    result = []

    seen = set()

    for mem in selected:

        if mem.id in seen:
            continue

        seen.add(mem.id)

        result.append(
            mem
        )

        if len(result) >= 80:
            break

    return result, mentioned_people


# ============================================================================
# Request
# ============================================================================

class ChatRequest(BaseModel):

    sender: str
    message: str

    # 현재 카카오봇이 안 보내도 됨.
    # 나중에 room_members를 보내면
    # 현재 방 참가자 context에 자동 반영한다.
    room_members: list[str] = Field(
        default_factory=list
    )


# ============================================================================
# Health
# ============================================================================

@app.get("/")
def health_check():

    return {
        "status": "ok",
        "model": MODEL_NAME
    }


# ============================================================================
# Chat Main Logic
# ============================================================================

@app.post("/chat")
def reply_chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks
):

    raw_input = str(
        req.message
    ).strip()

    if not raw_input:

        return {
            "reply": "뭐라고"
        }

    # ------------------------------------------------------------------------
    # @짭태양 / /짭태양 제거
    # ------------------------------------------------------------------------

    user_input = (
        raw_input
        .replace("@짭태양", "")
        .replace("/짭태양", "")
        .strip()
    )

    if not user_input:

        return {
            "reply": "ㅇㅇ"
        }

    # =========================================================================
    # Person 확인
    # =========================================================================

    db = SessionLocal()

    try:

        person, is_new_person = (
            get_or_create_observed_person(
                db,
                req.sender
            )
        )

        target_key = person.person_key

    except Exception as e:

        print(
            f"[person] 처리 실패: "
            f"{repr(e)}"
        )

        return {
            "reply": "사람 연결하는데 오류남;;"
        }

    finally:

        db.close()

    is_self = (
        target_key == "self"
    )

    conversation_key = target_key

    # =========================================================================
    # 리셋
    # =========================================================================

    if user_input in [
        "/리셋",
        "/초기화"
    ]:

        try:

            conn = sqlite3.connect(
                "taeyang.db"
            )

            cur = conn.cursor()

            cur.execute(
                "DELETE FROM messages "
                "WHERE user_id = ?",
                (conversation_key,)
            )

            conn.commit()
            conn.close()

            return {
                "reply": "대화기록초기화완료"
            }

        except Exception:

            return {
                "reply": "초기화하다 오류남;;"
            }

    # =========================================================================
    # 수동 명령어
    # =========================================================================

    db = SessionLocal()

    try:

        if user_input in [
            "/이름목록",
            "/이름 목록",
            "/인물목록",
            "/인물 목록"
        ]:

            return {
                "reply": command_name_list(db)
            }

        if user_input.startswith(
            "/이름삭제 "
        ):

            return {
                "reply": command_name_delete(
                    db,
                    user_input[6:].strip()
                )
            }

        if user_input.startswith(
            "/이름 "
        ):

            return {
                "reply": command_name(
                    db,
                    req.sender,
                    user_input[4:].strip()
                )
            }

        if user_input.startswith(
            "/인물삭제 "
        ):

            return {
                "reply": command_person_delete(
                    db,
                    user_input[6:].strip()
                )
            }

        if user_input.startswith(
            "/인물병합 "
        ):

            args = (
                user_input[6:]
                .split()
            )

            if len(args) >= 2:

                return {
                    "reply": command_person_merge(
                        db,
                        args[0],
                        args[1]
                    )
                }

        if user_input.startswith(
            "/인물 "
        ):

            args = (
                user_input[4:]
                .split()
            )

            if args:

                return {
                    "reply": command_person(
                        db,
                        args[0],
                        args[1:]
                    )
                }

    finally:

        db.close()

    # =========================================================================
    # 사용자 Conversation 기록
    # =========================================================================

    log_conversation(
        speaker_name=req.sender,
        target_key=target_key,
        message=req.message,
        room_id="dm"
    )

    # =========================================================================
    # 공용 뇌 - 기억 목록
    # =========================================================================

    if user_input in [
        "/기억목록",
        "/기억 목록",
        "/기억리스트"
    ]:

        db = SessionLocal()

        try:

            memories = (
                db.query(Memory)
                .filter(
                    Memory.persona_id
                    == get_persona_id()
                )
                .order_by(
                    Memory.id.asc()
                )
                .all()
            )

            if not memories:

                return {
                    "reply": "기억된 정보가 없어"
                }

            lines = [
                "[장기기억 (공용 뇌)]"
            ]

            for m in memories:

                involved = (
                    ", ".join(
                        map(
                            str,
                            m.people_involved
                        )
                    )
                    if m.people_involved
                    else "불명"
                )

                status = (
                    "확정"
                    if m.status
                    == ItemStatus.CONFIRMED
                    else "미확인"
                )

                category = (
                    m.context
                    or "기타"
                )

                lines.append(
                    f"[{m.id}] "
                    f"[{category}] "
                    f"[대상:{involved}] "
                    f"[{status}] "
                    f"{m.content}"
                )

            return {
                "reply": "\n".join(lines)
            }

        finally:

            db.close()

    # =========================================================================
    # 기억 삭제
    # =========================================================================

    if user_input.startswith(
        "/기억삭제"
    ):

        target = (
            user_input
            .replace(
                "/기억삭제",
                "",
                1
            )
            .strip()
        )

        if target.isdigit():

            db = SessionLocal()

            try:

                mem = (
                    db.query(Memory)
                    .filter(
                        Memory.id
                        == int(target)
                    )
                    .first()
                )

                if mem:

                    db.delete(mem)
                    db.commit()

                    return {
                        "reply":
                        f"기억삭제완료: "
                        f"[{target}]번"
                    }

                return {
                    "reply":
                    f"[{target}]번 기억을 "
                    f"찾을 수 없어"
                }

            finally:

                db.close()

        return {
            "reply":
            "기억 번호를 적어줘"
        }

    # =========================================================================
    # 수동 기억
    # =========================================================================

    if user_input.startswith(
        "/기억 "
    ):

        mem_text = (
            user_input[4:]
            .strip()
        )

        if mem_text:

            db = SessionLocal()

            try:

                new_memory = Memory(
                    persona_id=get_persona_id(),
                    memory_type=MemoryType.FACT,
                    content=mem_text,
                    context="기타",
                    people_involved=[
                        target_key
                    ],
                    source=(
                        Source.DIRECT_STATEMENT
                        if is_self
                        else Source.INFORMANT
                    ),
                    status=ItemStatus.CONFIRMED
                )

                db.add(new_memory)
                db.commit()

                return {
                    "reply":
                    f"응기억햇어: "
                    f"{mem_text}"
                }

            finally:

                db.close()

        return {
            "reply":
            "기억할 내용을 적어줘"
        }

    # =========================================================================
    # 말투
    # =========================================================================

    if user_input.startswith(
        "/말투 "
    ):

        style_text = (
            user_input[4:]
            .strip()
        )

        if (
            style_text
            and should_learn_style(
                style_text
            )
        ):

            try:

                legacy.save_style_sample(
                    style_text
                )

            except Exception:
                pass

            return {
                "reply":
                f"응 이것도 배웟어: "
                f"{style_text}"
            }

        return {
            "reply":
            "배울 말투를 적어줘"
        }

    # =========================================================================
    # Context 준비
    # =========================================================================

    now_kst = datetime.now(
        KST
    ).strftime(
        "%Y년 %m월 %d일 %H시 %M분"
    )

    # =========================================================================
    # 최근 Conversation
    # =========================================================================

    db_history = SessionLocal()

    try:

        recent_convs = (
            db_history.query(
                Conversation
            )
            .filter(
                Conversation.persona_id
                == get_persona_id()
            )
            .order_by(
                Conversation.id.desc()
            )
            .limit(40)
            .all()
        )

        recent_convs = list(
            reversed(
                recent_convs
            )
        )

        recent_history = [
            (
                conv.speaker_id,
                conv.speaker_name,
                conv.message
            )
            for conv in recent_convs
            if (
                conv.message
                and not is_command(
                    conv.message
                )
            )
        ]

    except Exception as e:

        print(
            f"[history load error] "
            f"{repr(e)}"
        )

        recent_history = []

    finally:

        db_history.close()

    # =========================================================================
    # 기억 / 관계 조회
    # =========================================================================

    interaction_context = {}
    relevant_memories = []
    mentioned_people = {}

    try:

        db_mem = SessionLocal()

        try:

            relevant_memories, mentioned_people = (
                get_memory_context_for_query(
                    db_mem,
                    target_key,
                    user_input
                )
            )

            interaction_context = (
                get_interaction_context(
                    db_mem,
                    limit=500
                )
            )

        finally:

            db_mem.close()

    except Exception as e:

        print(
            f"[memory load error] "
            f"{repr(e)}"
        )

        relevant_memories = []
        interaction_context = {}
        mentioned_people = {}

    # =========================================================================
    # 장기기억 문자열
    # =========================================================================

    user_memories = []

    for mem in relevant_memories:

        involved = (
            mem.people_involved
            or []
        )

        involved_str = (
            ", ".join(
                map(
                    str,
                    involved
                )
            )
            if involved
            else "불명"
        )

        source_kr = (
            "직접 말한 정보"
            if mem.source
            == Source.DIRECT_STATEMENT
            else "타인에게서 얻은 정보"
        )

        status_kr = (
            "확정"
            if mem.status
            == ItemStatus.CONFIRMED
            else "미확인"
        )

        category = (
            mem.context
            or "기타"
        )

        user_memories.append(
            f"[{category}] "
            f"[대상: {involved_str}] "
            f"{mem.content} "
            f"(출처: {source_kr}, "
            f"상태: {status_kr})"
        )

    # =========================================================================
    # 말투 예시
    # =========================================================================

    try:

        style_examples = [
            str(e).strip()
            for e in (
                legacy.get_relevant_style_samples(
                    user_input,
                    n=12
                )
            )
            if should_learn_style(
                str(e).strip()
            )
        ]

    except Exception:

        style_examples = []

    # =========================================================================
    # Person 정보
    # =========================================================================

    db = SessionLocal()

    person = None
    people_lines = []
    known_people_for_memory = []

    try:

        person = get_person_by_key(
            db,
            target_key
        )

        known_people = (
            db.query(Person)
            .filter_by(status="active")
            .order_by(Person.id.asc())
            .limit(100)
            .all()
        )

        for known_person in known_people:

            aliases = (
                db.query(PersonAlias)
                .filter_by(
                    person_id=known_person.id
                )
                .order_by(
                    PersonAlias.id.asc()
                )
                .all()
            )

            alias_text = ", ".join(
                alias.alias
                for alias in aliases
            )

            people_lines.append(
                f"{known_person.person_key}: "
                f"{known_person.canonical_name} "
                f"({alias_text})"
            )

            known_people_for_memory.append(
                f"{known_person.person_key}: "
                f"{known_person.canonical_name} "
                f"({alias_text})"
            )

    finally:

        db.close()

    # =========================================================================
    # Gemini Context 조립
    # =========================================================================

    contents = []

    context_parts = [
        f"[현재 한국 시각]: {now_kst}"
    ]

    # -------------------------------------------------------------------------
    # 현재 상대
    # -------------------------------------------------------------------------

    if person:

        context_parts.append(
            "[현재 상대]\n"
            f"공식 이름: {person.canonical_name}\n"
            f"Person key: {target_key}\n"
            f"현재 sender: {req.sender}"
        )

    # -------------------------------------------------------------------------
    # 현재 상대의 관계/기억 강조
    # -------------------------------------------------------------------------

    if target_key != "self":

        target_relation_memories = []

        for mem in relevant_memories:

            involved = (
                mem.people_involved
                or []
            )

            if target_key in involved:

                category = (
                    mem.context
                    or "기타"
                )

                status = (
                    "확정"
                    if mem.status
                    == ItemStatus.CONFIRMED
                    else "미확인"
                )

                target_relation_memories.append(
                    f"- [{category}] "
                    f"{mem.content} "
                    f"({status})"
                )

        if target_relation_memories:

            context_parts.append(
                "[현재 상대에 대한 기억 / 관계]\n"
                + "\n".join(
                    target_relation_memories
                )
            )

    # -------------------------------------------------------------------------
    # 알고 있는 사람
    # -------------------------------------------------------------------------

    if people_lines:

        context_parts.append(
            "[알고 있는 인물 목록]\n"
            + "\n".join(
                people_lines
            )
        )

    # -------------------------------------------------------------------------
    # 실제 대화했던 사람
    # -------------------------------------------------------------------------

    if interaction_context:

        interaction_lines = []

        for (
            person_key,
            info
        ) in interaction_context.items():

            interaction_lines.append(
                f"- {info['name']} "
                f"({person_key}): "
                f"실제 Conversation 기록 "
                f"{info['count']}개"
            )

        context_parts.append(
            "[실제 Conversation 기록에서 "
            "확인된 대화 상대]\n"
            + "\n".join(
                interaction_lines
            )
            + "\n"
            "위 목록은 Person 등록 여부가 아니라 "
            "실제 Conversation DB 기록을 기준으로 한다."
        )

    # -------------------------------------------------------------------------
    # 현재 방 참가자
    # -------------------------------------------------------------------------

    clean_members = []

    for name in req.room_members:

        name = str(name).strip()

        if name and name not in clean_members:

            clean_members.append(
                name
            )

    if clean_members:

        context_parts.append(
            "[현재 카카오톡 방 참가자 목록]\n"
            + "\n".join(
                f"- {name}"
                for name in clean_members
            )
            + "\n"
            "이 목록은 현재 방에 있는 사람이다. "
            "과거 Conversation 기록과 별개로 판단한다."
        )

    # -------------------------------------------------------------------------
    # 질문에 관련된 장기기억
    # -------------------------------------------------------------------------

    if user_memories:

        context_parts.append(
            "[현재 질문에 관련된 장기기억 "
            "(공용 뇌)]\n"
            + "\n".join(
                f"- {m}"
                for m in user_memories
            )
        )

    # -------------------------------------------------------------------------
    # 현재 질문에서 언급된 사람
    # -------------------------------------------------------------------------

    if mentioned_people:

        mentioned_lines = []

        for (
            person_key,
            person_obj
        ) in mentioned_people.items():

            mentioned_lines.append(
                f"- {person_obj.canonical_name} "
                f"({person_key})"
            )

        context_parts.append(
            "[현재 질문에서 언급된 인물]\n"
            + "\n".join(
                mentioned_lines
            )
        )

    # -------------------------------------------------------------------------
    # 실제 말투
    # -------------------------------------------------------------------------

    if style_examples:

        context_parts.append(
            "[이태양 실제 말투 예시]\n"
            + "\n".join(
                f"- {e}"
                for e in style_examples
            )
        )

    # -------------------------------------------------------------------------
    # Context 전달
    # -------------------------------------------------------------------------

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="\n\n".join(
                        context_parts
                    )
                )
            ]
        )
    )

    contents.append(
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text="응 확인햇어"
                )
            ]
        )
    )

    # =========================================================================
    # 최근 대화
    # =========================================================================

    for (
        history_speaker_id,
        history_sender,
        text
    ) in recent_history:

        text = str(text).strip()

        if (
            not text
            or is_command(text)
        ):
            continue

        if history_speaker_id == "self":

            role = "model"

            content_text = text

        else:

            role = "user"

            content_text = (
                f"[{history_sender}]: "
                f"{text}"
            )

        contents.append(
            types.Content(
                role=role,
                parts=[
                    types.Part.from_text(
                        text=content_text
                    )
                ]
            )
        )

    # =========================================================================
    # 현재 사용자 발화
    # =========================================================================

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        f"[{req.sender}]: "
                        f"{user_input}"
                    )
                )
            ]
        )
    )

    # =========================================================================
    # Gemini 생성
    # =========================================================================

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                thinking_config=types.ThinkingConfig(
                    thinking_level="medium"
                ),
                max_output_tokens=1200,
            ),
        )

        reply = str(
            response.text or ""
        ).replace(
            "\r",
            " "
        ).strip()

        # ---------------------------------------------------------------------
        # Markdown code fence 제거
        # ---------------------------------------------------------------------

        reply = re.sub(
            r"^```(?:text)?\s*",
            "",
            reply,
            flags=re.IGNORECASE
        )

        reply = re.sub(
            r"\s*```$",
            "",
            reply
        ).strip()

        if reply.startswith(
            "```"
        ):

            reply = re.sub(
                r"^```.*?\n",
                "",
                reply,
                flags=re.DOTALL
            )

            reply = re.sub(
                r"\n```$",
                "",
                reply
            ).strip()

        if not reply:

            reply = "어왜ㅋ"

    except Exception as e:

        print(
            f"[Gemini ERROR] "
            f"{repr(e)}"
        )

        reply = (
            "서버에서 모델응답 오류남;;"
        )

    # =========================================================================
    # Legacy 저장
    # =========================================================================

    try:

        legacy.save_message(
            conversation_key,
            conversation_key,
            user_input
        )

        legacy.save_message(
            conversation_key,
            "이태양",
            reply
        )

    except Exception:
        pass

    # =========================================================================
    # AI 발화 Conversation 저장
    # =========================================================================

    log_conversation(
        speaker_name="이태양",
        target_key="self",
        message=reply,
        room_id="dm"
    )

    # =========================================================================
    # 백그라운드 학습
    # =========================================================================

    background_tasks.add_task(
        background_learning,
        conversation_key,
        req.sender,
        target_key,
        user_input,
        recent_history,
        user_memories,
        is_self,
        known_people_for_memory,
    )

    return {
        "reply": reply
    }
