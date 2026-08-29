
"""
장산범 Persona Engine v4 - main.py

핵심
- /이름 이태양 -> 현재 sender를 self로 연결
- sender -> self Identity 유지
- Identity / Person / Alias 분리
- 기존 legacy 대화/말투 유지
- 장기기억은 신규 DB(Memory)로 통합
- 본인(self)과 타인(만세, 챠 등)의 공용 뇌 구성
- 찐태양(원본)과 짭태양(복제본)의 특수 관계성 프롬프트 적용
- 장기기억 판정은 백그라운드에서 처리
- 단기 기억은 최근 대화 흐름을 화자 ID 기준으로 구성
- Conversation에서 실제로 누구와 대화했는지 별도로 추적
- 방별로 기억을 분리하지 않음
- room_members가 전달되는 경우 현재 방 참가자 목록을 context에 추가
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
                raise RuntimeError(
                    "태양 Persona가 없습니다."
                )

            _PERSONA_ID_CACHE["id"] = persona.id

        finally:
            db.close()

    return _PERSONA_ID_CACHE["id"]


# ---------------------------------------------------------------------------
# Person / Identity
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 현재 sender -> Person
# ---------------------------------------------------------------------------

def get_or_create_observed_person(db, display_name):
    display_name = str(display_name).strip()

    if not display_name:
        raise ValueError(
            "display_name이 비어있음"
        )

    # -------------------------------------------------------
    # 1. Identity
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # 2. Alias
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # 3. 처음 보는 사람
    # -------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 이름 명령
# ---------------------------------------------------------------------------

def command_name(db, sender, new_name):
    sender = str(sender).strip()
    new_name = str(new_name).strip()

    if not sender:
        return "sender가 없어"

    if not new_name:
        return "이름을 적어줘"

    # -------------------------------------------------------
    # self Person 확보
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # 현재 sender Identity 확인
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # sender alias 제거
    # -------------------------------------------------------

    sender_aliases = (
        db.query(PersonAlias)
        .filter_by(alias=sender)
        .all()
    )

    for alias in sender_aliases:
        if alias.person_id != self_person.id:
            db.delete(alias)

    db.flush()

    # -------------------------------------------------------
    # Identity -> self
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # sender alias
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # canonical name alias
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # 기존 관측 Person 비활성화
    # -------------------------------------------------------

    if (
        old_person
        and old_person.id != self_person.id
        and old_person.person_key not in ["self", "cha"]
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


# ---------------------------------------------------------------------------
# 명령 - 조회 / 삭제 / 병합
# ---------------------------------------------------------------------------

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

    # -------------------------------------------------------
    # Alias 병합
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # Identity 병합
    # -------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

def log_conversation(
    speaker_name,
    target_key,
    message,
    room_id="dm"
):
    """
    실제 Conversation 기록.

    target_key에는 반드시 Person key를 넣는다.
    본인 발화는 self.
    타인 발화는 person_XXX / cha 등.
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


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

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
        and text.startswith(COMMAND_PREFIXES)
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
- 본인이 알려주는 내용은 높은 신뢰도로 취급한다.
- 다만 농담, 과장, 가설, 추측처럼 보이는 내용을 무조건 사실로 확정하지 않는다.

[기억의 활용]
- 대화 중에 언급된 사람에 대한 기존 기억이 제공되면 자연스럽게 대화에 녹여라.
- 실제 Conversation 기록에 특정 사람이 존재하면 그 사람과 실제로 대화한 사실을 기억한다.
- 과거에 대화한 사람과 현재 카카오톡 방에 있는 사람은 서로 다른 정보임을 구분한다.
- 현재 카카오톡 방 참가자 목록이 제공되면 그것을 현재 방에 있는 사람으로 취급한다.
- 현재 방 참가자 목록이 제공되지 않았다면, 과거 Conversation 기록만으로 그 사람이 지금 이 방에 있다고 단정하지 않는다.
- 모르는 내용은 아는 척하지 말고 원본에게 되물어라.

{STYLE_RULES}
"""


SYSTEM_INSTRUCTION_FOR_CHA = f"""
너는 21살 대학생 이태양의 AI 페르소나다.
상대는 친한 게임 친구 챠다.
챠와는 친밀도가 높은 친구처럼 자연스럽게 카카오톡으로 대화한다.
기억에 없는 사실은 아는 척하지 않는다.

[대화 기록]
실제 Conversation 기록에 과거 대화가 있으면 그 사실을 자연스럽게 기억한다.
현재 카카오톡 방 참가자 목록이 제공되면 현재 방의 사람으로 취급한다.

{STYLE_RULES}
"""


SYSTEM_INSTRUCTION_FOR_UNKNOWN = f"""
너는 21살 대학생 이태양의 AI 페르소나다.
상대가 누구인지 확실하지 않으면 특정 친구로 단정하지 않는다.
상대에 대한 기억이 없으면 아는 척하지 않는다.

[대화 기록]
실제 Conversation 기록에 과거 대화가 있으면 그 사실을 참고한다.
현재 카카오톡 방 참가자 목록이 제공되면 현재 방의 사람으로 취급한다.
참가자 목록이 없으면 과거 대화 기록만으로 현재 방의 참가자를 단정하지 않는다.

{STYLE_RULES}
"""


# ---------------------------------------------------------------------------
# 장기기억 판정
# ---------------------------------------------------------------------------

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
        "fact": "사실",
        "facts": "사실",
    }

    if category in [
        "취향",
        "사람",
        "관계",
        "사실",
        "기타"
    ]:
        return category

    return mapping.get(
        category,
        "기타"
    )


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
        data = json.loads(cleaned)

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

    if not memory:
        return {
            "save": False
        }

    return {
        "save": True,
        "category": category,
        "memory": memory,
        "people_involved": people_involved
    }


def judge_long_term_memory(
    user_input,
    recent_history=None,
    existing_memories=None
):
    if (
        not user_input
        or len(str(user_input).strip()) < 4
        or is_command(user_input)
    ):
        return None

    history_text = "\n".join(
        [
            f"{speaker_name}: {text}"
            for _, speaker_name, text
            in recent_history[-6:]
            if not is_command(text)
        ]
    ) if recent_history else ""

    memory_text = "\n".join(
        [
            f"- {m}"
            for m in existing_memories[-30:]
        ]
    ) if existing_memories else ""

    prompt = f"""
너는 장기기억 선별기다.

사용자의 현재 발화를 보고 앞으로도 이 사람을 이해하는 데 도움이 될 만한 안정적인 정보만 장기기억으로 저장해라.

[중요: 관련 인물 태깅]
이 기억이 누구와 관련된 정보인지 파악해서 'people_involved' 리스트에 담아라.

가능하면 실제 등록된 이름이나 Person key를 기준으로 판단한다.

예:
본인이 "만세 매운거 환장함"이라고 했다면
-> ["만세"]

예:
만세가 직접 "나 오이 싫어"라고 했다면
-> ["만세"]

예:
본인이 "나는 야구가 좋아"라고 했다면
-> ["self"]

일상적인 단순 질문, 인사, 감탄, 임시 상황, 단순한 대화 내용은 저장하지 않는다.

반드시 JSON 하나만 반환한다.

저장 예시:
{{
  "save": true,
  "category": "취향",
  "memory": "매운 것을 좋아함",
  "people_involved": ["만세"]
}}

저장하지 않을 경우:
{{ "save": false }}

[현재 발화]
{user_input}

[최근 대화]
{history_text or "(없음)"}

[기존 장기기억]
{memory_text or "(없음)"}
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


def save_auto_memory_if_worthy(
    conversation_key,
    user_input,
    is_self,
    recent_history=None,
    existing_memories=None
):
    result = judge_long_term_memory(
        user_input,
        recent_history,
        existing_memories
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

    db = SessionLocal()

    try:

        existing = (
            db.query(Memory)
            .filter(
                Memory.persona_id
                == get_persona_id()
            )
            .all()
        )

        normalized_new = (
            memory.lower()
        )

        for old in existing:

            if (
                normalized_new
                in old.content.lower()
            ):
                return None

        new_memory = Memory(
            persona_id=get_persona_id(),
            memory_type=MemoryType.FACT,
            content=memory,
            context=category,
            people_involved=people,
            source=(
                Source.DIRECT_STATEMENT
                if is_self
                else Source.INFORMANT
            ),
            status=(
                ItemStatus.CONFIRMED
                if is_self
                else ItemStatus.CANDIDATE
            )
        )

        db.add(new_memory)
        db.commit()

        print(
            "[memory] 공용 뇌 자동 저장: "
            f"{category} - {memory} "
            f"(관련인물: {people})"
        )

        return {
            "category": category,
            "memory": memory
        }

    except Exception as e:

        print(
            f"[memory save] 실패: "
            f"{repr(e)}"
        )

    finally:
        db.close()

    return None


# ---------------------------------------------------------------------------
# 백그라운드 학습
# ---------------------------------------------------------------------------

def background_learning(
    conversation_key,
    user_input,
    recent_history,
    existing_memories,
    is_self
):
    # -------------------------------------------------------
    # 본인 말투 학습
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # 장기기억 학습
    # -------------------------------------------------------

    try:

        save_auto_memory_if_worthy(
            conversation_key=conversation_key,
            user_input=user_input,
            is_self=is_self,
            recent_history=recent_history,
            existing_memories=existing_memories,
        )

    except Exception as e:

        print(
            f"[auto memory] 전체 실패: "
            f"{repr(e)}"
        )


# ---------------------------------------------------------------------------
# Conversation 상호작용 분석
# ---------------------------------------------------------------------------

def get_interaction_context(
    db,
    limit=300
):
    """
    방을 구분하지 않고 공용 Conversation 기록에서
    실제로 짭태양과 대화한 사람들을 추출한다.

    목적:
    "만세랑 대화했어?"
    같은 질문에 실제 대화 기록을 근거로 답할 수 있게 한다.
    """

    rows = (
        db.query(Conversation)
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

        # 짭태양 본인 발화는 제외
        if speaker_id == "self":
            continue

        if speaker_id not in people:

            people[speaker_id] = {
                "name": row.speaker_name,
                "count": 0,
                "last_message": None,
            }

        people[speaker_id]["count"] += 1
        people[speaker_id]["last_message"] = (
            row.message
        )

    return people


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    sender: str
    message: str

    # 현재 카카오봇에서 안 보내도 됨.
    # 나중에 room_members를 보내면 자동으로 현재 방 참가자를 context에 넣는다.
    room_members: list[str] = Field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_NAME
    }


# ---------------------------------------------------------------------------
# Chat Main Logic
# ---------------------------------------------------------------------------

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

    # -------------------------------------------------------
    # @짭태양 / /짭태양 제거
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # Person 확인
    # -------------------------------------------------------

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

    # 중요:
    # 이름이 아니라 Person key로 self 판별
    is_self = (
        target_key == "self"
    )

    conversation_key = target_key

    # -------------------------------------------------------
    # 리셋 처리
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # 수동 명령어
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # 사용자 Conversation 기록
    # -------------------------------------------------------

    log_conversation(
        speaker_name=req.sender,
        target_key=target_key,
        message=req.message,
        room_id="dm"
    )

    # -------------------------------------------------------
    # 공용 뇌 명령어
    # -------------------------------------------------------

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
                        m.people_involved
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

                lines.append(
                    f"[{m.id}] "
                    f"[대상:{involved}] "
                    f"[{status}] "
                    f"{m.content}"
                )

            return {
                "reply": "\n".join(lines)
            }

        finally:
            db.close()

    # -------------------------------------------------------
    # 기억 삭제
    # -------------------------------------------------------

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

    # -------------------------------------------------------
    # 수동 기억
    # -------------------------------------------------------

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
                    status=(
                        ItemStatus.CONFIRMED
                        if is_self
                        else ItemStatus.CANDIDATE
                    )
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

    # -------------------------------------------------------
    # 말투
    # -------------------------------------------------------

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

    # ---------------------------------------------------------
    # Context 준비
    # ---------------------------------------------------------

    now_kst = datetime.now(
        KST
    ).strftime(
        "%Y년 %m월 %d일 %H시 %M분"
    )

    # ---------------------------------------------------------
    # 최근 Conversation
    # ---------------------------------------------------------

    db_history = SessionLocal()

    try:

        recent_convs = (
            db_history.query(
                Conversation
            )
            .order_by(
                Conversation.id.desc()
            )
            .limit(30)
            .all()
        )

        recent_convs = list(
            reversed(
                recent_convs
            )
        )

        # speaker_id를 반드시 유지한다.
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

    # ---------------------------------------------------------
    # 장기기억 조회 + 실제 상호작용 조회
    # ---------------------------------------------------------

    interaction_context = {}
    user_memories = []

    try:

        db_mem = SessionLocal()

        try:

            all_memories = (
                db_mem.query(Memory)
                .filter(
                    Memory.persona_id
                    == get_persona_id()
                )
                .all()
            )

            mentioned_keywords = [
                word
                for word in user_input.split()
                if len(word) >= 2
            ]

            for mem in all_memories:

                involved = (
                    mem.people_involved
                    or []
                )

                is_relevant = (
                    target_key
                    in involved
                )

                if not is_relevant:

                    for keyword in mentioned_keywords:

                        if any(
                            keyword
                            in str(person)
                            for person
                            in involved
                        ):

                            is_relevant = True
                            break

                if is_relevant:

                    source_kr = (
                        "본인(너)이 주입함"
                        if mem.source
                        == Source.DIRECT_STATEMENT
                        else "타인과의 대화에서 얻음"
                    )

                    involved_str = (
                        ", ".join(involved)
                        if involved
                        else "불명"
                    )

                    status_kr = (
                        "확정"
                        if mem.status
                        == ItemStatus.CONFIRMED
                        else "미확인"
                    )

                    user_memories.append(
                        f"[대상: {involved_str}] "
                        f"{mem.content} "
                        f"(출처: {source_kr}, "
                        f"상태: {status_kr})"
                    )

            # 실제 Conversation에서 확인되는 사람
            interaction_context = (
                get_interaction_context(
                    db_mem,
                    limit=300
                )
            )

        finally:
            db_mem.close()

    except Exception as e:

        print(
            f"[memory load error] "
            f"{repr(e)}"
        )

        user_memories = []
        interaction_context = {}

    # ---------------------------------------------------------
    # 말투 예시
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 현재 상대에 따른 system instruction
    # ---------------------------------------------------------

    if is_self:

        system_instruction = (
            SYSTEM_INSTRUCTION_FOR_SELF
        )

    elif target_key == "cha":

        system_instruction = (
            SYSTEM_INSTRUCTION_FOR_CHA
        )

    else:

        system_instruction = (
            SYSTEM_INSTRUCTION_FOR_UNKNOWN
        )

    # ---------------------------------------------------------
    # Person 목록
    # ---------------------------------------------------------

    db = SessionLocal()

    person = None
    people_lines = []

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

    finally:
        db.close()

    # ---------------------------------------------------------
    # Gemini Context 조립
    # ---------------------------------------------------------

    contents = []

    context_parts = [
        f"[현재 한국 시각]: {now_kst}"
    ]

    # ---------------------------------------------------------
    # 현재 상대
    # ---------------------------------------------------------

    if person:

        context_parts.append(
            "[현재 상대]\n"
            f"공식 이름: {person.canonical_name}\n"
            f"Person key: {target_key}\n"
            f"현재 sender: {req.sender}"
        )

    # ---------------------------------------------------------
    # 알고 있는 사람
    # ---------------------------------------------------------

    if people_lines:

        context_parts.append(
            "[알고 있는 인물 목록]\n"
            + "\n".join(
                people_lines
            )
        )

    # ---------------------------------------------------------
    # 실제로 대화했던 사람
    # ---------------------------------------------------------

    if interaction_context:

        interaction_lines = []

        for (
            person_key,
            info
        ) in interaction_context.items():

            interaction_lines.append(
                f"- {info['name']} "
                f"({person_key}) : "
                f"실제 대화 기록 "
                f"{info['count']}개"
            )

        context_parts.append(
            "[실제 Conversation 기록에서 "
            "확인된 대화 상대]\n"
            + "\n".join(
                interaction_lines
            )
        )

    # ---------------------------------------------------------
    # 현재 카카오톡 방 참가자
    #
    # 현재 카카오봇이 room_members를 안 보내면
    # 이 부분은 그냥 생략된다.
    # ---------------------------------------------------------

    if req.room_members:

        clean_members = []

        for name in req.room_members:

            name = str(name).strip()

            if name:
                clean_members.append(name)

        if clean_members:

            context_parts.append(
                "[현재 카카오톡 방 참가자 목록]\n"
                + "\n".join(
                    f"- {name}"
                    for name in clean_members
                )
            )

    # ---------------------------------------------------------
    # 장기기억
    # ---------------------------------------------------------

    if user_memories:

        context_parts.append(
            "[대화 관련 장기기억 "
            "(공용 뇌)]\n"
            + "\n".join(
                f"- {m}"
                for m in user_memories
            )
        )

    # ---------------------------------------------------------
    # 실제 말투
    # ---------------------------------------------------------

    if style_examples:

        context_parts.append(
            "[실제 말투 예시]\n"
            + "\n".join(
                f"- {e}"
                for e in style_examples
            )
        )

    # ---------------------------------------------------------
    # Context 전달
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 최근 대화
    # ---------------------------------------------------------

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

        # ---------------------------------------------------
        # 중요:
        # "이태양"이라는 이름이 아니라
        # speaker_id == self인지 확인한다.
        # ---------------------------------------------------

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

    # ---------------------------------------------------------
    # 현재 사용자 발화
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Gemini 생성
    # ---------------------------------------------------------

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
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

        # ---------------------------------------------------
        # Markdown code fence 제거
        # ---------------------------------------------------

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

    # ---------------------------------------------------------
    # Legacy 저장
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # AI 발화도 Conversation에 저장
    #
    # 중요:
    # speaker_id = self
    # 이름 문자열로 판별하지 않는다.
    # ---------------------------------------------------------

    log_conversation(
        speaker_name="이태양",
        target_key="self",
        message=reply,
        room_id="dm"
    )

    # ---------------------------------------------------------
    # 백그라운드 학습
    # ---------------------------------------------------------

    background_tasks.add_task(
        background_learning,
        conversation_key,
        user_input,
        recent_history,
        user_memories,
        is_self,
    )

    return {
        "reply": reply
    }
