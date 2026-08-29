"""
장산범 Persona Engine v6 - main.py

핵심 구조
- 짭태양의 기본 Persona는 하나
- sender display_name과 실제 Person identity를 분리
- self는 하나의 Person
- 카카오톡 프로필명 여러 개 -> 같은 Person(self)
- /이름은 "현재 sender 본인의 이름/정체성 연결"
- /인물은 제3자 인물 등록
- 상대별 차이는 Person + Memory + Relationship으로 보정
- 자기정보 직접 진술은 높은 신뢰도로 확정
- 제3자 발언은 후보
- 농담/과장/순간 감정/단순 욕/단순 평가 제외
- 실제 Conversation으로 대화 상대 추적
- Conversation의 speaker_id=self면 sender_name이 무엇이든 본인 발화
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

from database import SessionLocal, init_db

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


# ============================================================================
# Gemini
# ============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY 환경변수를 설정해주세요."
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)

MODEL_NAME = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
)


# ============================================================================
# 기본 설정
# ============================================================================

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
# Person
# ============================================================================

def get_person_by_key(db, person_key):
    return (
        db.query(Person)
        .filter_by(person_key=person_key)
        .first()
    )


def get_person_by_alias(db, alias):
    alias = str(alias or "").strip()

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
    display_name = str(
        display_name or ""
    ).strip()

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
# Self
# ============================================================================

def get_self_person(db):
    """
    self Person은 시스템 전체에서 딱 하나만 존재한다.
    """

    person = (
        db.query(Person)
        .filter_by(person_key="self")
        .first()
    )

    return person


def ensure_self_person(db, canonical_name=None):

    person = get_self_person(db)

    if not person:

        person = Person(
            person_key="self",
            canonical_name=(
                canonical_name
                or "이태양"
            ),
            person_type="self",
            status="active",
            observed_in_chat=1,
            confirmed=1,
        )

        db.add(person)
        db.commit()
        db.refresh(person)

    else:

        person.person_type = "self"
        person.status = "active"
        person.confirmed = 1

        if canonical_name:
            person.canonical_name = canonical_name

        db.commit()

    return person


# ============================================================================
# 현재 sender -> Person
# ============================================================================

def get_or_create_observed_person(
    db,
    display_name
):
    """
    일반적인 sender resolution.

    중요:
    여기서는 무조건 self로 만들지 않는다.

    실제 self 연결은 /이름 명령을 통해 한다.
    """

    display_name = str(
        display_name or ""
    ).strip()

    if not display_name:
        raise ValueError(
            "display_name이 비어있음"
        )

    # ------------------------------------------------------------------------
    # 1. 카카오톡 Identity
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

        if person.status in [
            "inactive",
            "merged"
        ]:
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

        if person.status in [
            "inactive",
            "merged"
        ]:
            person.status = "active"

        person.observed_in_chat = 1

        db.add(
            Identity(
                person_id=person.id,
                target_key=person.person_key,
                platform="kakaotalk",
                display_name=display_name,
                is_primary=(
                    1
                    if person.person_key == "self"
                    else 0
                ),
            )
        )

        db.commit()

        return person, False

    # ------------------------------------------------------------------------
    # 3. 처음 보는 sender
    # ------------------------------------------------------------------------

    person = Person(
        person_key=make_person_key(db),
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
            confidence=0.5,
        )
    )

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

    return person, True


# ============================================================================
# /이름
# ============================================================================

def command_name(
    db,
    sender,
    new_name
):
    """
    /이름은 현재 sender 본인의 정체성을 연결한다.

    예:

    성시연:
        /이름 이태양

    => 성시연 Identity -> self
    => 성시연 Alias -> self
    => 이태양 Alias -> self

    다른 사람:

    만세:
        /이름 만세

    => 만세 sender -> 만세 Person

    즉 /이름은 모두가 사용할 수 있다.
    """

    sender = str(sender or "").strip()
    new_name = str(new_name or "").strip()

    if not sender:
        return "sender가 없어"

    if not new_name:
        return "이름을 적어줘"

    # ------------------------------------------------------------------------
    # 현재 sender Identity
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
    # 기존 self 확인
    # ------------------------------------------------------------------------

    self_person = get_self_person(db)

    # ------------------------------------------------------------------------
    # 이미 self의 이름/별칭이면 self로 연결
    # ------------------------------------------------------------------------

    is_self_name = False

    if self_person:

        if (
            self_person.canonical_name
            and self_person.canonical_name
            == new_name
        ):
            is_self_name = True

        self_alias = (
            db.query(PersonAlias)
            .filter(
                PersonAlias.person_id
                == self_person.id,
                PersonAlias.alias
                == new_name
            )
            .first()
        )

        if self_alias:
            is_self_name = True

    # ------------------------------------------------------------------------
    # 처음 /이름을 사용하면서 "이태양"을 지정하면 self
    #
    # 기존 코드의 Persona 이름이 "태양"이므로
    # 이태양 / 태양 둘 다 self 이름으로 인정한다.
    # ------------------------------------------------------------------------

    self_names = {
        "이태양",
        "태양",
    }

    env_self_names = os.environ.get(
        "SELF_NAMES",
        ""
    )

    if env_self_names:

        for name in env_self_names.split(","):

            name = name.strip()

            if name:
                self_names.add(name)

    if new_name in self_names:
        is_self_name = True

    # =========================================================================
    # SELF
    # =========================================================================

    if is_self_name:

        self_person = ensure_self_person(
            db,
            canonical_name=new_name
        )

        # --------------------------------------------------------------------
        # sender Identity -> self
        # --------------------------------------------------------------------

        if current_identity:

            current_identity.person_id = (
                self_person.id
            )

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
                    is_primary=1,
                )
            )

        # --------------------------------------------------------------------
        # sender alias -> self
        # --------------------------------------------------------------------

        sender_alias = (
            db.query(PersonAlias)
            .filter_by(
                person_id=self_person.id,
                alias=sender,
            )
            .first()
        )

        if not sender_alias:

            db.add(
                PersonAlias(
                    person_id=self_person.id,
                    alias=sender,
                    source=Source.DIRECT_STATEMENT,
                    confidence=1.0,
                )
            )

        # --------------------------------------------------------------------
        # canonical name -> self
        # --------------------------------------------------------------------

        name_alias = (
            db.query(PersonAlias)
            .filter_by(
                person_id=self_person.id,
                alias=new_name,
            )
            .first()
        )

        if not name_alias:

            db.add(
                PersonAlias(
                    person_id=self_person.id,
                    alias=new_name,
                    source=Source.DIRECT_STATEMENT,
                    confidence=1.0,
                )
            )

        # --------------------------------------------------------------------
        # 기존 관측 Person이 자기 자신이었다면 비활성화
        # --------------------------------------------------------------------

        if (
            old_person
            and old_person.id != self_person.id
            and old_person.person_key != "self"
            and not old_person.confirmed
        ):

            old_person.status = "inactive"

        db.commit()

        return (
            f"{sender} -> {new_name} "
            f"(self) 연결햇어"
        )

    # =========================================================================
    # OTHER PERSON
    # =========================================================================

    # 이미 등록된 다른 Person이 이름이면 그 사람으로 연결
    target_person = get_person_by_alias(
        db,
        new_name
    )

    if not target_person:

        target_person = (
            get_person_by_identity(
                db,
                new_name
            )
        )

    # 없으면 새로운 사람
    if not target_person:

        target_person = Person(
            person_key=make_person_key(db),
            canonical_name=new_name,
            person_type="person",
            status="active",
            observed_in_chat=1,
            confirmed=1,
        )

        db.add(target_person)
        db.commit()
        db.refresh(target_person)

    else:

        target_person.status = "active"
        target_person.confirmed = 1
        target_person.observed_in_chat = 1

    # ------------------------------------------------------------------------
    # 기존 sender Identity -> target
    # ------------------------------------------------------------------------

    if current_identity:

        current_identity.person_id = target_person.id
        current_identity.target_key = (
            target_person.person_key
        )
        current_identity.platform = "kakaotalk"
        current_identity.is_primary = 1

    else:

        db.add(
            Identity(
                person_id=target_person.id,
                target_key=target_person.person_key,
                platform="kakaotalk",
                display_name=sender,
                is_primary=1,
            )
        )

    # ------------------------------------------------------------------------
    # sender alias
    # ------------------------------------------------------------------------

    sender_alias = (
        db.query(PersonAlias)
        .filter_by(
            person_id=target_person.id,
            alias=sender,
        )
        .first()
    )

    if not sender_alias:

        db.add(
            PersonAlias(
                person_id=target_person.id,
                alias=sender,
                source=Source.DIRECT_STATEMENT,
                confidence=1.0,
            )
        )

    # ------------------------------------------------------------------------
    # new_name alias
    # ------------------------------------------------------------------------

    name_alias = (
        db.query(PersonAlias)
        .filter_by(
            person_id=target_person.id,
            alias=new_name,
        )
        .first()
    )

    if not name_alias:

        db.add(
            PersonAlias(
                person_id=target_person.id,
                alias=new_name,
                source=Source.DIRECT_STATEMENT,
                confidence=1.0,
            )
        )

    # canonical name
    target_person.canonical_name = new_name

    db.commit()

    return (
        f"{sender} -> {new_name} "
        f"({target_person.person_key}) 연결햇어"
    )


# ============================================================================
# 이름 목록
# ============================================================================

def command_name_list(db):

    persons = (
        db.query(Person)
        .filter(
            Person.status.in_(
                ["active", "merged"]
            )
        )
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
            .filter_by(
                person_id=person.id
            )
            .order_by(PersonAlias.id.asc())
            .all()
        )

        identities = (
            db.query(Identity)
            .filter_by(
                person_id=person.id,
                platform="kakaotalk"
            )
            .all()
        )

        alias_text = ", ".join(
            alias.alias
            for alias in aliases
            if alias.alias
        )

        identity_text = ", ".join(
            identity.display_name
            for identity in identities
            if identity.display_name
        )

        role = (
            "본인"
            if person.person_key == "self"
            else "타인"
        )

        observed = (
            "채팅에서 봄"
            if person.observed_in_chat
            else "채팅에서 아직 못 봄"
        )

        lines.append(
            f"{person.person_key} | "
            f"{person.canonical_name} | "
            f"{role} | "
            f"{observed} | "
            f"별칭: {alias_text or '-'} | "
            f"카톡ID: {identity_text or '-'}"
        )

    return "\n".join(lines)


# ============================================================================
# 이름 삭제
# ============================================================================

def command_name_delete(db, name):

    name = str(name or "").strip()

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
        return (
            f"{name}이라는 이름 연결은 없어"
        )

    db.commit()

    return (
        f"{name} 이름 연결만 삭제햇어 "
        f"(대화/기억은 그대로임)"
    )


# ============================================================================
# 인물 등록
# ============================================================================

def command_person(
    db,
    canonical_name,
    aliases
):

    canonical_name = str(
        canonical_name or ""
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
                confirmed=1,
            )

            db.add(person)
            db.commit()
            db.refresh(person)

    for name in [canonical_name] + aliases:

        name = str(name or "").strip()

        if not name:
            continue

        existing_alias = (
            db.query(PersonAlias)
            .filter_by(alias=name)
            .first()
        )

        if existing_alias:

            if (
                existing_alias.person_id
                != person.id
            ):

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
                confidence=1.0,
            )
        )

    person.confirmed = 1
    person.status = "active"

    db.commit()

    return (
        f"{person.canonical_name} 등록햇어 "
        f"({person.person_key})"
    )


# ============================================================================
# 인물 삭제
# ============================================================================

def command_person_delete(
    db,
    name
):

    name = str(name or "").strip()

    person = get_person_by_alias(
        db,
        name
    )

    if not person:
        return (
            f"{name}이라는 인물을 못 찾겠어"
        )

    if person.person_key == "self":

        return (
            "본인은 /인물삭제 말고 "
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


# ============================================================================
# 인물 병합
# ============================================================================

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

    if old_person.person_key == "self":
        return "self는 다른 사람으로 병합할 수 없어"

    # ------------------------------------------------------------------------
    # Alias
    # ------------------------------------------------------------------------

    for alias in (
        db.query(PersonAlias)
        .filter_by(
            person_id=old_person.id
        )
        .all()
    ):

        duplicate = (
            db.query(PersonAlias)
            .filter(
                PersonAlias.alias == alias.alias,
                PersonAlias.person_id
                == target_person.id,
            )
            .first()
        )

        if duplicate:
            db.delete(alias)
        else:
            alias.person_id = target_person.id

    # ------------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------------

    for identity in (
        db.query(Identity)
        .filter_by(
            person_id=old_person.id
        )
        .all()
    ):

        identity.person_id = target_person.id
        identity.target_key = (
            target_person.person_key
        )

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
    "/초기화",
)


def is_command(text):

    text = str(text or "").strip()

    return bool(
        text
        and text.startswith(
            COMMAND_PREFIXES
        )
    )


# ============================================================================
# 본인 말투 학습
# ============================================================================

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
# Persona
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

중요:
너에게 전달되는 카카오톡 sender 이름은
반드시 실제 인물의 본명과 같은 것이 아니다.

예:
- 카카오톡 프로필명: 성시연
- 실제 Person: self
- 실제 이름: 이태양

이 경우 성시연은 타인이 아니다.
성시연은 이태양 본인이 사용하는 카카오톡 식별명이다.

[Identity 최우선 규칙]

현재 발화자의 Person key가 "self"라면
그 발화자는 무조건 이태양 본인이다.

현재 sender가 "성시연"이어도
Person key가 self라면
성시연을 타인으로 해석하지 않는다.

과거 Conversation에서도
speaker_id == self이면
speaker_name이 성시연이든 다른 이름이든
모두 이태양 본인의 발화다.

반대로 Person key가 person_XXX라면
그 사람은 이태양이 아닌 타인이다.

즉 이름 문자열보다 Person key / Identity 관계를 우선한다.

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
4. 단순히 누군가를 욕하거나 평가한 기록을
   그 사람의 고정된 성격으로 단정하지 않는다.
5. 농담으로 한 말을 진짜 신념으로 취급하지 않는다.
6. 실제 Conversation 기록에 대화가 있으면 실제 대화 사실로 취급한다.
7. 현재 방 참가자와 과거에 대화했던 사람은 구분한다.
8. 현재 방 참가자 정보가 없으면 과거 기록만으로
   현재 방에 있다고 단정하지 않는다.
9. 모르는 내용은 아는 척하지 않는다.

[사람별 관계 보정]

Person마다 다음과 같은 차이를 기억할 수 있다.

- 진지하게 대화하는 사람
- 장난을 많이 치는 사람
- 욕하면서 편하게 대화하는 사람
- 게임 얘기를 많이 하는 사람
- 상담이나 고민을 이야기하는 사람
- 특정 주제로 자주 이야기하는 사람

하지만 이것은 별도의 성격을 창조하는 것이 아니다.

기본 짭태양 성격은 유지하면서
실제 관계에 따라 표현 방식만 조금 보정한다.

예:
"이씨한테는 진정성 있게 말하는 편"
"변씨한테는 욕쟁이처럼 편하게 말하는 편"

이런 기억이 있다면
이씨에게는 조금 더 진지하게,
변씨에게는 조금 더 거칠고 장난스럽게 말할 수 있다.

하지만 없는 관계를 상상해서 만들면 안 된다.

{STYLE_RULES}
"""


# ============================================================================
# Memory category
# ============================================================================

def normalize_memory_category(category):

    category = str(
        category or ""
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
# Memory judgement
# ============================================================================

def parse_memory_judgement(text):

    if not text:
        return None

    cleaned = str(text).strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    ).strip()

    try:

        data = json.loads(cleaned)

    except Exception:

        match = re.search(
            r"\{.*\}",
            cleaned,
            flags=re.DOTALL,
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
        False,
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
                "예",
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
            "",
        )
    ).strip()[:300].rstrip()

    people_involved = data.get(
        "people_involved",
        [],
    )

    if not isinstance(
        people_involved,
        list,
    ):
        people_involved = []

    cleaned_people = []

    for person in people_involved:

        person = str(
            person or ""
        ).strip()

        if (
            person
            and person not in cleaned_people
        ):
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
        "people_involved": cleaned_people,
    }


# ============================================================================
# Long-term memory judge
# ============================================================================

def judge_long_term_memory(
    speaker_name,
    speaker_key,
    user_input,
    recent_history=None,
    existing_memories=None,
    known_people=None,
):

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

현재 발화를 보고 앞으로도 사람을 이해하는 데 도움이 되는
안정적인 정보만 장기기억으로 저장한다.

가장 중요한 것은
"누가 말했는가"
"그 사람이 자기 자신에 대해 말했는가"
"다른 사람에 대해 말했는가"
를 구분하는 것이다.

현재 발화자:
- 이름: {speaker_name}
- Person key: {speaker_key}

[Identity 규칙]

Person key가 self이면 현재 발화자는 이태양 본인이다.

sender 이름이 성시연이어도
speaker_key가 self이면 이태양 본인이다.

[저장 규칙]

1. 사람이 자기 자신의 취향/습관/사실을 직접 말한 경우 저장 가능.

예:
만세: "나 떡볶이 좋아해"

=> 저장
=> 대상: 만세
=> category: 취향
=> memory: "떡볶이를 좋아함"

2. self가 자기 정보를 직접 말한 경우도 저장 가능.

예:
성시연(self):
"나 떡볶이 좋아해"

=> 대상: self
=> 확정

3. 자기 관계나 대화방식을 직접 설명한 경우 저장 가능.

예:
만세:
"나 이씨랑은 서로 욕하면서 놈"

=> category: 관계 또는 대화스타일

4. 제3자가 다른 사람의 정보를 말하는 경우
확정하지 않는다.

예:
이씨:
"만세 떡볶이 좋아하던데"

=> 유용하다면 저장 가능
=> 하지만 status는 candidate

5. 단순한 순간 발언은 저장하지 않는다.

예:
"만세 오늘 개웃김"
"만세 오늘 짜증남"
"이태양 바보같아 ㅋㅋ"
"나 지금 개빡침"

6. 농담/과장/비유/밈은 저장하지 않는다.

예:
"나 떡볶이 백만개 먹을 수 있음"
"나 게임 평생 안 접음"

7. 단순 욕이나 순간적인 가치판단은 저장하지 않는다.

예:
"이태양 바보같음"
"변씨 개싸가지"
"만세 존나 답답함"

이것만으로 고정적인 관계나 성격을 만들지 않는다.

8. 반복적이고 안정적인 관계 패턴은 저장할 수 있다.

예:
"이씨랑은 항상 진지하게 얘기함"
"변씨랑은 서로 욕하면서 장난침"
"만세랑은 게임 얘기를 자주 함"

9. 한 번의 발화만으로
"누구는 나와 친하다"
"누구는 나를 싫어한다"
같은 관계를 확정하지 않는다.

10. 사람의 성격 자체를 함부로 만들어내지 않는다.

11. people_involved는 실제 Person key를 우선한다.

현재 발화자가 self라면 자기 정보의 대상은
["self"]

현재 발화자가 person_001이라면
["person_001"]

두 사람의 관계라면:
["person_001", "person_002"]

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

category:
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
                    ],
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
# Memory duplicate
# ============================================================================

def memory_similarity_exists(
    db,
    memory_text,
    people_involved,
):

    existing = (
        db.query(Memory)
        .filter(
            Memory.persona_id
            == get_persona_id()
        )
        .all()
    )

    normalized_new = (
        str(memory_text or "")
        .strip()
        .lower()
    )

    new_people = set(
        map(
            str,
            people_involved or []
        )
    )

    for old in existing:

        old_text = (
            str(old.content or "")
            .strip()
            .lower()
        )

        old_people = set(
            map(
                str,
                old.people_involved or []
            )
        )

        if new_people != old_people:
            continue

        if (
            normalized_new == old_text
            or normalized_new in old_text
            or old_text in normalized_new
        ):
            return True

    return False


# ============================================================================
# Memory save
# ============================================================================

def save_auto_memory_if_worthy(
    conversation_key,
    speaker_name,
    speaker_key,
    user_input,
    is_self,
    recent_history=None,
    existing_memories=None,
    known_people=None,
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
        "기타",
    )

    memory = result.get(
        "memory",
        "",
    )

    people = result.get(
        "people_involved",
        [],
    )

    db = SessionLocal()

    try:

        # --------------------------------------------------------------------
        # people -> Person key 정규화
        # --------------------------------------------------------------------

        normalized_people = []

        for person_value in people:

            person_value = str(
                person_value or ""
            ).strip()

            if not person_value:
                continue

            person_obj = get_person_by_key(
                db,
                person_value
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

        # --------------------------------------------------------------------
        # 모델이 대상 인물을 안 줬으면 발화자
        # --------------------------------------------------------------------

        if not normalized_people:
            normalized_people = [
                speaker_key
            ]

        normalized_people = list(
            dict.fromkeys(
                normalized_people
            )
        )

        # --------------------------------------------------------------------
        # 중복
        # --------------------------------------------------------------------

        if memory_similarity_exists(
            db,
            memory,
            normalized_people,
        ):
            return None

        # --------------------------------------------------------------------
        # 자기 자신에 대한 직접 진술인지
        # --------------------------------------------------------------------

        speaker_is_subject = (
            speaker_key
            in normalized_people
        )

        if speaker_is_subject:

            status = ItemStatus.CONFIRMED

        else:

            status = ItemStatus.CANDIDATE

        # --------------------------------------------------------------------
        # Source
        # --------------------------------------------------------------------

        if speaker_is_subject:

            source = Source.DIRECT_STATEMENT

        else:

            source = Source.INFORMANT

        new_memory = Memory(
            persona_id=get_persona_id(),
            memory_type=MemoryType.FACT,
            content=memory,
            context=category,
            people_involved=normalized_people,
            source=source,
            status=status,
        )

        db.add(new_memory)
        db.commit()

        print(
            "[memory] 저장: "
            f"{category} - {memory} "
            f"(대상={normalized_people}, "
            f"상태="
            f"{'확정' if status == ItemStatus.CONFIRMED else '후보'})"
        )

        return {
            "category": category,
            "memory": memory,
            "people_involved": normalized_people,
            "status": (
                "confirmed"
                if status == ItemStatus.CONFIRMED
                else "candidate"
            ),
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
# Background learning
# ============================================================================

def background_learning(
    conversation_key,
    speaker_name,
    speaker_key,
    user_input,
    recent_history,
    existing_memories,
    is_self,
    known_people,
):

    # ------------------------------------------------------------------------
    # self 말투 학습
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
            f"[auto memory] 실패: "
            f"{repr(e)}"
        )


# ============================================================================
# 실제 대화 상대
# ============================================================================

def get_interaction_context(
    db,
    limit=500,
):

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
# 언급 인물
# ============================================================================

def extract_mentioned_people(
    db,
    text,
):

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

        names.extend(
            [
                str(alias.alias).strip()
                for alias in aliases
                if alias.alias
            ]
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

    # ------------------------------------------------------------------------
    # self 이름도 질문에서 찾는다.
    # ------------------------------------------------------------------------

    self_person = get_self_person(db)

    if self_person:

        names = []

        if self_person.canonical_name:
            names.append(
                self_person.canonical_name
            )

        aliases = (
            db.query(PersonAlias)
            .filter_by(
                person_id=self_person.id
            )
            .all()
        )

        names.extend(
            [
                alias.alias
                for alias in aliases
                if alias.alias
            ]
        )

        for name in names:

            if (
                name
                and name in text
            ):

                found["self"] = self_person
                break

    return found


# ============================================================================
# Memory context
# ============================================================================

def get_memory_context_for_query(
    db,
    target_key,
    user_input,
):

    mentioned_people = extract_mentioned_people(
        db,
        user_input
    )

    relevant_keys = {
        target_key
    }

    relevant_keys.update(
        mentioned_people.keys()
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

        # 직접 관련 인물
        if any(
            key in involved
            for key in relevant_keys
        ):

            selected.append(mem)
            continue

        # 내용에 언급된 인물 이름이 있는 경우
        matched = False

        for (
            person_key,
            person_obj
        ) in mentioned_people.items():

            names = [
                person_obj.canonical_name
            ]

            aliases = (
                db.query(PersonAlias)
                .filter_by(
                    person_id=person_obj.id
                )
                .all()
            )

            names.extend(
                [
                    alias.alias
                    for alias in aliases
                    if alias.alias
                ]
            )

            for name in names:

                if not name:
                    continue

                if (
                    name in str(
                        mem.content or ""
                    )
                ):

                    matched = True
                    break

            if matched:
                break

        if matched:

            selected.append(mem)
            continue

        # 내용 키워드
        for keyword in keywords:

            if (
                keyword.lower()
                in str(
                    mem.content or ""
                ).lower()
            ):

                selected.append(mem)
                break

    result = []
    seen = set()

    for mem in selected:

        if mem.id in seen:
            continue

        seen.add(mem.id)
        result.append(mem)

        if len(result) >= 80:
            break

    return result, mentioned_people


# ============================================================================
# Request
# ============================================================================

class ChatRequest(BaseModel):

    sender: str
    message: str

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
        "model": MODEL_NAME,
    }


# ============================================================================
# Chat
# ============================================================================

@app.post("/chat")
def reply_chat(
    req: ChatRequest,
    background_tasks: BackgroundTasks,
):

    raw_input = str(
        req.message or ""
    ).strip()

    if not raw_input:

        return {
            "reply": "뭐라고"
        }

    # ------------------------------------------------------------------------
    # 호출 태그 제거
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
    # sender -> Person
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
    # Reset
    # =========================================================================

    if user_input in [
        "/리셋",
        "/초기화",
    ]:

        try:

            conn = sqlite3.connect(
                "taeyang.db"
            )

            cur = conn.cursor()

            cur.execute(
                "DELETE FROM messages "
                "WHERE user_id = ?",
                (conversation_key,),
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
    # 명령어
    # =========================================================================

    db = SessionLocal()

    try:

        if user_input in [
            "/이름목록",
            "/이름 목록",
            "/인물목록",
            "/인물 목록",
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
    # Conversation - 사용자 발화
    # =========================================================================

    log_conversation(
        speaker_name=req.sender,
        target_key=target_key,
        message=req.message,
        room_id="dm",
    )

    # =========================================================================
    # 기억 목록
    # =========================================================================

    if user_input in [
        "/기억목록",
        "/기억 목록",
        "/기억리스트",
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
                            or []
                        )
                    )
                    or "불명"
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
                    status=ItemStatus.CONFIRMED,
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
    # Context
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
                conv.message,
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
    # Memory
    # =========================================================================

    interaction_context = {}
    relevant_memories = []
    mentioned_people = {}

    try:

        db_mem = SessionLocal()

        try:

            (
                relevant_memories,
                mentioned_people,
            ) = get_memory_context_for_query(
                db_mem,
                target_key,
                user_input,
            )

            interaction_context = (
                get_interaction_context(
                    db_mem,
                    limit=500,
                )
            )

        finally:

            db_mem.close()

    except Exception as e:

        print(
            f"[memory load error] "
            f"{repr(e)}"
        )

    # =========================================================================
    # Person
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
                if alias.alias
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
    # Gemini context
    # =========================================================================

    contents = []

    context_parts = [
        f"[현재 한국 시각]: {now_kst}"
    ]

    # -------------------------------------------------------------------------
    # 현재 발화자 Identity
    # -------------------------------------------------------------------------

    if person:

        if target_key == "self":

            context_parts.append(
                "[현재 발화자 = 이태양 본인]\n"
                "Person key: self\n"
                f"현재 카카오톡 sender: {req.sender}\n"
                f"등록된 본인 이름: {person.canonical_name}\n"
                "중요: 현재 sender 이름은 프로필명일 뿐이며 "
                "타인을 의미하지 않는다.\n"
                "현재 발화자는 이태양 본인이다."
            )

        else:

            context_parts.append(
                "[현재 발화자 = 타인]\n"
                f"Person key: {target_key}\n"
                f"공식 이름: {person.canonical_name}\n"
                f"현재 카카오톡 sender: {req.sender}\n"
                "현재 발화자는 이태양이 아닌 타인이다."
            )

    # -------------------------------------------------------------------------
    # self Identity 별칭
    # -------------------------------------------------------------------------

    db_identity = SessionLocal()

    try:

        self_person = get_self_person(
            db_identity
        )

        if self_person:

            self_aliases = (
                db_identity.query(
                    PersonAlias
                )
                .filter_by(
                    person_id=self_person.id
                )
                .order_by(
                    PersonAlias.id.asc()
                )
                .all()
            )

            self_identities = (
                db_identity.query(
                    Identity
                )
                .filter_by(
                    person_id=self_person.id,
                    platform="kakaotalk",
                )
                .all()
            )

            alias_names = [
                a.alias
                for a in self_aliases
                if a.alias
            ]

            identity_names = [
                i.display_name
                for i in self_identities
                if i.display_name
            ]

            context_parts.append(
                "[이태양 본인의 Identity]\n"
                f"Person key: self\n"
                f"본명/대표 이름: "
                f"{self_person.canonical_name}\n"
                f"별칭/프로필명: "
                f"{', '.join(alias_names) or '-'}\n"
                f"카카오톡 Identity: "
                f"{', '.join(identity_names) or '-'}\n"
                "위 이름들은 모두 같은 사람, 즉 이태양 본인을 가리킬 수 있다."
            )

    finally:

        db_identity.close()

    # -------------------------------------------------------------------------
    # 현재 상대 관계 기억
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
    # 알고 있는 인물
    # -------------------------------------------------------------------------

    if people_lines:

        context_parts.append(
            "[알고 있는 인물 목록]\n"
            + "\n".join(
                people_lines
            )
        )

    # -------------------------------------------------------------------------
    # 실제 대화 상대
    # -------------------------------------------------------------------------

    if interaction_context:

        interaction_lines = []

        for (
            person_key,
            info,
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
            "speaker_id=self인 기록은 이태양 본인의 발화이므로 "
            "대화 상대 목록에 포함하지 않는다."
        )

    # -------------------------------------------------------------------------
    # 현재 방
    # -------------------------------------------------------------------------

    clean_members = []

    for name in req.room_members:

        name = str(name or "").strip()

        if (
            name
            and name not in clean_members
        ):

            clean_members.append(name)

    if clean_members:

        context_parts.append(
            "[현재 카카오톡 방 참가자 목록]\n"
            + "\n".join(
                f"- {name}"
                for name in clean_members
            )
            + "\n"
            "이 목록은 현재 방 참가자일 뿐이다. "
            "과거 Conversation 기록과 혼동하지 않는다."
        )

    # -------------------------------------------------------------------------
    # 관련 장기기억
    # -------------------------------------------------------------------------

    if relevant_memories:

        memory_lines = []

        for mem in relevant_memories:

            involved = (
                ", ".join(
                    map(
                        str,
                        mem.people_involved
                        or []
                    )
                )
                or "불명"
            )

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

            memory_lines.append(
                f"- [{category}] "
                f"[대상: {involved}] "
                f"{mem.content} "
                f"({status})"
            )

        context_parts.append(
            "[현재 질문에 관련된 장기기억]\n"
            + "\n".join(
                memory_lines
            )
        )

    # -------------------------------------------------------------------------
    # 언급 인물
    # -------------------------------------------------------------------------

    if mentioned_people:

        mentioned_lines = []

        for (
            person_key,
            person_obj,
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

    try:

        style_examples = [
            str(e).strip()
            for e in (
                legacy.get_relevant_style_samples(
                    user_input,
                    n=12,
                )
            )
            if should_learn_style(
                str(e).strip()
            )
        ]

    except Exception:

        style_examples = []

    if style_examples:

        context_parts.append(
            "[이태양 실제 말투 예시]\n"
            + "\n".join(
                f"- {e}"
                for e in style_examples
            )
        )

    # -------------------------------------------------------------------------
    # Context
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
            ],
        )
    )

    contents.append(
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text="ㅇㅋ 확인햇어"
                )
            ],
        )
    )

    # =========================================================================
    # 최근 대화
    # =========================================================================

    for (
        history_speaker_id,
        history_sender,
        text,
    ) in recent_history:

        text = str(text or "").strip()

        if (
            not text
            or is_command(text)
        ):
            continue

        # ---------------------------------------------------------------------
        # speaker_id=self -> 무조건 이태양 본인
        # ---------------------------------------------------------------------

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
                ],
            )
        )

    # =========================================================================
    # 현재 발화
    # =========================================================================

    if target_key == "self":

        current_text = (
            f"[현재 발화자: 이태양 본인]\n"
            f"[카카오톡 프로필명: {req.sender}]\n"
            f"{user_input}"
        )

    else:

        current_text = (
            f"[현재 발화자: {person.canonical_name if person else req.sender}]\n"
            f"[카카오톡 프로필명: {req.sender}]\n"
            f"{user_input}"
        )

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=current_text
                )
            ],
        )
    )

    # =========================================================================
    # Gemini
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

        reply = re.sub(
            r"^```(?:text)?\s*",
            "",
            reply,
            flags=re.IGNORECASE,
        )

        reply = re.sub(
            r"\s*```$",
            "",
            reply,
        ).strip()

        if reply.startswith("```"):

            reply = re.sub(
                r"^```.*?\n",
                "",
                reply,
                flags=re.DOTALL,
            )

            reply = re.sub(
                r"\n```$",
                "",
                reply,
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
    # Legacy
    # =========================================================================

    try:

        legacy.save_message(
            conversation_key,
            conversation_key,
            user_input,
        )

        legacy.save_message(
            conversation_key,
            "이태양",
            reply,
        )

    except Exception:
        pass

    # =========================================================================
    # AI Conversation
    # =========================================================================

    log_conversation(
        speaker_name="이태양",
        target_key="self",
        message=reply,
        room_id="dm",
    )

    # =========================================================================
    # Background learning
    # =========================================================================

    background_tasks.add_task(
        background_learning,
        conversation_key,
        req.sender,
        target_key,
        user_input,
        recent_history,
        [
            mem.content
            for mem in relevant_memories
        ],
        is_self,
        known_people_for_memory,
    )

    return {
        "reply": reply
    }
