"""
장산범 Persona Engine v2 - main.py

핵심:
- /이름 이태양 -> 현재 sender를 self로 연결
- 이후 해당 sender는 self로 인식
- Identity / Person / Alias 분리
- 기존 legacy 기억/대화 유지
- 기존 /chat 계약 유지
"""

import os
import sqlite3
import re

from datetime import datetime, timezone, timedelta

from fastapi import FastAPI
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
    PersonaRelationship,
    Source,
)

import seed_and_migrate


# ---------------------------------------------------------------------------
# 초기화
# ---------------------------------------------------------------------------

seed_and_migrate.run()

imported_count = legacy.import_style_samples(
    "style_samples.txt"
)

if imported_count:
    print(
        f"[legacy] style_samples.txt "
        f"{imported_count}개 로드"
    )


GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY 환경변수를 설정해주세요."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


MODEL_NAME = "gemini-3.6-flash"


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

            _PERSONA_ID_CACHE["id"] = (
                persona.id
            )

        finally:

            db.close()

    return _PERSONA_ID_CACHE["id"]


# ---------------------------------------------------------------------------
# Person / Identity
# ---------------------------------------------------------------------------

def get_person_by_key(
    db,
    person_key,
):

    return (
        db.query(Person)
        .filter_by(
            person_key=person_key
        )
        .first()
    )


def get_person_by_alias(
    db,
    alias,
):

    alias = alias.strip()

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


def get_person_by_identity(
    db,
    display_name,
):

    display_name = display_name.strip()

    if not display_name:
        return None

    identity = (
        db.query(Identity)
        .filter_by(
            display_name=display_name,
            platform="kakaotalk",
        )
        .first()
    )

    if identity:
        return identity.person

    return None


def make_person_key(db):
    """
    person_001
    person_002
    ...
    """

    rows = (
        db.query(Person)
        .filter(
            Person.person_key.like(
                "person_%"
            )
        )
        .all()
    )

    max_number = 0

    for person in rows:

        match = re.match(
            r"person_(\d+)$",
            person.person_key,
        )

        if match:

            max_number = max(
                max_number,
                int(match.group(1)),
            )

    return (
        f"person_{max_number + 1:03d}"
    )


# ---------------------------------------------------------------------------
# 현재 sender -> Person
# ---------------------------------------------------------------------------

def get_or_create_observed_person(
    db,
    display_name,
):
    """
    실제 카톡 sender를 Person으로 해석한다.

    우선순위:

    1. Identity
    2. Alias
    3. 신규 Person
    """

    display_name = display_name.strip()

    if not display_name:
        raise ValueError(
            "display_name이 비어있음"
        )

    # ---------------------------------------------------------
    # 1. Identity
    # ---------------------------------------------------------

    identity = (
        db.query(Identity)
        .filter_by(
            display_name=display_name,
            platform="kakaotalk",
        )
        .first()
    )

    if identity:

        person = identity.person

        if person.status == "inactive":
            person.status = "active"

        person.observed_in_chat = 1

        db.commit()

        return (
            person,
            False,
        )

    # ---------------------------------------------------------
    # 2. Alias
    # ---------------------------------------------------------

    person = get_person_by_alias(
        db,
        display_name,
    )

    if person:

        if person.status == "inactive":
            person.status = "active"

        person.observed_in_chat = 1

        # Identity가 없으므로 새로 연결
        identity = Identity(
            person_id=person.id,
            target_key=person.person_key,
            platform="kakaotalk",
            display_name=display_name,
            is_primary=1,
        )

        db.add(identity)

        db.commit()

        print(
            f"[identity] 알려진 인물 별칭 매칭: "
            f"{display_name} -> "
            f"{person.person_key}"
        )

        return (
            person,
            False,
        )

    # ---------------------------------------------------------
    # 3. 처음 보는 사람
    # ---------------------------------------------------------

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

    # sender 이름 alias
    db.add(
        PersonAlias(
            person_id=person.id,
            alias=display_name,
            source=Source.OBSERVED,
            confidence=0.5,
        )
    )

    # 실제 카카오톡 Identity
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

    print(
        f"[identity] 새로운 사람 등록: "
        f"{display_name} -> "
        f"{person_key}"
    )

    return (
        person,
        True,
    )


# ---------------------------------------------------------------------------
# 이름 명령
# ---------------------------------------------------------------------------

def command_name(
    db,
    sender,
    new_name,
):
    """
    현재 카카오톡 sender를 self로 연결한다.

    예:

        sender = 시연

        /이름 이태양

    결과:

        시연 -> self
        이태양 -> self

    기존 대화/기억은 삭제하지 않는다.
    """

    sender = sender.strip()
    new_name = new_name.strip()

    if not sender:
        return "sender가 없어"

    if not new_name:
        return "이름을 적어줘"

    # ---------------------------------------------------------
    # 1. self Person 확보
    # ---------------------------------------------------------

    self_person = (
        db.query(Person)
        .filter_by(
            person_key="self"
        )
        .first()
    )

    if not self_person:

        self_person = Person(
            person_key="self",
            canonical_name=new_name,
            person_type="self",
            status="active",
            observed_in_chat=1,
            confirmed=1,
        )

        db.add(self_person)

        db.commit()

        db.refresh(self_person)

    # ---------------------------------------------------------
    # 2. self 정보 갱신
    # ---------------------------------------------------------

    self_person.canonical_name = new_name
    self_person.person_type = "self"
    self_person.status = "active"
    self_person.confirmed = 1
    self_person.observed_in_chat = 1

    # ---------------------------------------------------------
    # 3. 현재 sender Identity 찾기
    # ---------------------------------------------------------

    current_identity = (
        db.query(Identity)
        .filter_by(
            display_name=sender,
            platform="kakaotalk",
        )
        .first()
    )

    old_person = None

    if current_identity:
        old_person = current_identity.person

    # ---------------------------------------------------------
    # 4. sender가 다른 Person의 alias로 남아있다면 제거
    # ---------------------------------------------------------

    sender_aliases = (
        db.query(PersonAlias)
        .filter_by(
            alias=sender
        )
        .all()
    )

    for alias in sender_aliases:

        if alias.person_id != self_person.id:
            db.delete(alias)

    db.flush()

    # ---------------------------------------------------------
    # 5. sender Identity -> self
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 6. sender를 self alias로 등록
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 7. 새 이름도 self alias
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # 8. 기존 자동 생성 Person 정리
    #
    # Person 자체는 삭제하지 않는다.
    # 기존 대화/기억도 삭제하지 않는다.
    # ---------------------------------------------------------

    if (
        old_person
        and old_person.id != self_person.id
        and old_person.person_key not in [
            "self",
            "cha",
        ]
        and not old_person.confirmed
    ):

        other_alias_count = (
            db.query(PersonAlias)
            .filter(
                PersonAlias.person_id
                == old_person.id,
                PersonAlias.alias
                != sender,
            )
            .count()
        )

        if other_alias_count == 0:

            old_person.status = "inactive"

    db.commit()

    print(
        f"[identity] 본인 연결: "
        f"{sender} -> {new_name} -> self"
    )

    return (
        f"{sender} -> {new_name} "
        f"(self) 연결햇어"
    )


# ---------------------------------------------------------------------------
# 이름 목록
# ---------------------------------------------------------------------------

def command_name_list(db):

    persons = (
        db.query(Person)
        .filter_by(
            status="active"
        )
        .order_by(
            Person.id.asc()
        )
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
            .order_by(
                PersonAlias.id.asc()
            )
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
            f"별칭: "
            f"{alias_text or '-'}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 이름 삭제
# ---------------------------------------------------------------------------

def command_name_delete(
    db,
    name,
):
    """
    이름 연결만 삭제한다.

    Person / 대화 / 기억은 삭제하지 않는다.
    """

    name = name.strip()

    if not name:
        return "삭제할 이름을 적어줘"

    deleted = False

    # Identity
    identities = (
        db.query(Identity)
        .filter_by(
            display_name=name,
            platform="kakaotalk",
        )
        .all()
    )

    for identity in identities:

        db.delete(identity)

        deleted = True

    # Alias
    aliases = (
        db.query(PersonAlias)
        .filter_by(
            alias=name
        )
        .all()
    )

    for alias in aliases:

        db.delete(alias)

        deleted = True

    if not deleted:
        return (
            f"{name}이라는 이름은 없어"
        )

    db.commit()

    return (
        f"{name} 이름 연결만 삭제햇어 "
        "(대화/기억은 그대로임)"
    )


# ---------------------------------------------------------------------------
# 인물
# ---------------------------------------------------------------------------

def command_person(
    db,
    canonical_name,
    aliases,
):
    canonical_name = canonical_name.strip()

    if not canonical_name:
        return "인물 이름을 적어줘"

    existing = get_person_by_alias(
        db,
        canonical_name,
    )

    if existing:

        person = existing

    else:

        existing_identity = (
            get_person_by_identity(
                db,
                canonical_name,
            )
        )

        if existing_identity:

            person = (
                existing_identity
                .person
            )

        else:

            person = Person(
                person_key=make_person_key(
                    db
                ),
                canonical_name=canonical_name,
                person_type="person",
                status="active",
                observed_in_chat=0,
                confirmed=1,
            )

            db.add(person)

            db.commit()

            db.refresh(person)

    names = [
        canonical_name
    ] + aliases

    for name in names:

        name = name.strip()

        if not name:
            continue

        existing_alias = (
            db.query(PersonAlias)
            .filter_by(
                alias=name
            )
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
        f"{person.canonical_name} "
        f"등록햇어 "
        f"({person.person_key})"
    )


# ---------------------------------------------------------------------------
# 인물 삭제
# ---------------------------------------------------------------------------

def command_person_delete(
    db,
    name,
):
    """
    인물 자체를 삭제하지 않고 inactive 처리.
    """

    person = get_person_by_alias(
        db,
        name,
    )

    if not person:
        return (
            f"{name}이라는 인물을 "
            f"못 찾겠어"
        )

    if person.person_key == "self":
        return (
            "본인은 인물삭제 말고 "
            "/이름삭제를 써"
        )

    person.status = "inactive"

    identities = (
        db.query(Identity)
        .filter_by(
            person_id=person.id
        )
        .all()
    )

    for identity in identities:

        identity.is_primary = 0

    db.commit()

    return (
        f"{person.canonical_name} "
        f"비활성화햇어 "
        "(대화/기억은 삭제 안 함)"
    )


# ---------------------------------------------------------------------------
# 인물 병합
# ---------------------------------------------------------------------------

def command_person_merge(
    db,
    old_name,
    target_name,
):
    """
    기존 인물을 다른 인물로 병합한다.

    대화/기억은 삭제하지 않는다.
    """

    old_person = get_person_by_alias(
        db,
        old_name,
    )

    target_person = get_person_by_alias(
        db,
        target_name,
    )

    if not old_person:
        return (
            f"{old_name}을 못 찾겠어"
        )

    if not target_person:
        return (
            f"{target_name}을 못 찾겠어"
        )

    if old_person.id == target_person.id:
        return "이미 같은 사람이야"

    # Alias 이동
    aliases = (
        db.query(PersonAlias)
        .filter_by(
            person_id=old_person.id
        )
        .all()
    )

    for alias in aliases:

        duplicate = (
            db.query(PersonAlias)
            .filter(
                PersonAlias.alias
                == alias.alias,
                PersonAlias.person_id
                == target_person.id,
            )
            .first()
        )

        if duplicate:

            db.delete(alias)

        else:

            alias.person_id = (
                target_person.id
            )

    # Identity 이동
    identities = (
        db.query(Identity)
        .filter_by(
            person_id=old_person.id
        )
        .all()
    )

    for identity in identities:

        identity.person_id = (
            target_person.id
        )

        identity.target_key = (
            target_person.person_key
        )

    old_person.status = "merged"

    old_person.notes = (
        f"merged_into="
        f"{target_person.person_key}"
    )

    target_person.confirmed = 1

    db.commit()

    return (
        f"{old_name} -> "
        f"{target_name} 병합햇어 "
        "(기존 대화/기억은 삭제 안 함)"
    )


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

def log_conversation(
    speaker_name,
    target_key,
    message,
    room_id="dm",
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
            f"[log_conversation] 실패: {e}"
        )

    finally:

        db.close()


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

STYLE_RULES = """
[문장 형식]
1. 카카오톡처럼 짧게 말한다.
2. 기본적으로 1~25자 정도.
3. 한 줄 출력.
4. 설명문처럼 길게 풀지 않는다.
5. 필요할 때만 조금 길어진다.

[말투]
1. 반말.
2. AI 비서처럼 친절하게 굴지 않는다.
3. "~냐" 종결어미는 가급적 피한다.
4. "~어?", "~지", "~네", "~함", "~음" 등을 자연스럽게 사용한다.
5. ㅋㅋㅋ / ㅋㅎㅋㅎ / ㅡㅡ / ㅎ;; 같은 표현을 상황에 맞게 사용한다.
6. 문장부호는 최소화한다.
7. 긍정할 때 무조건 ㅇㅇ만 반복하지 않는다.
8. 실제 카톡에서 사람이 쓸 법한 불완전한 문장을 허용한다.

[금지]
1. "무엇을 도와드릴까요?"
2. "안녕하세요"
3. "좋은 하루 보내세요"
4. AI/시스템/프롬프트/모델이라는 말을 먼저 꺼내지 않는다.
"""


SYSTEM_INSTRUCTION_FOR_SELF = f"""
너는 실제 인물의 말투와 성향을 학습하는 AI 클론 '짭태양'이다.

현재 상대는 페르소나의 실제 본인이다.

이 대화는 페르소나 학습 과정이다.
상대가 자연스럽게 대화하면서 자기 말투, 습관, 취향 등을 알려줄 수 있다.

상대가 직접 정정한 내용은 매우 중요하게 취급한다.

억지로 모든 말을 학습했다고 주장하지 마라.

{STYLE_RULES}
"""


SYSTEM_INSTRUCTION_FOR_CHA = f"""
너는 21살 대학생 이태양의 AI 페르소나다.

상대는 친한 게임 친구다.
친밀도가 높은 친구처럼 자연스럽게 카카오톡으로 대화한다.

기억에 없는 사실은 아는 척하지 않는다.
상대가 알려준 사실과 페르소나 기억을 자연스럽게 활용한다.

{STYLE_RULES}
"""


SYSTEM_INSTRUCTION_FOR_UNKNOWN = f"""
너는 21살 대학생 이태양의 AI 페르소나다.

상대가 누구인지 아직 확실하지 않으면
특정 친구로 단정하지 않는다.

상대에 대한 기억이 없으면 아는 척하지 않는다.

{STYLE_RULES}
"""


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):

    sender: str
    message: str


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def handle_command(
    sender,
    user_input,
):

    db = SessionLocal()

    try:

        # /이름목록
        if user_input in [
            "/이름목록",
            "/이름 목록",
        ]:

            return command_name_list(
                db
            )

        # /이름삭제 이름
        if user_input.startswith(
            "/이름삭제 "
        ):

            name = user_input[
                len("/이름삭제 "):
            ].strip()

            return command_name_delete(
                db,
                name,
            )

        # /이름 이름
        if user_input.startswith(
            "/이름 "
        ):

            name = user_input[
                len("/이름 "):
            ].strip()

            return command_name(
                db,
                sender,
                name,
            )

        # /인물목록
        if user_input in [
            "/인물목록",
            "/인물 목록",
        ]:

            return command_name_list(
                db
            )

        # /인물삭제
        if user_input.startswith(
            "/인물삭제 "
        ):

            name = user_input[
                len("/인물삭제 "):
            ].strip()

            return command_person_delete(
                db,
                name,
            )

        # /인물병합
        if user_input.startswith(
            "/인물병합 "
        ):

            args = user_input[
                len("/인물병합 "):
            ].split()

            if len(args) < 2:

                return (
                    "예: /인물병합 "
                    "배코 백호"
                )

            return command_person_merge(
                db,
                args[0],
                args[1],
            )

        # /인물
        if user_input.startswith(
            "/인물 "
        ):

            args = user_input[
                len("/인물 "):
            ].split()

            if not args:
                return (
                    "예: /인물 백호 배코"
                )

            canonical = args[0]
            aliases = args[1:]

            return command_person(
                db,
                canonical,
                aliases,
            )

        return None

    finally:

        db.close()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():

    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@app.post("/chat")
def reply_chat(
    req: ChatRequest,
):

    raw_input = req.message.strip()

    # 멘션 제거
    user_input = (
        raw_input
        .replace("@짭태양", "")
        .replace("/짭태양", "")
        .strip()
    )

    # ---------------------------------------------------------
    # Command
    # ---------------------------------------------------------

    command_reply = handle_command(
        req.sender,
        user_input,
    )

    if command_reply is not None:

        return {
            "reply": command_reply
        }

    # ---------------------------------------------------------
    # Person 확인
    # ---------------------------------------------------------

    db = SessionLocal()

    try:

        person, is_new_person = (
            get_or_create_observed_person(
                db,
                req.sender,
            )
        )

        target_key = (
            person.person_key
        )

    finally:

        db.close()

    # ---------------------------------------------------------
    # 원본 신규 DB 대화 기록
    # ---------------------------------------------------------

    log_conversation(
        speaker_name=req.sender,
        target_key=target_key,
        message=req.message,
        room_id="dm",
    )

    # legacy conversation key
    conversation_key = target_key

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------

    if user_input in [
        "/리셋",
        "/초기화",
    ]:

        conn = sqlite3.connect(
            "taeyang.db"
        )

        cur = conn.cursor()

        cur.execute(
            """
            DELETE FROM messages
            WHERE user_id = ?
            """,
            (conversation_key,),
        )

        conn.commit()
        conn.close()

        return {
            "reply":
                "대화기록초기화완료"
        }

    # ---------------------------------------------------------
    # 기억 목록
    # ---------------------------------------------------------

    if user_input in [
        "/기억목록",
        "/기억 목록",
        "/기억리스트",
    ]:

        rows = (
            legacy.get_memories_with_id(
                conversation_key
            )
        )

        if not rows:

            return {
                "reply":
                    "기억된 정보가 없어"
            }

        items = [
            f"[{row[0]}] "
            f"{str(row[1]).replace(chr(10), ' ')}"
            for row in rows
        ]

        return {
            "reply":
                " | ".join(items)
        }

    # ---------------------------------------------------------
    # 기억 삭제
    # ---------------------------------------------------------

    if user_input.startswith(
        "/기억삭제"
    ):

        target = (
            user_input
            .replace(
                "/기억삭제",
                "",
            )
            .replace(
                "/기억 삭제",
                "",
            )
            .strip()
        )

        if target.isdigit():

            success = (
                legacy.delete_memory_by_id(
                    conversation_key,
                    int(target),
                )
            )

            if success:

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

    # ---------------------------------------------------------
    # 기억 추가
    # ---------------------------------------------------------

    if user_input.startswith(
        "/기억 "
    ):

        mem_text = user_input[
            len("/기억 "):
        ].strip()

        if mem_text:

            legacy.save_memory(
                conversation_key,
                mem_text,
            )

            return {
                "reply":
                    f"응기억햇어: "
                    f"{mem_text}"
            }

    # ---------------------------------------------------------
    # 말투 추가
    # ---------------------------------------------------------

    if user_input.startswith(
        "/말투 "
    ):

        style_text = user_input[
            len("/말투 "):
        ].strip()

        if style_text:

            legacy.save_style_sample(
                style_text
            )

            return {
                "reply":
                    f"응 이것도 배웟어: "
                    f"{style_text}"
            }

    # ---------------------------------------------------------
    # Self 여부
    # ---------------------------------------------------------

    is_self = (
        target_key == "self"
    )

    # 본인이 직접 한 말이면 말투 학습
    if is_self and user_input:

        legacy.save_style_sample(
            user_input
        )

    # ---------------------------------------------------------
    # 시간
    # ---------------------------------------------------------

    now_kst = datetime.now(
        KST
    )

    current_time_str = (
        now_kst.strftime(
            "%Y년 %m월 %d일 %H시 %M분"
        )
    )

    # ---------------------------------------------------------
    # 최근 대화
    # ---------------------------------------------------------

    recent_history = (
        legacy.get_recent_messages(
            conversation_key,
            limit=6,
        )
    )

    # ---------------------------------------------------------
    # 기억
    # ---------------------------------------------------------

    user_memories = (
        legacy.get_memories(
            conversation_key
        )
    )

    # ---------------------------------------------------------
    # 말투
    # ---------------------------------------------------------

    style_examples = (
        legacy.get_relevant_style_samples(
            user_input,
            n=10,
        )
    )

    # ---------------------------------------------------------
    # System instruction
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
    # Known people
    # ---------------------------------------------------------

    db = SessionLocal()

    try:

        person = get_person_by_key(
            db,
            target_key,
        )

        people_lines = []

        known_people = (
            db.query(Person)
            .filter_by(
                status="active"
            )
            .order_by(
                Person.id.asc()
            )
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
                f"{known_person.canonical_name}"
                f" ({alias_text})"
            )

    finally:

        db.close()

    # ---------------------------------------------------------
    # Gemini Context
    # ---------------------------------------------------------

    contents = []

    context_parts = [

        f"[현재 한국 시각]: "
        f"{current_time_str}",

    ]

    if person:

        context_parts.append(
            f"[현재 상대]: "
            f"{person.canonical_name} "
            f"/ {target_key}"
        )

    if people_lines:

        context_parts.append(
            "[알고 있는 인물 목록]\n"
            + "\n".join(
                people_lines
            )
        )

    if user_memories:

        context_parts.append(
            "[현재 상대에 대한 기억]\n"
            + "\n".join(
                f"- {memory}"
                for memory in user_memories
            )
        )

    if style_examples:

        context_parts.append(
            "[실제 말투 예시]\n"
            + " / ".join(
                style_examples
            )
        )

    # Context
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

    # Context 확인
    contents.append(
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text="응 확인햇어"
                )
            ],
        )
    )

    # 최근 대화
    for (
        history_sender,
        text,
    ) in recent_history:

        role = (
            "model"
            if history_sender == "이태양"
            else "user"
        )

        contents.append(
            types.Content(
                role=role,
                parts=[
                    types.Part.from_text(
                        text=text
                    )
                ],
            )
        )

    # 현재 입력
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=user_input
                )
            ],
        )
    )

    # ---------------------------------------------------------
    # Gemini
    # ---------------------------------------------------------

    try:

        response = (
            client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        system_instruction
                    ),
                    thinking_config=(
                        types.ThinkingConfig(
                            thinking_level="low"
                        )
                    ),
                    max_output_tokens=100,
                ),
            )
        )

        if response.text:

            reply = (
                response.text
                .replace("\n", " ")
                .strip()
            )

        else:

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

    # ---------------------------------------------------------
    # 신규 DB 저장
    # ---------------------------------------------------------

    log_conversation(
        speaker_name="이태양",
        target_key="self",
        message=reply,
        room_id="dm",
    )

    return {
        "reply": reply
    }
