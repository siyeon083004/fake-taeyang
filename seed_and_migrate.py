"""
장산범 Persona Engine v2 초기화.

- self는 내부적으로 person_key="self" 사용
- 실제 카카오톡 이름은 /이름 명령으로 연결
- 챠 등의 기존 인물은 별도로 시딩
- 기존 legacy DB는 삭제하지 않고 신규 DB로 복사
"""

import sqlite3

from database import SessionLocal, init_db

from models import (
    Persona,
    Person,
    PersonAlias,
    Identity,
    PersonaRelationship,
    PersonaItem,
    Memory,
    Conversation,
    Source,
    ItemStatus,
    PersonaCategory,
    MemoryType,
)


LEGACY_DB_PATH = "taeyang.db"


# ---------------------------------------------------------------------------
# Persona
# ---------------------------------------------------------------------------

def get_or_create_persona(db):

    persona = (
        db.query(Persona)
        .filter_by(name="태양")
        .first()
    )

    if persona:
        return persona

    persona = Persona(
        name="태양",
        nickname="짭태양",
        description="실제 인물의 말투와 기억을 학습하는 Persona",
        status="active",
    )

    db.add(persona)
    db.commit()
    db.refresh(persona)

    print(
        f"[seed] persona 생성: id={persona.id}"
    )

    return persona


# ---------------------------------------------------------------------------
# Person
# ---------------------------------------------------------------------------

def get_or_create_person(
    db,
    person_key,
    canonical_name,
    person_type="person",
    confirmed=False,
    observed=False,
):
    person = (
        db.query(Person)
        .filter_by(
            person_key=person_key
        )
        .first()
    )

    if person:

        changed = False

        if (
            canonical_name
            and person.canonical_name != canonical_name
        ):
            person.canonical_name = canonical_name
            changed = True

        if confirmed and not person.confirmed:
            person.confirmed = 1
            changed = True

        if observed and not person.observed_in_chat:
            person.observed_in_chat = 1
            changed = True

        if changed:
            db.commit()

        return person

    person = Person(
        person_key=person_key,
        canonical_name=canonical_name,
        person_type=person_type,
        status="active",
        confirmed=1 if confirmed else 0,
        observed_in_chat=1 if observed else 0,
    )

    db.add(person)
    db.commit()
    db.refresh(person)

    return person


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------

def add_alias(
    db,
    person,
    alias,
    source=Source.DIRECT_STATEMENT,
):
    alias = alias.strip()

    if not alias:
        return False

    existing = (
        db.query(PersonAlias)
        .filter_by(alias=alias)
        .first()
    )

    if existing:

        if existing.person_id == person.id:
            return True

        return False

    db.add(
        PersonAlias(
            person_id=person.id,
            alias=alias,
            source=source,
            confidence=1.0,
        )
    )

    db.commit()

    return True


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def add_identity(
    db,
    person,
    display_name,
    target_key=None,
):
    display_name = display_name.strip()

    if not display_name:
        return False

    existing = (
        db.query(Identity)
        .filter_by(
            display_name=display_name,
            platform="kakaotalk",
        )
        .first()
    )

    if existing:

        if existing.person_id != person.id:
            existing.person_id = person.id
            existing.target_key = (
                target_key
                or person.person_key
            )
            existing.is_primary = 1

            db.commit()

        return True

    db.add(
        Identity(
            person_id=person.id,
            target_key=(
                target_key
                or person.person_key
            ),
            platform="kakaotalk",
            display_name=display_name,
            is_primary=1,
        )
    )

    db.commit()

    return True


# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------

def seed_people(db):

    # 본인
    #
    # 실제 카카오톡 이름은 하드코딩하지 않는다.
    # /이름 이태양 명령을 통해 self로 연결된다.
    self_person = get_or_create_person(
        db,
        person_key="self",
        canonical_name="본인",
        person_type="self",
        confirmed=True,
        observed=False,
    )

    # 챠
    cha_person = get_or_create_person(
        db,
        person_key="cha",
        canonical_name="챠",
        person_type="person",
        confirmed=True,
        observed=False,
    )

    # 챠 별칭
    add_alias(
        db,
        cha_person,
        "챠",
    )

    add_alias(
        db,
        cha_person,
        "한이현",
    )

    add_alias(
        db,
        cha_person,
        "Mo",
    )

    return (
        self_person,
        cha_person,
    )


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------

def seed_identities(
    db,
    self_person,
    cha_person,
):
    """
    실제 사용자 이름은 여기서 등록하지 않는다.
    """

    add_identity(
        db,
        cha_person,
        "챠",
        "cha",
    )

    add_identity(
        db,
        cha_person,
        "한이현",
        "cha",
    )

    add_identity(
        db,
        cha_person,
        "Mo",
        "cha",
    )


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

def seed_relationships(
    db,
    persona,
):
    defaults = [

        dict(
            target_key="cha",
            relation_type="친구",
            intimacy="HIGH",
            trust="HIGH",
            interaction_style="편하고 다정한 게임 친구",
            teasing_style="친밀도 높음, 팩트로 찌르는 장난 가능",
            nickname="챠",
            shared_history="마피아42 게임에서 알게 됨",
        ),

        dict(
            target_key="self",
            relation_type="본인",
            intimacy="N/A",
            trust="N/A",
            interaction_style="본인과 대화하는 학습 관계",
            teasing_style=None,
            nickname=None,
            shared_history=None,
        ),
    ]

    for data in defaults:

        existing = (
            db.query(PersonaRelationship)
            .filter_by(
                persona_id=persona.id,
                target_key=data["target_key"],
            )
            .first()
        )

        if existing:
            continue

        db.add(
            PersonaRelationship(
                persona_id=persona.id,
                **data,
            )
        )

    db.commit()


# ---------------------------------------------------------------------------
# Initial Persona Items
# ---------------------------------------------------------------------------

INITIAL_SEED_ITEMS = [

    (
        PersonaCategory.IDENTITY,
        "age",
        "21살",
    ),

    (
        PersonaCategory.IDENTITY,
        "status",
        "대학교 휴학 중, 공무원 시험 준비",
    ),

    (
        PersonaCategory.LIFESTYLE,
        "study",
        "행정법/국어 공부 중",
    ),

    (
        PersonaCategory.LIFESTYLE,
        "work",
        "과외 알바",
    ),

    (
        PersonaCategory.LIFESTYLE,
        "work",
        "쿠팡 심야 알바",
    ),

    (
        PersonaCategory.IDENTITY,
        "location",
        "대전 거주",
    ),

    (
        PersonaCategory.PREFERENCE,
        "game",
        "마피아42 고인물",
    ),

    (
        PersonaCategory.PREFERENCE,
        "game",
        "마피아42 이벤트/랭크에 높은 관심",
    ),

    (
        PersonaCategory.PREFERENCE,
        "game",
        "프로젝트 세카이 관심",
    ),

    (
        PersonaCategory.PREFERENCE,
        "character",
        "미즈키/에나 선호",
    ),

    (
        PersonaCategory.PREFERENCE,
        "entertainment",
        "잠뜰TV 및 미수반 관심",
    ),

    (
        PersonaCategory.OPINION,
        "game_social",
        "게임 내 친목질에 부정적",
    ),

    (
        PersonaCategory.OPINION,
        "game_social",
        "랜덤 연애(랜연)에 부정적",
    ),

    (
        PersonaCategory.PERSONALITY,
        "temperament",
        "현실주의적",
    ),

    (
        PersonaCategory.PERSONALITY,
        "temperament",
        "까칠하고 시니컬한 성향",
    ),

    (
        PersonaCategory.HUMOR,
        "teasing",
        "팩트로 상대를 찌르는 농담",
    ),

    (
        PersonaCategory.BEHAVIOR,
        "teasing",
        "친한 사람에게 거리낌 없이 놀림",
    ),

    (
        PersonaCategory.SPEECH,
        "sentence_length",
        "카톡식 짧은 문장",
    ),

    (
        PersonaCategory.SPEECH,
        "endings",
        "반말",
    ),

    (
        PersonaCategory.SPEECH,
        "laughter",
        "ㅋㅋ / ㅋㅎㅋㅎ 등의 웃음 표현",
    ),

    (
        PersonaCategory.SPEECH,
        "reaction_words",
        "ㅡㅡ / ㅉㅉ / ㅎ;; 등의 반응",
    ),

    (
        PersonaCategory.EMOTION,
        "affection_reaction",
        "감성적 과장이나 애정 표현에 철벽/비꼼",
    ),
]


def seed_initial_persona_items(
    db,
    persona,
):
    existing = (
        db.query(PersonaItem)
        .filter_by(
            persona_id=persona.id,
            source=Source.INITIAL_SEED,
        )
        .count()
    )

    if existing:
        print(
            f"[seed] 초기 persona item 존재: {existing}개"
        )
        return

    for (
        category,
        subcategory,
        content,
    ) in INITIAL_SEED_ITEMS:

        db.add(
            PersonaItem(
                persona_id=persona.id,
                category=category,
                subcategory=subcategory,
                content=content,
                source=Source.INITIAL_SEED,
                status=ItemStatus.CANDIDATE,
                confidence=0.4,
                importance=0.5,
            )
        )

    db.commit()

    print(
        f"[seed] 초기 persona item "
        f"{len(INITIAL_SEED_ITEMS)}개 생성"
    )


# ---------------------------------------------------------------------------
# Legacy Migration
# ---------------------------------------------------------------------------

def migrate_legacy_messages_and_memories(
    db,
    persona,
):
    """
    기존 taeyang.db 데이터를 신규 DB로 복사한다.

    원본 legacy 데이터는 삭제하지 않는다.
    """

    try:

        conn = sqlite3.connect(
            LEGACY_DB_PATH
        )

        cur = conn.cursor()

    except Exception as e:

        print(
            f"[migrate] legacy DB 접근 실패: {e}"
        )

        return

    # ---------------------------------------------------------
    # Messages
    # ---------------------------------------------------------

    already = (
        db.query(Conversation)
        .filter_by(
            room_id="legacy_migration"
        )
        .first()
    )

    if not already:

        try:

            cur.execute(
                """
                SELECT
                    user_id,
                    sender,
                    content,
                    created_at
                FROM messages
                ORDER BY id ASC
                """
            )

            rows = cur.fetchall()

        except sqlite3.OperationalError:

            rows = []

        count = 0

        for (
            user_id,
            sender,
            content,
            created_at,
        ) in rows:

            target = (
                user_id
                or sender
                or "unknown"
            )

            db.add(
                Conversation(
                    persona_id=persona.id,
                    room_id="legacy_migration",
                    speaker_id=target,
                    speaker_name=sender,
                    message=content or "",
                    message_type="text",
                )
            )

            count += 1

        db.commit()

        print(
            f"[migrate] legacy messages "
            f"{count}건 복사 완료"
        )

    # ---------------------------------------------------------
    # Memories
    # ---------------------------------------------------------

    already_memory = (
        db.query(Memory)
        .filter_by(
            context="legacy_migration"
        )
        .first()
    )

    if not already_memory:

        try:

            cur.execute(
                """
                SELECT
                    user_id,
                    memory_text
                FROM memories
                ORDER BY id ASC
                """
            )

            rows = cur.fetchall()

        except sqlite3.OperationalError:

            rows = []

        count = 0

        for (
            user_id,
            memory_text,
        ) in rows:

            target = (
                user_id
                or "unknown"
            )

            db.add(
                Memory(
                    persona_id=persona.id,
                    memory_type=MemoryType.FACT,
                    content=memory_text,
                    context="legacy_migration",
                    people_involved=[target],
                    source=Source.INFORMANT,
                    status=ItemStatus.CANDIDATE,
                    confidence=0.5,
                )
            )

            count += 1

        db.commit()

        print(
            f"[migrate] legacy memories "
            f"{count}건 복사 완료"
        )

    conn.close()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run():

    init_db()

    db = SessionLocal()

    try:

        persona = get_or_create_persona(db)

        (
            self_person,
            cha_person,
        ) = seed_people(db)

        seed_identities(
            db,
            self_person,
            cha_person,
        )

        seed_relationships(
            db,
            persona,
        )

        seed_initial_persona_items(
            db,
            persona,
        )

        migrate_legacy_messages_and_memories(
            db,
            persona,
        )

    finally:

        db.close()

    print(
        "[seed] Persona Engine v2 초기화 완료"
    )


if __name__ == "__main__":
    run()
