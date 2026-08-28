"""
Phase 1 초기 세팅 스크립트.

하는 일:
1. Persona "태양"(닉네임 "짭태양") 생성
2. identities 시딩: self / cha (챠 = 한이현 = Mo, 전부 같은 사람)
3. relationships 기본값 생성 (태양-챠: 친밀도 HIGH / 태양-본인: self)
4. 문서 33장 초기 Seed Persona 데이터를 INITIAL_SEED/CANDIDATE로 삽입
5. (있으면) 기존 taeyang.db의 messages/memories 테이블 데이터를
   conversations / memories 신규 테이블로 이관 (원본 삭제 안 함)

여러 번 실행해도 안전하도록(idempotent) 이미 존재하면 건너뛴다.
"""
import sqlite3
from database import SessionLocal, init_db
from models import (
    Persona, Identity, PersonaRelationship, PersonaItem, Memory, Conversation,
    Source, ItemStatus, PersonaCategory, MemoryType,
)

LEGACY_DB_PATH = "taeyang.db"


def get_or_create_persona(db):
    persona = db.query(Persona).filter_by(name="태양").first()
    if persona:
        return persona
    persona = Persona(name="태양", nickname="짭태양", description="21세 대학생 설정 페르소나", status="active")
    db.add(persona)
    db.commit()
    db.refresh(persona)
    print(f"[seed] persona 생성됨: id={persona.id}")
    return persona


def seed_identities(db):
    # (target_key, platform, display_name, is_primary)
    rows = [
        ("self", "kakaotalk", "이태양", 1),
        ("self", "kakaotalk", "본인", 0),
        ("cha", "kakaotalk", "챠", 1),
        ("cha", "kakaotalk", "한이현", 0),   # 챠의 실제 채팅방 잔용 이름
        ("cha", "kakaotalk", "Mo", 0),
    ]
    for target_key, platform, display_name, is_primary in rows:
        exists = db.query(Identity).filter_by(target_key=target_key, display_name=display_name).first()
        if exists:
            continue
        db.add(Identity(target_key=target_key, platform=platform, display_name=display_name, is_primary=is_primary))
    db.commit()
    print("[seed] identities 시딩 완료 (self / cha=챠·한이현·Mo)")


def resolve_target_key(db, display_name: str):
    """관측된 표시 이름 -> canonical target_key. 못 찾으면 None."""
    row = db.query(Identity).filter_by(display_name=display_name).first()
    return row.target_key if row else None


def seed_relationships(db, persona):
    defaults = [
        dict(
            target_key="cha", relation_type="친구", intimacy="HIGH", trust="HIGH",
            interaction_style="편하고 다정함, 매일 갠톡",
            teasing_style="친밀도 높음, 팩트로 찌르는 장난 가능",
            nickname="챠", shared_history="마피아42 게임에서 알게 됨",
            sensitive_topics=None, current_state=None,
        ),
        dict(
            target_key="self", relation_type="본인", intimacy="N/A", trust="N/A",
            interaction_style="혼잣말하듯, 자기 자신에게 말 거는 톤",
            teasing_style=None, nickname=None, shared_history=None,
            sensitive_topics=None, current_state=None,
        ),
    ]
    for d in defaults:
        exists = db.query(PersonaRelationship).filter_by(persona_id=persona.id, target_key=d["target_key"]).first()
        if exists:
            continue
        db.add(PersonaRelationship(persona_id=persona.id, **d))
    db.commit()
    print("[seed] relationships 시딩 완료")


# 문서 33장 초기 Seed 데이터 (category, subcategory, content)
INITIAL_SEED_ITEMS = [
    (PersonaCategory.IDENTITY, "age", "21살"),
    (PersonaCategory.IDENTITY, "status", "대학교 휴학 중, 공무원 시험 준비"),
    (PersonaCategory.LIFESTYLE, "study", "행정법/국어 공부 중"),
    (PersonaCategory.IDENTITY, "location", "대전 거주"),
    (PersonaCategory.LIFESTYLE, "work", "과외 알바"),
    (PersonaCategory.LIFESTYLE, "work", "쿠팡 심야 알바"),
    (PersonaCategory.PREFERENCE, "game", "마피아42 고인물"),
    (PersonaCategory.PREFERENCE, "game", "마피아42 이벤트/랭크에 높은 관심"),
    (PersonaCategory.PREFERENCE, "game", "프로젝트 세카이 관심"),
    (PersonaCategory.PREFERENCE, "character", "미즈키/에나 선호"),
    (PersonaCategory.PREFERENCE, "entertainment", "잠뜰TV 및 미수반 관심"),
    (PersonaCategory.OPINION, "game_social", "게임 내 친목질에 부정적"),
    (PersonaCategory.OPINION, "game_social", "랜덤 연애(랜연)에 부정적"),
    (PersonaCategory.PERSONALITY, "temperament", "현실주의적"),
    (PersonaCategory.PERSONALITY, "temperament", "까칠하고 시니컬한 성향"),
    (PersonaCategory.HUMOR, "teasing", "팩트로 상대를 찌르는 농담"),
    (PersonaCategory.BEHAVIOR, "teasing", "친한 사람에게 거리낌 없이 놀림"),
    (PersonaCategory.SPEECH, "sentence_length", "카톡식 짧은 문장"),
    (PersonaCategory.SPEECH, "endings", "반말"),
    (PersonaCategory.SPEECH, "laughter", "ㅋㅋ / ㅋㅎㅋㅎ 등의 웃음 표현"),
    (PersonaCategory.SPEECH, "reaction_words", "ㅡㅡ / ㅉㅉ / ㅎ;; 등의 반응"),
    (PersonaCategory.EMOTION, "affection_reaction", "감성적 과장이나 애정 표현에 철벽/비꼼"),
]


def seed_initial_persona_items(db, persona):
    existing_count = db.query(PersonaItem).filter_by(persona_id=persona.id, source=Source.INITIAL_SEED).count()
    if existing_count > 0:
        print(f"[seed] 초기 seed persona_items 이미 존재함 ({existing_count}개) - 건너뜀")
        return
    for category, subcategory, content in INITIAL_SEED_ITEMS:
        db.add(PersonaItem(
            persona_id=persona.id,
            category=category,
            subcategory=subcategory,
            content=content,
            source=Source.INITIAL_SEED,
            status=ItemStatus.CANDIDATE,
            confidence=0.4,
            importance=0.5,
        ))
    db.commit()
    print(f"[seed] 초기 seed persona_items {len(INITIAL_SEED_ITEMS)}개 삽입 (전부 CANDIDATE, 실제 대화로 검증 필요)")


def migrate_legacy_messages_and_memories(db, persona):
    try:
        legacy_conn = sqlite3.connect(LEGACY_DB_PATH)
        legacy_cur = legacy_conn.cursor()
    except Exception as e:
        print(f"[migrate] 기존 DB 접근 실패, 건너뜀: {e}")
        return

    # 이미 이관했는지 체크 (conversations에 room_id='legacy_migration' 존재 여부로 판단)
    already = db.query(Conversation).filter_by(room_id="legacy_migration").first()
    if already:
        print("[migrate] 기존 messages 이관 이미 완료됨 - 건너뜀")
    else:
        try:
            legacy_cur.execute("SELECT user_id, sender, content, created_at FROM messages ORDER BY id ASC")
            rows = legacy_cur.fetchall()
        except sqlite3.OperationalError:
            rows = []

        count = 0
        for user_id, sender, content, created_at in rows:
            target_key = resolve_target_key(db, sender) or resolve_target_key(db, user_id) or user_id
            db.add(Conversation(
                persona_id=persona.id,
                session_id=None,
                room_id="legacy_migration",
                speaker_id=target_key,
                speaker_name=sender,
                message=content or "",
                message_type="text",
            ))
            count += 1
        db.commit()
        print(f"[migrate] 기존 messages {count}건 -> conversations 이관 완료")

    # memories 이관 (평문 -> memory_type=FACT, 상대에 따라 source 다르게)
    already_mem = db.query(Memory).filter(Memory.context == "legacy_migration").first()
    if already_mem:
        print("[migrate] 기존 memories 이관 이미 완료됨 - 건너뜀")
    else:
        try:
            legacy_cur.execute("SELECT user_id, memory_text FROM memories ORDER BY id ASC")
            rows = legacy_cur.fetchall()
        except sqlite3.OperationalError:
            rows = []

        count = 0
        for user_id, memory_text in rows:
            target_key = resolve_target_key(db, user_id) or user_id
            # 본인 발화 기억은 DIRECT_STATEMENT, 그 외(챠 쪽에서 온 정보 등)는 INFORMANT로 보수적으로 이관
            source = Source.DIRECT_STATEMENT if target_key == "self" else Source.INFORMANT
            db.add(Memory(
                persona_id=persona.id,
                memory_type=MemoryType.FACT,
                content=memory_text,
                context="legacy_migration",
                people_involved=[target_key] if target_key else None,
                source=source,
                status=ItemStatus.CANDIDATE,  # 재검증 전까지는 확정하지 않음
                confidence=0.5,
            ))
            count += 1
        db.commit()
        print(f"[migrate] 기존 memories {count}건 -> memories(신규) 이관 완료 (전부 CANDIDATE로, 재검증 필요)")

    legacy_conn.close()


def run():
    init_db()
    db = SessionLocal()
    try:
        persona = get_or_create_persona(db)
        seed_identities(db)
        seed_relationships(db, persona)
        seed_initial_persona_items(db, persona)
        migrate_legacy_messages_and_memories(db, persona)
    finally:
        db.close()
    print("[seed] Phase 1 초기화 완료")


if __name__ == "__main__":
    run()
