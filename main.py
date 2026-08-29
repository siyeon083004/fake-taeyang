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

# 본인(self)과 대화할 때의 프롬프트 (관계성 재정립)
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
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.IGNORECASE).strip()
    try: data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match: return None
        try: data = json.loads(match.group(0))
        except Exception: return None

    if not isinstance(data, dict): return None
    save_value = data.get("save", False)
    if isinstance(save_value, str):
        save_value = (save_value.strip().lower() in ["true", "yes", "y", "1", "저장", "예"])
    else: save_value = bool(save_value)

    if not save_value: return {"save": False}

    category = normalize_memory_category(data.get("category"))
    memory = str(data.get("memory", "")).strip()[:300].rstrip()
    people_involved = data.get("people_involved", [])
    if not isinstance(people_involved, list): people_involved = []

    if not memory: return {"save": False}
    return {"save": True, "category": category, "memory": memory, "people_involved": people_involved}

def judge_long_term_memory(user_input, recent_history=None, existing_memories=None):
    if not user_input or len(str(user_input).strip()) < 4 or is_command(user_input): return None

    history_text = "\n".join([f"{s}: {t}" for s, t in recent_history[-6:] if not is_command(t)]) if recent_history else ""
    memory_text = "\n".join([f"- {m}" for m in existing_memories[-30:]]) if existing_memories else ""

    prompt = f"""
너는 장기기억 선별기다.
사용자의 현재 발화를 보고 앞으로도 이 사람을 이해하는 데 도움이 될 만한 안정적인 정보만 장기기억으로 저장해라.

[중요 변경점: 관련 인물 태깅]
이 기억이 누구와 관련된 정보인지 파악해서 'people_involved' 리스트에 담아라.
예) 본인이 "만세 매운거 환장함"이라고 했다면 -> ["만세"]
예) 만세가 직접 "나 오이 싫어"라고 했다면 -> ["만세"]
예) 본인이 "나는 야구가 좋아"라고 했다면 -> ["self"]

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
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                max_output_tokens=300, temperature=0.1, response_mime_type="application/json",
            ),
        )
        return parse_memory_judgement(response.text if response else "")
    except Exception as e:
        print(f"[memory judge] 실패: {repr(e)}")
        return None


def save_auto_memory_if_worthy(conversation_key, user_input, is_self, recent_history=None, existing_memories=None):
    result = judge_long_term_memory(user_input, recent_history, existing_memories)
    if not result or not result.get("save"): return None

    category = result.get("category", "기타")
    memory = result.get("memory", "")
    people = result.get("people_involved", [])

    db = SessionLocal()
    try:
        # 중복 체크 (간단히)
        existing = db.query(Memory).filter(Memory.persona_id == get_persona_id()).all()
        normalized_new = memory.lower()
        for old in existing:
            if normalized_new in old.content.lower():
                return None

        # 새 공용 뇌 모델에 바로 저장
        new_memory = Memory(
            persona_id=get_persona_id(),
            memory_type=MemoryType.FACT,
            content=memory,
            context=category,
            people_involved=people,
            # 본인이 말한거면 확실(CONFIRMED/DIRECT_STATEMENT), 남이 말한거면 CANDIDATE/INFORMANT
            source=Source.DIRECT_STATEMENT if is_self else Source.INFORMANT,
            status=ItemStatus.CONFIRMED if is_self else ItemStatus.CANDIDATE
        )
        db.add(new_memory)
        db.commit()
        print(f"[memory] 공용 뇌 자동 저장: {category} - {memory} (관련인물: {people})")
        return {"category": category, "memory": memory}
    except Exception as e:
        print(f"[memory save] 실패: {repr(e)}")
    finally:
        db.close()
    return None


# ---------------------------------------------------------------------------
# 백그라운드 학습
# ---------------------------------------------------------------------------
def background_learning(conversation_key, user_input, recent_history, existing_memories, is_self):
    # 1. 말투 학습 (본인 것만)
    if is_self and should_learn_style(user_input):
        try: legacy.save_style_sample(user_input)
        except: pass

    # 2. 장기기억 학습 (모든 대화 상대에게서 수집!)
    try:
        save_auto_memory_if_worthy(
            conversation_key=conversation_key,
            user_input=user_input,
            is_self=is_self,
            recent_history=recent_history,
            existing_memories=existing_memories,
        )
    except Exception as e:
        print(f"[auto memory] 전체 실패: {repr(e)}")


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    sender: str
    message: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "ok", "model": MODEL_NAME}


# ---------------------------------------------------------------------------
# Chat Main Logic
# ---------------------------------------------------------------------------
@app.post("/chat")
def reply_chat(req: ChatRequest, background_tasks: BackgroundTasks):
    raw_input = str(req.message).strip()
    if not raw_input: return {"reply": "뭐라고"}

    user_input = raw_input.replace("@짭태양", "").replace("/짭태양", "").strip()
    if not user_input: return {"reply": "ㅇㅇ"}

    # Person 확인
    db = SessionLocal()
    try:
        person, is_new_person = get_or_create_observed_person(db, req.sender)
        target_key = person.person_key
    except Exception as e:
        print(f"[person] 처리 실패: {repr(e)}")
        return {"reply": "사람 연결하는데 오류남;;"}
    finally:
        db.close()

    is_self = (target_key == "self")
    conversation_key = target_key

    # 리셋 처리
    if user_input in ["/리셋", "/초기화"]:
        try:
            conn = sqlite3.connect("taeyang.db")
            cur = conn.cursor()
            cur.execute("DELETE FROM messages WHERE user_id = ?", (conversation_key,))
            conn.commit()
            conn.close()
            return {"reply": "대화기록초기화완료"}
        except:
            return {"reply": "초기화하다 오류남;;"}

    # 수동 명령어 (기존 핸들러)
    db = SessionLocal()
    try:
        # /이름, /인물 관련 로직
        if user_input in ["/이름목록", "/이름 목록", "/인물목록", "/인물 목록"]: return {"reply": command_name_list(db)}
        if user_input.startswith("/이름삭제 "): return {"reply": command_name_delete(db, user_input[6:].strip())}
        if user_input.startswith("/이름 "): return {"reply": command_name(db, req.sender, user_input[4:].strip())}
        if user_input.startswith("/인물삭제 "): return {"reply": command_person_delete(db, user_input[6:].strip())}
        if user_input.startswith("/인물병합 "):
            args = user_input[6:].split()
            if len(args) >= 2: return {"reply": command_person_merge(db, args[0], args[1])}
        if user_input.startswith("/인물 "):
            args = user_input[4:].split()
            if args: return {"reply": command_person(db, args[0], args[1:])}
    finally:
        db.close()

    log_conversation(speaker_name=req.sender, target_key=target_key, message=req.message, room_id="dm")

    # ---------------------------------------------------------
    # 공용 뇌 명령어 (조회/삭제/수동저장)
    # ---------------------------------------------------------
    if user_input in ["/기억목록", "/기억 목록", "/기억리스트"]:
        db = SessionLocal()
        try:
            memories = db.query(Memory).filter(Memory.persona_id == get_persona_id()).all()
            if not memories: return {"reply": "기억된 정보가 없어"}
            lines = ["[장기기억 (공용 뇌)]"]
            for m in memories:
                involved = ", ".join(m.people_involved) if m.people_involved else "불명"
                lines.append(f"[{m.id}] [대상:{involved}] {m.content}")
            return {"reply": "\n".join(lines)}
        finally: db.close()

    if user_input.startswith("/기억삭제"):
        target = user_input.replace("/기억삭제", "", 1).strip()
        if target.isdigit():
            db = SessionLocal()
            try:
                mem = db.query(Memory).filter(Memory.id == int(target)).first()
                if mem:
                    db.delete(mem)
                    db.commit()
                    return {"reply": f"기억삭제완료: [{target}]번"}
                return {"reply": f"[{target}]번 기억을 찾을 수 없어"}
            finally: db.close()
        return {"reply": "기억 번호를 적어줘"}

    if user_input.startswith("/기억 "):
        mem_text = user_input[4:].strip()
        if mem_text:
            db = SessionLocal()
            try:
                new_memory = Memory(
                    persona_id=get_persona_id(),
                    memory_type=MemoryType.FACT,
                    content=mem_text,
                    context="기타",
                    people_involved=[target_key],
                    source=Source.DIRECT_STATEMENT if is_self else Source.INFORMANT,
                    status=ItemStatus.CONFIRMED if is_self else ItemStatus.CANDIDATE
                )
                db.add(new_memory)
                db.commit()
                return {"reply": f"응기억햇어: {mem_text}"}
            finally: db.close()
        return {"reply": "기억할 내용을 적어줘"}

    if user_input.startswith("/말투 "):
        style_text = user_input[4:].strip()
        if style_text and should_learn_style(style_text):
            try: legacy.save_style_sample(style_text)
            except: pass
            return {"reply": f"응 이것도 배웟어: {style_text}"}
        return {"reply": "배울 말투를 적어줘"}

    # ---------------------------------------------------------
    # Context 준비
    # ---------------------------------------------------------
    now_kst = datetime.now(KST).strftime("%Y년 %m월 %d일 %H시 %M분")

    try: recent_history = legacy.get_recent_messages(conversation_key, limit=8)
    except: recent_history = []

    # 공용 뇌에서 기억 꺼내오기 (현재 상대 + 멘션된 사람)
    try:
        mentioned_keywords = [word for word in user_input.split() if len(word) >= 2]
        db_mem = SessionLocal()
        all_memories = db_mem.query(Memory).filter(Memory.persona_id == get_persona_id()).all()
        
        user_memories = []
        for mem in all_memories:
            involved = mem.people_involved or []
            is_relevant = (target_key in involved)
            if not is_relevant:
                for keyword in mentioned_keywords:
                    if any(keyword in str(person) for person in involved):
                        is_relevant = True
                        break
            if is_relevant:
                source_kr = "본인(너)이 주입함" if mem.source == Source.DIRECT_STATEMENT else "타인과의 대화에서 얻음"
                involved_str = ", ".join(involved) if involved else "불명"
                user_memories.append(f"[대상: {involved_str}] {mem.content} (출처: {source_kr})")
        db_mem.close()
    except Exception as e:
        print(f"[memory load error] {repr(e)}")
        user_memories = []

    try:
        style_examples = [str(e).strip() for e in legacy.get_relevant_style_samples(user_input, n=12) if should_learn_style(str(e).strip())]
    except:
        style_examples = []

    if is_self: system_instruction = SYSTEM_INSTRUCTION_FOR_SELF
    elif target_key == "cha": system_instruction = SYSTEM_INSTRUCTION_FOR_CHA
    else: system_instruction = SYSTEM_INSTRUCTION_FOR_UNKNOWN

    db = SessionLocal()
    person = None
    people_lines = []
    try:
        person = get_person_by_key(db, target_key)
        known_people = db.query(Person).filter_by(status="active").order_by(Person.id.asc()).limit(100).all()
        for known_person in known_people:
            aliases = db.query(PersonAlias).filter_by(person_id=known_person.id).order_by(PersonAlias.id.asc()).all()
            alias_text = ", ".join(alias.alias for alias in aliases)
            people_lines.append(f"{known_person.person_key}: {known_person.canonical_name} ({alias_text})")
    finally:
        db.close()

    # ---------------------------------------------------------
    # Gemini Context 조립
    # ---------------------------------------------------------
    contents = []
    context_parts = [f"[현재 한국 시각]: {now_kst}"]
    if person: context_parts.append(f"[현재 상대]: {person.canonical_name} / {target_key}")
    if people_lines: context_parts.append("[알고 있는 인물 목록]\n" + "\n".join(people_lines))
    if user_memories: context_parts.append("[대화 관련 장기기억 (공용 뇌)]\n" + "\n".join(f"- {m}" for m in user_memories))
    if style_examples: context_parts.append("[실제 말투 예시]\n" + "\n".join(f"- {e}" for e in style_examples))

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n\n".join(context_parts))]))
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="응 확인햇어")]))

    for (history_sender, text) in recent_history:
        text = str(text).strip()
        if not text or is_command(text): continue
        role = "model" if history_sender == "이태양" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

    # ---------------------------------------------------------
    # 생성 및 정리
    # ---------------------------------------------------------
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
                max_output_tokens=1200,
            ),
        )
        reply = str(response.text or "").replace("\r", " ").strip()
        reply = re.sub(r"^```(?:text)?\s*|\s*```$", "", reply, flags=re.IGNORECASE).strip()
        if reply.startswith("```"):
            reply = re.sub(r"^```.*?\n|\n```$", "", reply, flags=re.DOTALL).strip()
        if not reply: reply = "어왜ㅋ"
    except Exception as e:
        print(f"[Gemini ERROR] {repr(e)}")
        reply = "서버에서 모델응답 오류남;;"

    # ---------------------------------------------------------
    # Legacy 저장 및 큐 등록
    # ---------------------------------------------------------
    try:
        legacy.save_message(conversation_key, conversation_key, user_input)
        legacy.save_message(conversation_key, "이태양", reply)
    except: pass
    log_conversation(speaker_name="이태양", target_key="self", message=reply, room_id="dm")

    background_tasks.add_task(
        background_learning,
        conversation_key,
        user_input,
        recent_history,
        user_memories,
        is_self,
    )

    return {"reply": reply}
