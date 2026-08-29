import json
import re
from google.genai import types
from config import client, MODEL_NAME
from models import Memory, MemoryType, ItemStatus, Source
from services.person_service import (
    get_person_by_key,
    get_person_by_alias,
    extract_mentioned_people,
)

COMMAND_PREFIXES = ("/이름", "/인물", "/기억", "/말투", "/리셋", "/초기화")

def is_command(text):
    text = str(text or "").strip()
    return bool(text and text.startswith(COMMAND_PREFIXES))

def should_learn_style(text):
    if not text:
        return False
    text = str(text).strip()
    if len(text) < 2 or is_command(text) or "@짭태양" in text or "/짭태양" in text:
        return False
    skip_patterns = [
        r"^ㅋㅋ+$", r"^ㅋ+$", r"^ㅎㅎ+$", r"^ㅎ+$", r"^ㅇㅇ+$", r"^ㄴㄴ+$", r"^ㅇㅋ+$",
        r"^오+$", r"^아+$", r"^헐+$", r"^뭐함\??$", r"^뭐해\??$", r"^자냐\??$",
    ]
    for p in skip_patterns:
        if re.fullmatch(p, text, flags=re.IGNORECASE):
            return False
    return True

def normalize_memory_category(category):
    category = str(category or "").strip().lower()
    mapping = {
        "preference": "취향", "preferences": "취향",
        "person": "사람", "people": "사람",
        "relationship": "관계", "relationships": "관계",
        "fact": "사실", "facts": "사실",
        "style": "대화스타일", "conversation_style": "대화스타일",
        "behavior": "습관", "habit": "습관",
    }
    allowed = ["취향", "사람", "관계", "사실", "대화스타일", "습관", "기타"]
    return category if category in allowed else mapping.get(category, "기타")

def parse_memory_judgement(text):
    if not text:
        return None
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except Exception:
            return None

    if not isinstance(data, dict) or not data.get("save"):
        return {"save": False}

    category = normalize_memory_category(data.get("category"))
    memory = str(data.get("memory", "")).strip()[:300].rstrip()
    people = [str(p).strip() for p in data.get("people_involved", []) if str(p).strip()]
    cleaned_people = list(dict.fromkeys(people))

    if not memory:
        return {"save": False}
    return {"save": True, "category": category, "memory": memory, "people_involved": cleaned_people}

def judge_long_term_memory(speaker_name, speaker_key, user_input, recent_history=None, existing_memories=None, known_people=None):
    if not user_input or len(str(user_input).strip()) < 4 or is_command(user_input):
        return None

    history_text = "\n".join([f"{s}: {t}" for _, s, t in recent_history[-8:] if not is_command(t)]) if recent_history else ""
    memory_text = "\n".join([f"- {m}" for m in existing_memories[-50:]]) if existing_memories else ""
    people_text = "\n".join([f"- {p}" for p in known_people]) if known_people else ""

    prompt = f"""
너는 장기기억 선별기다.
현재 발화를 보고 앞으로도 이 사람을 이해하는 데 도움이 되는 안정적인 정보만 장기기억으로 저장한다.

현재 발화자:
- 이름: {speaker_name}
- Person key: {speaker_key}

[원칙]
1. 자기 자신의 취향/습관/사실 직접 발화 -> 확정 저장
2. 제3자에 대한 발화 -> 미확인 정보
3. 농담/과장/비유/밈/순간적 감정/욕설 -> 저장 안 함
4. people_involved에는 실제 인물의 Person key 사용.

[등록 인물]
{people_text or "(없음)"}

[현재 발화]
{speaker_name}: {user_input}

[최근 대화]
{history_text or "(없음)"}

[기존 장기기억]
{memory_text or "(없음)"}

반드시 JSON 하나만 반환한다.
{{
  "save": true,
  "category": "취향",
  "memory": "떡볶이를 좋아함",
  "people_involved": ["{speaker_key}"]
}}
"""
    try:
        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                max_output_tokens=300,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        return parse_memory_judgement(res.text if res else "")
    except Exception as e:
        print(f"[memory judge] 실패: {repr(e)}")
        return None

def save_auto_memory_if_worthy(db, persona_id, speaker_name, speaker_key, user_input, is_self, recent_history, existing_memories, known_people):
    result = judge_long_term_memory(speaker_name, speaker_key, user_input, recent_history, existing_memories, known_people)
    if not result or not result.get("save"):
        return None

    category = result.get("category", "기타")
    memory = result.get("memory", "")
    people = result.get("people_involved", []) or [speaker_key]

    normalized_people = []
    for p_val in people:
        p_obj = get_person_by_key(db, p_val) or get_person_by_alias(db, p_val)
        normalized_people.append(p_obj.person_key if p_obj else p_val)
    people = list(dict.fromkeys(normalized_people))

    existing = db.query(Memory).filter(Memory.persona_id == persona_id).all()
    norm_new = str(memory).strip().lower()
    for old in existing:
        if set(map(str, people)) == set(map(str, old.people_involved or [])):
            old_t = str(old.content or "").strip().lower()
            if norm_new == old_t or norm_new in old_t or old_t in norm_new:
                return None

    status = ItemStatus.CONFIRMED if speaker_key in people else ItemStatus.CANDIDATE
    source = Source.DIRECT_STATEMENT if is_self else Source.INFORMANT

    new_memory = Memory(
        persona_id=persona_id,
        memory_type=MemoryType.FACT,
        content=memory,
        context=category,
        people_involved=people,
        source=source,
        status=status,
    )
    db.add(new_memory)
    db.commit()
    print(f"[memory] 저장: {category} - {memory} (관련인물={people})")
    return {"category": category, "memory": memory, "people_involved": people}

def get_memory_context_for_query(db, persona_id, target_key, user_input):
    mentioned_people = extract_mentioned_people(db, user_input)
    relevant_keys = {target_key} | set(mentioned_people.keys())
    all_memories = db.query(Memory).filter(Memory.persona_id == persona_id).order_by(Memory.id.desc()).all()

    selected = []
    keywords = [w for w in re.split(r"\s+", user_input) if len(w) >= 2]

    for mem in all_memories:
        involved = mem.people_involved or []
        if any(k in involved for k in relevant_keys):
            selected.append(mem)
            continue
        if any(kw.lower() in str(mem.content or "").lower() for kw in keywords):
            selected.append(mem)

    result, seen = [], set()
    for mem in selected:
        if mem.id not in seen:
            seen.add(mem.id)
            result.append(mem)
            if len(result) >= 80:
                break
    return result, mentioned_people

