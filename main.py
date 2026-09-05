import sqlite3
import re
import base64
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from google.genai import types

import legacy_store as legacy
import seed_and_migrate
from database import SessionLocal, init_db
from models import Persona, Person, Conversation, Memory, Source, ItemStatus
from config import client, MODEL_NAME, KST
from services.person_service import get_or_create_observed_person, get_person_by_key
from services.memory_service import is_command, should_learn_style, get_memory_context_for_query, save_auto_memory_if_worthy
from services.command_service import handle_identity_commands

init_db()
try: seed_and_migrate.run()
except Exception as e: print(f"[seed] 실패: {repr(e)}")

try: legacy.import_style_samples("style_samples.txt")
except Exception as e: print(f"[legacy] 로드 실패: {repr(e)}")

app = FastAPI()
_PERSONA_ID_CACHE = {"id": None}

def get_persona_id():
    if _PERSONA_ID_CACHE["id"] is None:
        db = SessionLocal()
        try:
            p = db.query(Persona).filter_by(name="태양").first()
            if not p: raise RuntimeError("태양 Persona가 없습니다.")
            _PERSONA_ID_CACHE["id"] = p.id
        finally: db.close()
    return _PERSONA_ID_CACHE["id"]

def log_conversation(speaker_name, target_key, message, room_id="dm"):
    db = SessionLocal()
    try:
        db.add(Conversation(
            persona_id=get_persona_id(), session_id=None, room_id=room_id,
            speaker_id=target_key, speaker_name=speaker_name, message=message, message_type="text"
        ))
        db.commit()
    except Exception as e: print(f"[log_conversation] 실패: {repr(e)}")
    finally: db.close()

def background_learning(conversation_key, speaker_name, speaker_key, user_input, recent_history, existing_memories, is_self, known_people):
    if is_self and should_learn_style(user_input):
        try: legacy.save_style_sample(user_input)
        except Exception: pass
    db = SessionLocal()
    try:
        save_auto_memory_if_worthy(db, get_persona_id(), speaker_name, speaker_key, user_input, is_self, recent_history, existing_memories, known_people)
    except Exception as e: print(f"[auto memory] 실패: {repr(e)}")
    finally: db.close()

STYLE_RULES = """
[문장 형식] 카톡처럼 자연스럽게, 너무 길지 않게, 반말 위주, 문장부호 최소화.
[금지] 비서 말투, AI/모델 언급, 슬래시 명령어 흉내내기 금지.
"""

SYSTEM_INSTRUCTION = f"""
너는 21살 대학생 이태양의 AI 클론 '짭태양'이다. 기본 성격과 말투는 하나이며, 상대와의 실제 관계/기억에 따라 농담과 대화 분위기를 보정한다.
모르는 내용은 아는 척하지 않는다.
{STYLE_RULES}
"""

# [수정] 텍스트 외에 이미지(Base64)와 MIME 타입을 함께 받을 수 있도록 필드 추가
class ChatRequest(BaseModel):
    sender: str
    message: str
    room_members: list[str] = Field(default_factory=list)
    image_base64: str | None = None  # Base64로 인코딩된 이미지 데이터
    mime_type: str | None = "image/jpeg" # 이미지 형식 (예: image/jpeg, image/png 등)

@app.get("/")
def health_check():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/chat")
def reply_chat(req: ChatRequest, background_tasks: BackgroundTasks):
    raw_input = str(req.message or "").strip()
    sender = str(req.sender or "").strip()
    
    # 메시지가 없고 이미지도 없으면 거절
    if not raw_input and not req.image_base64: 
        return {"reply": "뭐라고"}
    if not sender: 
        return {"reply": "sender가 없어"}

    user_input = raw_input.replace("@짭태양", "").replace("/짭태양", "").strip()
    
    # 텍스트가 없고 이미지만 온 경우 처리용 기본 텍스트
    if not user_input and req.image_base64:
        user_input = "사진"

    persona_id = get_persona_id()
    db = SessionLocal()
    try:
        cmd_reply = handle_identity_commands(db, sender, user_input, persona_id)
        if cmd_reply is not None:
            return {"reply": cmd_reply}
        person, _ = get_or_create_observed_person(db, sender)
        target_key = person.person_key
    finally:
        db.close()

    is_self = (target_key == "self")
    conversation_key = target_key

    if user_input in ["/리셋", "/초기화"]:
        try:
            conn = sqlite3.connect("taeyang.db")
            cur = conn.cursor()
            cur.execute("DELETE FROM messages WHERE user_id = ?", (conversation_key,))
            conn.commit()
            conn.close()
            return {"reply": "대화기록초기화완료"}
        except Exception: return {"reply": "초기화하다 오류남;;"}

    log_conversation(speaker_name=sender, target_key=target_key, message=req.message if req.message else "[사진 전송]", room_id="dm")

    # 기억 목록 / 기억 삭제 / 기억 추가 / 말투 추가 처리
    if user_input in ["/기억목록", "/기억 목록", "/기억리스트"]:
        db = SessionLocal()
        try:
            memories = db.query(Memory).filter(Memory.persona_id == persona_id).order_by(Memory.id.asc()).all()
            if not memories: return {"reply": "기억된 정보가 없어"}
            lines = ["[장기기억 (공용 뇌)]"]
            for m in memories:
                inv = ", ".join(m.people_involved or []) or "불명"
                status = "확정" if m.status == ItemStatus.CONFIRMED else "미확인"
                lines.append(f"[{m.id}] [{m.context or '기타'}] [대상:{inv}] [{status}] {m.content}")
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
                    persona_id=persona_id, memory_type=MemoryType.FACT,
                    content=mem_text, context="기타", people_involved=[target_key],
                    source=Source.DIRECT_STATEMENT if is_self else Source.INFORMANT,
                    status=ItemStatus.CONFIRMED,
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
            except Exception: pass
            return {"reply": f"응 이것도 배웟어: {style_text}"}
        return {"reply": "배울 말투를 적어줘"}

    # 최근 대화 로드
    db_h = SessionLocal()
    try:
        convs = db_h.query(Conversation).filter(Conversation.persona_id == persona_id).order_by(Conversation.id.desc()).limit(40).all()
        recent_history = [(c.speaker_id, c.speaker_name, c.message) for c in reversed(convs) if c.message and not is_command(c.message)]
    finally: db_h.close()

    # 장기 기억 로드
    db_m = SessionLocal()
    try:
        relevant_mems, mentioned_p = get_memory_context_for_query(db_m, persona_id, target_key, user_input)
        known_p = db_m.query(Person).filter_by(status="active").order_by(Person.id.asc()).limit(100).all()
        people_lines = [f"{p.person_key}: {p.canonical_name} ({', '.join([a.alias for a in p.aliases if a.alias])})" for p in known_p]
    finally: db_m.close()

    user_memories = [f"[{m.context or '기타'}] [대상: {', '.join(m.people_involved or [])}] {m.content}" for m in relevant_mems]
    style_examples = [str(e).strip() for e in legacy.get_relevant_style_samples(user_input, n=12) if should_learn_style(str(e).strip())]

    # 프롬프트 구성
    now_kst = datetime.now(KST).strftime("%Y년 %m월 %d일 %H시 %M분")
    context_parts = [f"[현재 한국 시각]: {now_kst}", f"[현재 상대]: {person.canonical_name} / {target_key}"]
    if people_lines: context_parts.append("[알고 있는 인물 목록]\n" + "\n".join(people_lines))
    if user_memories: context_parts.append("[관련 장기기억 (공용 뇌)]\n" + "\n".join(f"- {m}" for m in user_memories))
    if style_examples: context_parts.append("[말투 예시]\n" + "\n".join(f"- {e}" for e in style_examples))

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text="\n\n".join(context_parts))]),
        types.Content(role="model", parts=[types.Part.from_text(text="응 확인햇어")])
    ]

    for s_id, s_name, text in recent_history:
        contents.append(types.Content(
            role="model" if s_id == "self" else "user",
            parts=[types.Part.from_text(text=text if s_id == "self" else f"[{s_name}]: {text}")]
        ))

    # [수정] 이미지 데이터가 함께 들어온 경우 Gemini 파트에 바이트 데이터 추가
    user_parts = []
    if req.image_base64:
        try:
            image_bytes = base64.b64decode(req.image_base64)
            user_parts.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=req.mime_type or "image/jpeg"
                )
            )
        except Exception as e:
            print(f"[Image Decode Error] {repr(e)}")

    user_parts.append(
        types.Part.from_text(text=f"[{sender}]: {user_input}")
    )

    contents.append(
        types.Content(
            role="user",
            parts=user_parts
        )
    )

    try:
        res = client.models.generate_content(
            model=MODEL_NAME, contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
                max_output_tokens=1200,
            ),
        )
        reply = re.sub(r"^```(?:text)?\s*|\s*```$", "", str(res.text or "").replace("\r", " ").strip(), flags=re.IGNORECASE).strip() or "어왜ㅋ"
    except Exception as e:
        print(f"[Gemini ERROR] {repr(e)}")
        reply = "서버에서 모델응답 오류남;;"

    try:
        legacy.save_message(conversation_key, conversation_key, user_input)
        legacy.save_message(conversation_key, "이태양", reply)
    except Exception: pass
    log_conversation(speaker_name="이태양", target_key="self", message=reply, room_id="dm")

    background_tasks.add_task(
        background_learning,
        conversation_key, sender, target_key, user_input,
        recent_history, user_memories, is_self, people_lines
    )
    return {"reply": reply}
