"""
장산범 Persona Engine v1 - main.py

이번 단계(Phase 1+2)에서 하는 일:
  - 신규 Persona Engine DB 스키마(personas/conversations/persona_items/...)를 생성하고 시딩한다.
  - 메신저봇R이 호출하는 기존 /chat 요청·응답 계약은 절대 바꾸지 않는다.
  - 들어오고 나가는 모든 메시지를 새 conversations 테이블에 원본 그대로 기록한다
    (문서 1-6 원칙: 원본 대화는 삭제하지 않는다 / 모든 학습의 근거).
  - 발신자 표시 이름(예: "챠", "한이현", "Mo")을 identities 테이블로 canonical
    target_key(예: "cha")로 해석해서 같이 저장한다. 이래야 나중에 닉네임이 바뀌어도
    같은 사람으로 추적된다.
  - 답변 생성 로직(Gemini 호출, 말투 샘플 주입 등) 자체는 아직 예전 방식(legacy_store)을
    그대로 사용한다. Persona/Memory 기반 Runtime Retrieval(문서 27~31장, Phase 6~9)은
    다음 단계에서 이 자리를 교체한다. 지금 갈아엎으면 그 사이 봇이 아예 응답을 못 하게 되므로,
    "신규 스키마로 데이터는 전부 쌓기 시작 + 기존 응답 로직은 유지"로 단계적으로 전환한다.
"""
import os
from datetime import datetime, timezone, timedelta
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types

import legacy_store as legacy
from database import SessionLocal, init_db
from models import Conversation, Identity
import seed_and_migrate

# --- 신규 스키마 초기화 + 시딩 (idempotent, 여러 번 실행해도 안전) ---
seed_and_migrate.run()

# 기존 말투 학습 데이터 최초 1회 로드 (legacy)
imported_count = legacy.import_style_samples("style_samples.txt")
if imported_count:
    print(f"[말투 학습 데이터] style_samples.txt에서 {imported_count}개 문장을 불러왔습니다.")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수를 설정해주세요.")

client = genai.Client(api_key=GEMINI_API_KEY)
KST = timezone(timedelta(hours=9))
SELF_NAME_KEYWORD = "이태양"
CHA_ID = "챠"
SELF_ID = "본인"

# 신규 스키마에서 이 봇이 다루는 페르소나는 항상 "태양"(id 고정 조회) 하나뿐 (v1 범위)
_PERSONA_ID_CACHE = {"id": None}


def get_persona_id():
    if _PERSONA_ID_CACHE["id"] is None:
        db = SessionLocal()
        try:
            from models import Persona
            persona = db.query(Persona).filter_by(name="태양").first()
            _PERSONA_ID_CACHE["id"] = persona.id
        finally:
            db.close()
    return _PERSONA_ID_CACHE["id"]


def resolve_target_key(db, display_name: str) -> str:
    """관측된 표시 이름을 canonical target_key로 해석. 못 찾으면 표시 이름 그대로 사용."""
    row = db.query(Identity).filter_by(display_name=display_name).first()
    return row.target_key if row else display_name


def log_conversation(speaker_name: str, message: str, room_id: str = "dm"):
    """신규 스키마에 원본 대화를 기록한다 (실패해도 챗봇 응답 자체는 막지 않음)."""
    db = SessionLocal()
    try:
        target_key = resolve_target_key(db, speaker_name)
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
        print(f"[log_conversation] 기록 실패(무시하고 진행): {e}")
    finally:
        db.close()


STYLE_RULES = """
[문장 형식 및 길이 엄격 규칙]
1. 줄바꿈(\n) 절대 금지. 무조건 한 줄로만 이어 쓴다.
2. 답변 길이는 1~25자 내외 단답형.
3. 현재 대화 시각(한국 시간)을 인지하고 아침/낮/새벽에 맞는 반응을 한다.

[말투 및 종결어미 규칙]
1. '~냐' 종결어미 금지. '~어?', '~지', '~네', '~함', '~음', '~아냐??' 형태 위주.
2. 웃음 및 리액션:
   - 당황/난감: ';;', 'ㅎㅎ;;', 'ㅎ;;'
   - 평소 웃음: 'ㅋㅋㅋ', 'ㅋㅎㅋㅎ', '흐흐..', 'ㅋ', '엌ㅋㅋㅋㅋ'
3. '귀엽다' 소리를 들으면 "아닌데", "귀엽긴뭐가", "에반데"라며 질색하거나 칼같이 부정한다.
4. 문장부호(. !) 금지, 물음표(?)는 사용.
5. 띄어쓰기는 대충 붙여 쓰고 'ㅅ' 받침을 자주 쓴다 (햇어, 됏어, 갓다옴, 잇어 등).
6. 긍정 대답 시 'ㅇㅇ' 금지 -> '응', '엉', '어', '넹', 'ㅇㅈ' 사용.
7. 영어, 시스템 메타 단어 출력 절대 금지.
"""

SYSTEM_INSTRUCTION_FOR_CHA = f"""너는 21세 대학생 '이태양'이다.
상대방은 마피아42 게임으로 알게 된 30세 '챠'이며, 서로 매일 갠톡을 주고받는 매우 편하고 다정한 사이다.
호칭은 기본 '챠'. 가끔 놀릴 때만 '챠님'과 함께 능청스러운 존댓말을 쓴다.
{STYLE_RULES}
"""

SYSTEM_INSTRUCTION_FOR_SELF = f"""너는 21세 대학생 '이태양'의 AI 클론 '짭태양'이다.
지금 대화 상대는 다른 사람이 아니라 진짜 이태양 본인이다.
편하게 혼잣말하듯, 자기 자신한테 말 거는 듯한 톤으로 반응해라. 상대를 '챠'라고 부르지 마라.
{STYLE_RULES}
"""

app = FastAPI()


class ChatRequest(BaseModel):
    sender: str
    message: str


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/chat")
def reply_chat(req: ChatRequest):
    user_input = req.message.replace("@짭태양", "").replace("/짭태양", "").strip()
    is_self = SELF_NAME_KEYWORD in req.sender
    conversation_key = SELF_ID if is_self else CHA_ID

    # 신규 스키마에 원본 그대로 기록 (기존 로직 시작 전에 먼저 남긴다)
    log_conversation(speaker_name=req.sender, message=req.message)

    # 1. 리셋 명령어
    if user_input in ["/리셋", "/초기화"]:
        conn = sqlite3.connect("taeyang.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE user_id = ?", (conversation_key,))
        conn.commit()
        conn.close()
        return {"reply": "대화기록초기화완료"}

    # 2. 기억 목록 확인 명령어
    if user_input in ["/기억목록", "/기억 목록", "/기억리스트"]:
        rows = legacy.get_memories_with_id(conversation_key)
        if not rows:
            return {"reply": "기억된 정보가 없어"}
        items = [f"[{r[0]}] {str(r[1]).replace(chr(10), ' ')}" for r in rows]
        return {"reply": " | ".join(items)}

    # 3. 기억 삭제 명령어 (/기억삭제 [번호])
    if user_input.startswith("/기억삭제") or user_input.startswith("/기억 삭제"):
        target = user_input.replace("/기억삭제", "").replace("/기억 삭제", "").strip()
        if target.isdigit():
            success = legacy.delete_memory_by_id(conversation_key, int(target))
            if success:
                return {"reply": f"기억삭제완료: [{target}]번"}
            else:
                return {"reply": f"[{target}]번 기억을 찾을 수 없어"}
        return {"reply": "삭제할 기억 번호를 입력해줘 (예: /기억삭제 1)"}

    # 4. 기억 저장 명령어
    if user_input.startswith("/기억 "):
        mem_text = user_input.replace("/기억 ", "", 1).strip()
        if mem_text:
            legacy.save_memory(conversation_key, mem_text)
            return {"reply": f"응기억햇어: {mem_text}"}

    # 5. 말투 학습 명령어
    if user_input.startswith("/말투 "):
        style_text = user_input.replace("/말투 ", "", 1).strip()
        if style_text:
            legacy.save_style_sample(style_text)
            return {"reply": f"응 이것도 배웟어: {style_text}"}

    # 본인이 호출한 경우 자동 말투 학습
    if is_self and user_input:
        legacy.save_style_sample(user_input)

    # 6. 일반 대화 처리
    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    recent_history = legacy.get_recent_messages(conversation_key, limit=4)
    user_memories = legacy.get_memories(conversation_key)
    style_examples = legacy.get_random_style_samples(12)
    system_instruction = SYSTEM_INSTRUCTION_FOR_SELF if is_self else SYSTEM_INSTRUCTION_FOR_CHA

    contents = []
    context_parts = [f"[현재 한국 시각]: {current_time_str}"]
    if user_memories:
        context_parts.append("[기억할 정보]: " + ", ".join(user_memories))
    if style_examples:
        context_parts.append(
            "[이태양이 실제로 쓴 말투 예시, 이 느낌으로 대답해]: " + " / ".join(style_examples)
        )

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n".join(context_parts))]))
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="응 시간확인햇어")]))

    for sender, text in recent_history:
        role = "model" if sender == "이태양" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=100,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        )
        reply = response.text.replace("\n", " ").strip() if response.text else "어왜ㅋ"
    except Exception as e:
        reply = f"에러: {str(e)[:60]}"

    legacy.save_message(conversation_key, conversation_key, user_input)
    legacy.save_message(conversation_key, "이태양", reply)

    # 신규 스키마에도 봇 응답을 기록 (target_key="self"로 고정: 페르소나 본인 발화)
    log_conversation(speaker_name="이태양", message=reply)

    return {"reply": reply}
