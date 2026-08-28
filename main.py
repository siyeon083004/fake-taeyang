import os
import re
from datetime import datetime, timezone, timedelta
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
import database as db

db.init_db()

# 실제 카톡 대화에서 추출한 55,000줄의 이태양 말투 원본 로드[span_3](start_span)[span_3](end_span)[span_4](start_span)[span_4](end_span)
imported_count = db.import_style_samples("style_samples.txt")[span_5](start_span)[span_5](end_span)
if imported_count:
    print(f"[말투 학습 데이터] style_samples.txt에서 {imported_count}개 문장을 불러왔습니다.")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다.")

client = genai.Client(api_key=GEMINI_API_KEY)
KST = timezone(timedelta(hours=9))

MODEL_NAME = "gemini-3.6-flash"

SELF_NAME_KEYWORD = "이태양"
SELF_ID = "본인"

# 팩트 기반 기본 상식[span_6](start_span)[span_6](end_span)
BACKGROUND_KNOWLEDGE = """
[이태양 기본 상식 & 관계]
- 21세 여자 대학생(공대생), 대전 거주, 한화 이글스 팬.[span_7](start_span)[span_7](end_span)
- 주로 하는 게임: 마피아42(맢42, 맢), 테일즈런너(테런).[span_8](start_span)[span_8](end_span)
- 나성수(성수): 마피아42 대표/개발자. 서버 터지거나 패치 망칠 때마다 욕함.[span_9](start_span)[span_9](end_span)
- 챠(한이현, 닉네임 만사랑 등): 마피아42에서 만난 30세 찐친. 갠톡 매일 함. 챠는 현질을 잘하고 이벤트를 잘함.[span_10](start_span)[span_10](end_span)
- 만세, 잘싸, 허디, 난힐, 불쾌, 심저: 맢42/테런 같이 하는 지인들.[span_11](start_span)[span_11](end_span)
"""

STYLE_RULES = f"""
{BACKGROUND_KNOWLEDGE}

[말투 규칙]
1. 인위적이거나 과장된 억지 말투(알앗겟음, 잇음?? 등) 절대 금지.
2. 아래 [이태양이 실제로 쓴 말투 예시]에 나오는 표현과 톤을 그대로 따라서 자연스럽게 대답해라.[span_12](start_span)[span_12](end_span)[span_13](start_span)[span_13](end_span)
3. 줄바꿈(\\n) 금지, 카톡 단답형(1~30자 내외)으로 툭툭 던진다.
4. 문장부호(. !)는 쓰지 않고 물음표(?)는 쓴다.
5. 마피아42 관련 질문이나 계산은 상황에 맞게 팩트로만 대답한다[span_14](start_span)[span_14](end_span).
"""

SYSTEM_INSTRUCTION_FOR_CHA = f"""너는 21세 대학생 '이태양'이다[span_15](start_span)[span_15](end_span).
대화 상대는 마피아42로 알게 된 오랜 지인 '챠'(만사랑, 한이현)이다[span_16](start_span)[span_16](end_span).
서로 모든 걸 다 아는 찐친 사이니까 쿨하고 털털하게 반말로 대화해라.
{STYLE_RULES}
"""

SYSTEM_INSTRUCTION_FOR_SELF = f"""너는 21세 대학생 '이태양'의 AI 클론 '짭태양'이다.
지금 대화 상대는 진짜 이태양 본인이다. 혼잣말하듯 털털하게 받아쳐라. 상대를 '챠'라고 부르지 마라.
{STYLE_RULES}
"""

DEFAULT_FRIEND_SYSTEM_INSTRUCTION = f"""너는 21세 대학생 '이태양'이다[span_17](start_span)[span_17](end_span).
대화 상대는 친한 게임 친구이다. 편하게 반말로 툭툭 던지듯이 대화해라.
{STYLE_RULES}
"""

FRIEND_PERSONAS = {
    "챠": {
        "keyword": "한이현",
        "system_instruction": SYSTEM_INSTRUCTION_FOR_CHA,
    },
    "만사랑": {
        "keyword": "만사랑",
        "system_instruction": SYSTEM_INSTRUCTION_FOR_CHA,
    },
    "바보만세": {
        "keyword": "만세",
        "system_instruction": DEFAULT_FRIEND_SYSTEM_INSTRUCTION,
    }
}

DEEP_TOPIC_KEYWORDS = [
    "고민", "힘들", "진지하게", "진지한", "걱정", "우울", "속상", "스트레스",
    "어떡하지", "어떻게 해야", "조언", "괜찮을까", "무섭", "불안", "헤어져",
    "그만두", "포기", "죽고싶", "죽고 싶"
]

HANGUL_SYLLABLE = re.compile(r"[가-힣]")

def is_deep_topic(text: str) -> bool:
    if len(text) >= 35:
        return True
    return any(keyword in text for keyword in DEEP_TOPIC_KEYWORDS)

def is_meaningful_style_sample(text: str) -> bool:
    return len(text) >= 2 and bool(HANGUL_SYLLABLE.search(text))

LEAK_PATTERNS = ["constraints check", "system prompt", "policy", "disallowed", "as an ai", "i cannot", "i can't"]

def sanitize_reply(text: str) -> str:
    lowered = text.lower()
    if any(p in lowered for p in LEAK_PATTERNS):
        return "어왜ㅋ"

    hangul_count = len(HANGUL_SYLLABLE.findall(text))
    alpha_count = len(re.findall(r"[a-zA-Z]", text))
    if alpha_count >= 8 and hangul_count == 0:
        return "어왜ㅋ"

    return text

def resolve_friend(sender: str):
    for friend_id, info in FRIEND_PERSONAS.items():
        if info["keyword"] in sender:
            return friend_id, info["system_instruction"]
    return sender, DEFAULT_FRIEND_SYSTEM_INSTRUCTION

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

    if is_self:
        conversation_key = SELF_ID
        system_instruction = SYSTEM_INSTRUCTION_FOR_SELF
    else:
        conversation_key, system_instruction = resolve_friend(req.sender)

    # 1. 리셋 명령어
    if user_input in ["/리셋", "/초기화"]:
        conn = sqlite3.connect("taeyang.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE user_id = ?", (conversation_key,))
        conn.commit()
        conn.close()
        return {"reply": "대화기록초기화완료"}

    # 2. 기억 목록 확인
    if user_input in ["/기억목록", "/기억 목록"]:
        rows = db.get_memories_with_id(conversation_key)[span_18](start_span)[span_18](end_span)
        if not rows:
            return {"reply": "기억된 정보가 없어"}
        items = [f"[{r[0]}] {str(r[1]).replace(chr(10), ' ')}" for r in rows]
        return {"reply": " | ".join(items)}

    # 3. 기억 삭제
    if user_input.startswith("/기억삭제"):
        target = user_input.replace("/기억삭제", "").strip()
        if target.isdigit():
            success = db.delete_memory_by_id(conversation_key, int(target))[span_19](start_span)[span_19](end_span)
            return {"reply": f"기억삭제완료: [{target}]번" if success else f"[{target}]번 기억을 찾을 수 없어"}
        return {"reply": "삭제할 번호를 숫자로 입력해줘 (예: /기억삭제 1)"}

    # 4. 기억 저장
    if user_input.startswith("/기억 "):
        mem_text = user_input.replace("/기억 ", "", 1).strip()
        if mem_text:
            db.save_memory(conversation_key, mem_text)[span_20](start_span)[span_20](end_span)
            return {"reply": f"응기억햇어: {mem_text}"}

    # 5. 말투 학습 명령어
    if user_input.startswith("/말투 "):
        style_text = user_input.replace("/말투 ", "", 1).strip()
        if style_text:
            db.save_style_sample(style_text)[span_21](start_span)[span_21](end_span)
            return {"reply": f"응 이것도 배웟어: {style_text}"}

    # 자동 말투 학습
    if is_self and is_meaningful_style_sample(user_input):
        db.save_style_sample(user_input)[span_22](start_span)[span_22](end_span)

    # 컨텍스트 조립
    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    recent_history = db.get_recent_messages(conversation_key, limit=10)[span_23](start_span)[span_23](end_span)
    user_memories = db.get_memories(conversation_key)[span_24](start_span)[span_24](end_span)
    
    # DB에서 질문과 가장 관련된 실제 본인 카톡 문장 12개 추출[span_25](start_span)[span_25](end_span)[span_26](start_span)[span_26](end_span)
    style_examples = db.get_relevant_style_samples(user_input, n=12)
    # DB에서 질문 속 키워드로 과거 카톡 내역 지식 검색[span_27](start_span)[span_27](end_span)[span_28](start_span)[span_28](end_span)
    knowledge_records = db.search_knowledge(user_input, limit=6)

    deep_mode = is_deep_topic(user_input)

    contents = []
    context_parts = [f"[현재 한국 시각]: {current_time_str}"]
    context_parts.append(f"[답변 모드]: {'진지 모드' if deep_mode else '평소 모드'}")
    
    if user_memories:
        context_parts.append("[기억할 정보]: " + ", ".join(user_memories))
        
    if knowledge_records:
        context_parts.append("[참고할 과거 실제 카톡 내용]: " + " / ".join(knowledge_records))
        
    if style_examples:
        context_parts.append(
            "[이태양이 실제로 쓴 말투 예시, 이 느낌과 표현으로 대답해]: " + " / ".join(style_examples)
        )

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n".join(context_parts))]))
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="응 확인햇어")]))

    for sender, text in recent_history:
        role = "model" if sender == "이태양" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=1500,
            )
        )
        reply = response.text.replace("\n", " ").strip() if response.text else "어왜ㅋ"
        reply = sanitize_reply(reply)
    except Exception as e:
        print(f"[Gemini 에러 상세] {e}")
        reply = f"에러: {str(e)[:60]}"

    db.save_message(conversation_key, conversation_key, user_input)[span_29](start_span)[span_29](end_span)
    db.save_message(conversation_key, "이태양", reply)[span_30](start_span)[span_30](end_span)

    return {"reply": reply}
