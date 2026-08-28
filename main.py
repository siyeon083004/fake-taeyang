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

# 실제 카톡 대화에서 뽑은 이태양 말투 예시 문장들을 DB에 채워넣음 (최초 1회만 실행됨)[cite: 8]
imported_count = db.import_style_samples("style_samples.txt")[cite: 8]
if imported_count:
    print(f"[말투 학습 데이터] style_samples.txt에서 {imported_count}개 문장을 불러왔습니다.")

# 제미나이 클라이언트
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다.")

client = genai.Client(api_key=GEMINI_API_KEY)
KST = timezone(timedelta(hours=9))

# Render 로그에서 요구한 권장 신규 모델
MODEL_NAME = "gemini-3.6-flash"

SELF_NAME_KEYWORD = "이태양"
SELF_ID = "본인"

STYLE_RULES = """
[문장 형식 및 길이 규칙]
1. 줄바꿈(엔터, \\n)은 절대 치지 마라. 무조건 한 줄로만 이어 써서 보낸다.
2. 카톡 답장다운 짧은 호흡(1~35자 내외)으로 쓴다.
3. 현재 대화 시각(한국 시간)을 정확히 인지하고 아침/낮/새벽에 맞는 현실적인 반응을 한다.
4. 상대방이 방금 한 말의 내용과 맥락을 정확히 파악해서, 엉뚱한 소리 대신 그 말에 실제로 맞는 대답을 해라.

[말투 및 텍스트 습관]
1. '~냐' 종결어미 금지. 친근하고 편안한 '~어?', '~지', '~네', '~함', '~음', '~아냐??' 형태로 대화한다.
2. 웃음 및 리액션:
   - 당황/뻘쭘/난감: ';;', 'ㅎㅎ;;', 'ㅎ;;' 자연스럽게 사용.
   - 평소 웃음: 'ㅋㅋㅋ', 'ㅋㅎㅋㅎ', '흐흐..', 'ㅋ' 등을 상황에 맞게 쓴다.
3. '귀엽다' 소리를 들으면 "아닌데", "귀엽긴뭐가", "에반데"라며 질색하거나 칼같이 부정한다.
4. 문장부호(. !)는 쓰지 않고 물음표(?)는 쓴다.
5. 띄어쓰기는 적당히 붙여 쓰고 'ㅅ' 받침을 자주 쓴다 (햇어, 됏어, 갓다옴, 먹엇음, 잇어 등).
6. 긍정 대답 시 'ㅇㅇ' 금지 -> '응', '엉', '어', '넹', 'ㅇㅈ' 사용.
7. 영어, 시스템 메타 단어 출력 절대 금지.
8. 끝말잇기나 퀴즈는 룰을 정확히 계산하고 단답으로 센스 있게 맞받아친다.
"""

SYSTEM_INSTRUCTION_FOR_CHA = f"""너는 21세 대학생 '이태양'이다. (여자다. 절대 남자처럼 말하지 마라.)
상대방은 마피아42 게임으로 알게 된 30세 여자 '챠'이며, 서로 매일 갠톡을 주고받는 매우 편하고 다정한 사이다.
호칭은 기본 '챠'. 가끔 놀릴 때만 '챠님'과 함께 능청스러운 존댓말을 쓴다.
{STYLE_RULES}
"""

SYSTEM_INSTRUCTION_FOR_SELF = f"""너는 21세 대학생 '이태양'의 AI 클론 '짭태양'이다. (이태양은 여자다. 절대 남자처럼 말하지 마라.)
지금 대화 상대는 다른 사람이 아니라 진짜 이태양 본인이다.
편하게 혼잣말하듯, 자기 자신한테 말 거는 듯한 톤으로 반응해라. 상대를 '챠'라고 부르지 마라.
{STYLE_RULES}
"""

DEFAULT_FRIEND_SYSTEM_INSTRUCTION = f"""너는 21세 대학생 '이태양'이다. (여자다. 절대 남자처럼 말하지 마라.) 
지금 대화 상대는 친하게 지내는 친구이다. 서로 편하게 반말하는 사이다.
{STYLE_RULES}
"""

FRIEND_PERSONAS = {
    "챠": {
        "keyword": "한이현",
        "system_instruction": SYSTEM_INSTRUCTION_FOR_CHA,
    },
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
        rows = db.get_memories_with_id(conversation_key)[cite: 8]
        if not rows:
            return {"reply": "기억된 정보가 없어"}
        items = [f"[{r[0]}] {str(r[1]).replace(chr(10), ' ')}" for r in rows]
        return {"reply": " | ".join(items)}

    # 3. 기억 삭제
    if user_input.startswith("/기억삭제"):
        target = user_input.replace("/기억삭제", "").strip()
        if target.isdigit():
            success = db.delete_memory_by_id(conversation_key, int(target))[cite: 8]
            return {"reply": f"기억삭제완료: [{target}]번" if success else f"[{target}]번 기억을 찾을 수 없어"}
        return {"reply": "삭제할 번호를 숫자로 입력해줘 (예: /기억삭제 1)"}

    # 4. 기억 저장
    if user_input.startswith("/기억 "):
        mem_text = user_input.replace("/기억 ", "").strip()
        if mem_text:
            db.save_memory(conversation_key, mem_text)[cite: 8]
            return {"reply": f"응기억햇어: {mem_text}"}

    # 5. 말투 학습 명령어
    if user_input.startswith("/말투 "):
        style_text = user_input.replace("/말투 ", "").strip()
        if style_text:
            db.save_style_sample(style_text)[cite: 8]
            return {"reply": f"응 이것도 배웟어: {style_text}"}

    # 자동 말투 학습
    if is_self and is_meaningful_style_sample(user_input):
        db.save_style_sample(user_input)[cite: 8]

    # 컨텍스트 조립
    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    recent_history = db.get_recent_messages(conversation_key, limit=10)[cite: 8]
    user_memories = db.get_memories(conversation_key)[cite: 8]
    
    if hasattr(db, "get_relevant_style_samples"):
        style_examples = db.get_relevant_style_samples(user_input, n=12)
    else:
        style_examples = db.get_random_style_samples(12)[cite: 8]

    deep_mode = is_deep_topic(user_input)

    contents = []
    context_parts = [f"[현재 한국 시각]: {current_time_str}"]
    context_parts.append(f"[답변 모드]: {'진지 모드' if deep_mode else '평소 모드'}")
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
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.75,
                max_output_tokens=1500,
            )
        )
        reply = response.text.replace("\n", " ").strip() if response.text else "어왜ㅋ"
        reply = sanitize_reply(reply)
    except Exception as e:
        print(f"[Gemini 에러 상세] {e}")
        reply = f"에러: {str(e)[:60]}"

    db.save_message(conversation_key, conversation_key, user_input)[cite: 8]
    db.save_message(conversation_key, "이태양", reply)[cite: 8]

    return {"reply": reply}
