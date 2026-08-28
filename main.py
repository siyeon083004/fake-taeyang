import os
from datetime import datetime, timezone, timedelta
import sqlite3
import re
import random
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
import database as db

db.init_db()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수를 설정해주세요.")

client = genai.Client(api_key=GEMINI_API_KEY)
KST = timezone(timedelta(hours=9))

TAEYANG_KEYWORD = "이태양"
TAEYANG_ID = "이태양"
OTHER_ID = "상대방"

# 잘싸 실제 발화 기반 랜덤 샘플 풀
JALSSA_STYLE_POOL = [
    "어허",
    "야 돼지 한판하고 가냐",
    "ㅉㅉ",
    "만세랑 손잡고와라 ㅋㅋㅋㅋ",
    "겜 ㄱㄱㄱ",
    "3일쉬는 이태양 겜 ㄱㄱㄱ",
    "스팀겜도 할거잇나 지금찾아봐야겟다",
    "이러면서 안온다에 만세검",
    "야 나 그래서 보석바꿨는데",
    "오늘 다른보석 쓸거다 난",
    "밥먹고 접메 ㄱㄱ",
    "어어ㅉㅉ 돼지on",
    "야 재밌는겜 알아와라 ㄹㅇ",
    "오늘 스팀 ㄱㄱ?",
    "맢할까 스팀할까",
    "뺏자 동생컴터 ㅋㅋㅋㅋㅋ",
    "나이스다 ㅋㅋㅋㅋㅋ",
    "생각이안나",
    "어엉",
    "ㅇㅋㅇㅋ"
]

# 잘싸 성격 및 대화 규칙
JALSSA_CHARACTER_RULES = """
[문장 형식 및 길이 엄격 규칙]
1. 줄바꿈(\n) 절대 금지. 무조건 한 줄로만 이어 쓴다.
2. 답변 길이는 1~25자 내외의 극단적 단답형.
3. 마침표(.)나 느낌표(!)는 쓰지 말고 물음표(?)는 사용 가능.
4. 띄어쓰기는 자연스럽게 뭉개며 친다.

[잘싸 성격 및 대화 스타일]
1. 말투: 무심하고 툭툭 던지는 단답 위주 ('어허', '어어', 'ㅇㅋㅇㅋ', 'ㄱㄱ', '나이스다', 'ㅉㅉ').
2. 이태양 상대 태도: 친한 맢42 게임 친구. 이태양을 가끔 '돼지', '이태양'이라 부르며 겜하자고 재촉함 ('겜 ㄱㄱㄱ', '스팀 ㄱㄱ?').
3. 지인 언급: 가끔 '만세'를 걸고 장난침 ('만세랑 손잡고와라', '안온다에 만세검').
4. 주요 소재: 마피아42 접메/보석 세팅, 스팀 게임 뭐할지, 동생 컴 뺏기, 피방 가기.
5. 설정을 한 번에 길게 풀지 말고 상대방 말에만 툭툭 짧게 반응한다.
6. 영어 및 시스템 메타 단어 출력 절대 금지.
"""

SYSTEM_INSTRUCTION_FOR_TAEYANG = f"""너는 마피아42와 스팀 게임을 좋아하는 쿨한 남동생/친구 '잘싸(잘생긴싸람)'이다.
대화 상대는 맢42를 같이 하는 '이태양'이다.
- 이태양을 편하게 '야', '돼지', '이태양'이라 부르며 툭툭 장난친다.
- 겜하자고 쪼아대거나 쿨하게 '어허', '겜 ㄱㄱㄱ', '스팀 ㄱㄱ' 거린다.
- 이태양이 하는 말에만 짧고 무심하게 한 줄로 맞받아쳐라.
{JALSSA_CHARACTER_RULES}
"""

SYSTEM_INSTRUCTION_FOR_OTHERS = f"""너는 마피아42와 스팀 게임을 좋아하는 쿨한 성격의 '잘싸'이다.
상대방에게 무심한 듯 툭툭 짧은 단답으로 반응해라.
{JALSSA_CHARACTER_RULES}
"""

app = FastAPI()

class ChatRequest(BaseModel):
    sender: str
    message: str

@app.get("/")
def health_check():
    return {"status": "jalssa_bot_online"}

@app.post("/chat")
def reply_chat(req: ChatRequest):
    raw_msg = req.message.strip()
    
    # 짭태양 호출 시 바로 짭잘싸로 넘기거나 사망 처리 방지
    user_input = raw_msg.replace("@짭잘싸", "").replace("/짭잘싸", "").replace("@잘싸", "").replace("/잘싸", "").replace("@짭태양", "").replace("/짭태양", "").strip()
    
    is_taeyang = TAEYANG_KEYWORD in req.sender
    conversation_key = TAEYANG_ID if is_taeyang else OTHER_ID

    # 1. 리셋 명령어
    if user_input in ["/리셋", "/초기화", "리셋", "초기화"]:
        conn = sqlite3.connect("taeyang.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE user_id = ?", (conversation_key,))
        conn.commit()
        conn.close()
        return {"reply": "짭잘싸 : 대화기록 리셋함"}

    # 2. 기억 목록 확인 명령어
    if user_input in ["/기억목록", "/기억 목록", "/기억리스트", "기억목록", "기억 목록"]:
        rows = db.get_memories_with_id(conversation_key)
        if not rows:
            return {"reply": "짭잘싸 : 기억된거 없는데??"}
        items = [f"[{r[0]}] {str(r[1]).replace(chr(10), ' ')}" for r in rows]
        return {"reply": "짭잘싸 : " + " | ".join(items)}

    # 3. 기억 삭제 명령어
    if user_input.startswith("/기억삭제") or user_input.startswith("/기억 삭제") or user_input.startswith("기억삭제") or user_input.startswith("기억 삭제"):
        target = re.sub(r"^/?기억\s*삭제\s*", "", user_input).strip()
        if target.isdigit():
            success = db.delete_memory_by_id(conversation_key, int(target))
            if success:
                return {"reply": f"짭잘싸 : [{target}]번 기억 삭제완료"}
            else:
                return {"reply": f"짭잘싸 : [{target}]번 기억 없음"}
        return {"reply": "짭잘싸 : 번호 써라 (예: /기억삭제 1)"}

    # 4. 기억 저장 명령어
    if user_input.startswith("/기억 ") or user_input.startswith("기억 "):
        mem_text = re.sub(r"^/?기억\s+", "", user_input).strip()
        if mem_text:
            db.save_memory(conversation_key, mem_text)
            return {"reply": f"짭잘싸 : 오냐 기억해둠 : {mem_text}"}

    # 5. 일반 대화 처리
    if not user_input:
        user_input = "겜 ㄱㄱ"

    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    recent_history = db.get_recent_messages(conversation_key, limit=4)
    user_memories = db.get_memories(conversation_key)
    
    style_examples = random.sample(JALSSA_STYLE_POOL, min(8, len(JALSSA_STYLE_POOL)))
    system_instruction = SYSTEM_INSTRUCTION_FOR_TAEYANG if is_taeyang else SYSTEM_INSTRUCTION_FOR_OTHERS

    contents = []
    context_parts = [f"[현재 한국 시각]: {current_time_str}"]
    if user_memories:
        context_parts.append("[기억할 정보]: " + ", ".join(user_memories))
    if style_examples:
        context_parts.append(
            "[잘싸가 실제로 쓴 말투 예시, 이 느낌으로 대답해]: " + " / ".join(style_examples)
        )

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n".join(context_parts))]))
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="어허 시간확인함")]))

    for sender, text in recent_history:
        role = "model" if sender == "잘싸" else "user"
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
        raw_reply = response.text.replace("\n", " ").strip() if response.text else "어허 ㅋㅋㅋㅋ"
    except Exception as e:
        raw_reply = f"에러: {str(e)[:60]}"

    db.save_message(conversation_key, req.sender, user_input)
    db.save_message(conversation_key, "잘싸", raw_reply)

    return {"reply": f"짭잘싸 : {raw_reply}"}
