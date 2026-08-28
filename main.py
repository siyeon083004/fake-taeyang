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

# 챠 실제 발화/성향 기반 랜덤 샘플 풀
CHA_STYLE_POOL = [
    "태양아 ㅋㅋㅋㅋ",
    "아니 개웃기네 ㅋㅋㅋㅋ",
    "태양아 겜 ㄱ?",
    "나 방금 레전드 트롤함 ㅋㅋㅋㅋ",
    "ㅋㅋㅋㅋㅋ 에반데",
    "태양아 밥 먹음??",
    "오늘 맢42 이벤 달린다",
    "나 또 미접해서 출보 날림 ㅠㅠ",
    "ㅋㅋㅋㅋㅋㅋ 아 웃겨",
    "밀서 잘못 보냄 샤갈 ㅋㅋㅋㅋ",
    "태양아 과외 끝남??",
    "부계로 컬 다 따놧음 ㅋㅋㅋㅋ",
    "만세 걔 또 억까당함??",
    "아니 진짜 개웃기네 ㅋㅋㅋㅋㅋ",
    "오늘 스팀 할 거 잇냐",
    "태양아 나와 ㅋㅋㅋㅋ"
]

# 챠 성격 및 대화 규칙
CHA_CHARACTER_RULES = """
[문장 형식 및 길이 엄격 규칙]
1. 줄바꿈(\n) 절대 금지. 무조건 한 줄로만 이어 쓴다.
2. 답변 길이는 1~25자 내외 단답형.
3. 마침표(.)나 느낌표(!)는 쓰지 말고 물음표(?)는 적극 사용.
4. 띄어쓰기는 대충 뭉개며 친다.

[챠 성격 및 대화 스타일]
1. 관계: 이태양과 맢42에서 1+1으로 통하는 가장 편하고 다정한 영혼의 단짝.
2. 태도: 이태양에게 항상 다정하고 능청스럽게 '태양아', '태양'이라 부르며 ㅋㅋㅋㅋ를 자주 침.
3. 게임 스타일: 즐겜러 겸 맢42 기행 장인(트롤, 사칭 밀서, 비인기 컬 수집 등).
4. 지인 언급: 가끔 만세, 잘싸, 불쾌 이야기를 자연스럽게 섞음.
5. 설정을 장황하게 늘어놓지 말고, 상대방의 말에만 즉각 반응하며 찰지게 티키타카를 친다.
6. 영어 및 시스템 메타 단어 출력 절대 금지.
"""

SYSTEM_INSTRUCTION_FOR_TAEYANG = f"""너는 이태양과 가장 친한 1+1 단짝 '챠'이다.
대화 상대는 매일 갠톡하고 게임하는 실친 같은 단짝 '이태양'이다.
- 이태양을 다정하고 장난스럽게 '태양아', '태양'이라 부르며 ㅋㅋㅋㅋ거리며 받아쳐라.
- 이태양이 하는 말에만 집중해서 자연스럽고 짧게 한 줄로 반응해라.
{CHA_CHARACTER_RULES}
"""

SYSTEM_INSTRUCTION_FOR_OTHERS = f"""너는 다정하고 유쾌한 성격의 '챠'이다.
상대방에게 친근하게 ㅋㅋㅋㅋ거리며 단답으로 반응해라.
{CHA_CHARACTER_RULES}
"""

app = FastAPI()

class ChatRequest(BaseModel):
    sender: str
    message: str

@app.get("/")
def health_check():
    return {"status": "cha_bot_online"}

@app.post("/chat")
def reply_chat(req: ChatRequest):
    raw_msg = req.message.strip()
    
    # 짭태양/짭만세/짭잘싸 호출어 호환 처리
    user_input = raw_msg.replace("@짭챠", "").replace("/짭챠", "").replace("@챠", "").replace("/챠", "").replace("@짭태양", "").replace("/짭태양", "").strip()
    
    is_taeyang = TAEYANG_KEYWORD in req.sender
    conversation_key = TAEYANG_ID if is_taeyang else OTHER_ID

    # 1. 리셋 명령어
    if user_input in ["/리셋", "/초기화", "리셋", "초기화"]:
        conn = sqlite3.connect("taeyang.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE user_id = ?", (conversation_key,))
        conn.commit()
        conn.close()
        return {"reply": "짭챠 : 대화기록 리셋햇어 ㅋㅋㅋㅋ"}

    # 2. 기억 목록 확인 명령어
    if user_input in ["/기억목록", "/기억 목록", "/기억리스트", "기억목록", "기억 목록"]:
        rows = db.get_memories_with_id(conversation_key)
        if not rows:
            return {"reply": "짭챠 : 아직 기억된 거 없는데??"}
        items = [f"[{r[0]}] {str(r[1]).replace(chr(10), ' ')}" for r in rows]
        return {"reply": "짭챠 : " + " | ".join(items)}

    # 3. 기억 삭제 명령어
    if user_input.startswith("/기억삭제") or user_input.startswith("/기억 삭제") or user_input.startswith("기억삭제") or user_input.startswith("기억 삭제"):
        target = re.sub(r"^/?기억\s*삭제\s*", "", user_input).strip()
        if target.isdigit():
            success = db.delete_memory_by_id(conversation_key, int(target))
            if success:
                return {"reply": f"짭챠 : [{target}]번 기억 지웟음!"}
            else:
                return {"reply": f"짭챠 : [{target}]번 기억 없는데??"}
        return {"reply": "짭챠 : 삭제할 번호 써줘 (예: /기억삭제 1)"}

    # 4. 기억 저장 명령어
    if user_input.startswith("/기억 ") or user_input.startswith("기억 "):
        mem_text = re.sub(r"^/?기억\s+", "", user_input).strip()
        if mem_text:
            db.save_memory(conversation_key, mem_text)
            return {"reply": f"짭챠 : 오키 기억햇음 ㅋㅋㅋㅋ : {mem_text}"}

    # 5. 일반 대화 처리
    if not user_input:
        user_input = "태양아 뭐해 ㅋㅋㅋㅋ"

    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    recent_history = db.get_recent_messages(conversation_key, limit=4)
    user_memories = db.get_memories(conversation_key)
    
    style_examples = random.sample(CHA_STYLE_POOL, min(8, len(CHA_STYLE_POOL)))
    system_instruction = SYSTEM_INSTRUCTION_FOR_TAEYANG if is_taeyang else SYSTEM_INSTRUCTION_FOR_OTHERS

    contents = []
    context_parts = [f"[현재 한국 시각]: {current_time_str}"]
    if user_memories:
        context_parts.append("[기억할 정보]: " + ", ".join(user_memories))
    if style_examples:
        context_parts.append(
            "[챠가 실제로 쓴 말투 예시, 이 느낌으로 대답해]: " + " / ".join(style_examples)
        )

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n".join(context_parts))]))
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="응 시간확인햇어 ㅋㅋㅋㅋ")]))

    for sender, text in recent_history:
        role = "model" if sender == "챠" else "user"
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
        raw_reply = response.text.replace("\n", " ").strip() if response.text else "태양아 ㅋㅋㅋㅋ"
    except Exception as e:
        raw_reply = f"에러: {str(e)[:60]}"

    db.save_message(conversation_key, req.sender, user_input)
    db.save_message(conversation_key, "챠", raw_reply)

    return {"reply": f"짭챠 : {raw_reply}"}
