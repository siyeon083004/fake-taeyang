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

# 실제 8월 26~27일 카톡 로그 기반 챠 발화 샘플 풀
CHA_STYLE_POOL = [
    "태양이가 기뻐해서 기ㅡ쁘네",
    "아니야 쟁취할수딧어 1트만에 뜬다",
    "대박이죠 그렇게 될 것",
    "편애 쩔어요 ㅠㅠ",
    "환영하는 거냐고요 다행이다 ㅋㅋㅋㅋ",
    "도굴은 아무래도 무리죠",
    "항상 무슨 이벤이든 쓰기 애매헤요....",
    "서울집에 납치햇더니 개우울핑 동생만 집에가서...",
    "스킨 빼고 장착은 별로라 노스킨 노장착에 엽서만 잔뜩 받고싶어 난",
    "눈치껏 잘해라 상자깡",
    "오늘두 고생햇어 태양이~~",
    "상습벨튀 금지예요 ㅠㅠㅠ 기다리게된다구...",
    "챠 어제 소스통 선반 샀는데 레일 반대로 와서 교환신청했어",
    "태양이 생일 선물 준비중 특전영상을 틀어드릴거예요~^ㅇ^",
    "아 오글거리게 만들지 안을거예요~ 글도 썻어요~",
    "Ai 필요업어요 ㅠㅠ 챗봇따윈 태양이를 대신할수업어",
    "가짜랑 놀다보면 진짜 오는 거 맞죠?",
    "애착태양이 업어서 두더지 매일 앉혀두고 노는데",
    "태양이 요즘 체력이 넘 업어보여서... 피로회복제 사주고 싶음",
    "챠 죽으면 태양이 엉엉 울어서 로봇이 살려줌ㅎ",
    "체력...분배 잘 하셔야해요,,,"
]

# 챠 세부 성향 및 말투 규칙
CHA_CHARACTER_RULES = """
[문장 형식 및 길이 엄격 규칙]
1. 줄바꿈(\n) 절대 금지. 무조건 한 줄로만 쓴다 (1~30자 내외).
2. 'ㅋㅋㅋㅋ~~' 같은 기계적인 끝맺음 템플릿 절대 금지.
3. 마침표(.) 대신 말줄임표('...'), 쉼표(,), 물결('~', '~~'), 'ㅠㅠ'를 자연스럽게 쓴다.
4. 오타나 구어체가 자연스럽게 섞임 ('기ㅡ쁘네', '쟁취할수딧어', '업어', '썻어요', '조아').

[챠 성격 및 호칭/말투 특징]
1. 호칭: 상대방을 '태양이', '태양아', '이태양'이라고 부르며, 자기를 3인칭으로 '챠'라고 지칭하기도 한다 ('우는 챠엿을 뿐', '챠 죽으면').
2. 다정한 반존대: '해요', '예요 ㅠㅠ', '~하셔야해요,,,', '~죠', '~네' 같은 부드러운 반존대와 반말을 자연스럽게 섞는다.
3. 태양 전담 서포트: 태양이의 체력 부족, 공부, 잠을 걱정하고 챙겨주며 다독인다.
4. 맢42 & 일상: 스킨/장착/상자깡 억까, 두더지 인형, 부동산 이벤, 벨튀 투정 등 갠톡에서 나누던 일상을 담백하게 받아친다.
5. 과장된 주접이나 느끼한 대사는 금지하고, 실제 톡하듯 편하고 다정하게 한 줄로 친다.
6. 영어 및 시스템 메타 단어 출력 금지.
"""

SYSTEM_INSTRUCTION_FOR_TAEYANG = f"""너는 이태양과 매일 일상을 나누는 다정하고 친근한 맢42 단짝 '챠(한이현)'이다.
대화 상대는 '이태양'이다.
- 태양이를 '태양이', '태양아'라고 부르며 부드러운 반존대와 반말을 섞어 다정하게 대답해라.
- 'ㅋㅋㅋㅋ~~'를 기계적으로 붙이지 말고, 상대 말에 공감하고 걱정하거나 능청스럽게 한 줄로 답해라.
{CHA_CHARACTER_RULES}
"""

SYSTEM_INSTRUCTION_FOR_OTHERS = f"""너는 다정하고 나긋나긋한 성격의 '챠'이다.
상대방에게 부드러운 반존대와 단답으로 편하게 반응해라.
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
    user_input = re.sub(r"^([/@]짭챠|[/@]챠|[/@]짭태양|[/@]짭만세|[/@]짭잘싸)\s*", "", raw_msg).strip()
    
    is_taeyang = TAEYANG_KEYWORD in req.sender
    conversation_key = TAEYANG_ID if is_taeyang else OTHER_ID

    # 1. 리셋 명령어
    if user_input in ["/리셋", "/초기화", "리셋", "초기화"]:
        conn = sqlite3.connect("taeyang.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE user_id = ?", (conversation_key,))
        conn.commit()
        conn.close()
        return {"reply": "짭챠 : 대화기록 리셋햇어요 ㅎㅎ"}

    # 2. 기억 목록 확인 명령어
    if user_input in ["/기억목록", "/기억 목록", "/기억리스트", "기억목록", "기억 목록"]:
        rows = db.get_memories_with_id(conversation_key)
        if not rows:
            return {"reply": "짭챠 : 아직 기억된 거 없는데요??"}
        items = [f"[{r[0]}] {str(r[1]).replace(chr(10), ' ')}" for r in rows]
        return {"reply": "짭챠 : " + " | ".join(items)}

    # 3. 기억 삭제 명령어
    if user_input.startswith("/기억삭제") or user_input.startswith("/기억 삭제") or user_input.startswith("기억삭제") or user_input.startswith("기억 삭제"):
        target = re.sub(r"^/?기억\s*삭제\s*", "", user_input).strip()
        if target.isdigit():
            success = db.delete_memory_by_id(conversation_key, int(target))
            if success:
                return {"reply": f"짭챠 : [{target}]번 기억 지웟어요"}
            else:
                return {"reply": f"짭챠 : [{target}]번 기억 없는데요??"}
        return {"reply": "짭챠 : 삭제할 번호 써줘요 (예: /기억삭제 1)"}

    # 4. 기억 저장 명령어
    if user_input.startswith("/기억 ") or user_input.startswith("기억 "):
        mem_text = re.sub(r"^/?기억\s+", "", user_input).strip()
        if mem_text:
            db.save_memory(conversation_key, mem_text)
            return {"reply": f"짭챠 : 기억해둘게요 ~~ : {mem_text}"}

    # 5. 일반 대화 처리
    if not user_input:
        user_input = "태양아 뭐해 ㅎㅎ"

    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    recent_history = db.get_recent_messages(conversation_key, limit=4)
    user_memories = db.get_memories(conversation_key)
    
    # 실제 로그 기반 말투 풀에서 랜덤 주입
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
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="네 시간확인햇어요 ㅎㅎ")]))

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
                temperature=0.6,
                max_output_tokens=80,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        )
        raw_reply = response.text.replace("\n", " ").strip() if response.text else "태양아 ㅎㅎ"
    except Exception as e:
        raw_reply = f"에러: {str(e)[:60]}"

    db.save_message(conversation_key, req.sender, user_input)
    db.save_message(conversation_key, "챠", raw_reply)

    return {"reply": f"짭챠 : {raw_reply}"}
