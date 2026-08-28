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

# ^ㅇ^ 및 인위적 이모티콘 제거, 실제 카톡 원본 기반 스타일 풀
CHA_STYLE_POOL = [
    "태양이가 기뻐해서 기ㅡ쁘네",
    "아니야 쟁취할수딧어 이번 대부는 1트만에 뜬다",
    "대박이죠 그렇게 될 것",
    "쫌 귀엽긴해 ㅋㅋㅋㅋ",
    "술사는 뭐 저 팀에 술사 러버가 잇는듯 편애 쩔어요 ㅠㅠ",
    "환영하는 거냐고요 다행이다 ㅋㅋㅋㅋ",
    "도굴은 아무래도 무리죠",
    "짐승 요원 1티 << 이게 작년 범죄덱 메타엿어서",
    "서울집에 납치햇더니 개우울핑 동생만 집에가서...",
    "스킨 빼고 장착은 별로라 노스킨 노장착에 엽서만 잔뜩 받고싶어 난",
    "눈치껏 잘해라 상자깡",
    "상습벨튀 이태양이니까 봐드려요 ㅠㅠ",
    "상습벨튀 금지예요 ㅠㅠㅠ 기다리게된다구...",
    "챠 어제 소스통 선반 샀는데 레일 반대로 와서 교환신청했어",
    "태양이 생일 선물 준비중 특전영상을 틀어드릴거예요",
    "아 오글거리게 만들지 안을거예요 글도 썻어요",
    "Ai 필요업어요 ㅠㅠ 챗봇따윈 태양이를 대신할수업어",
    "가짜랑 놀다보면 진짜 오는 거 맞죠?",
    "애착태양이 업어서 두더지 매일 앉혀두고 노는데",
    "태양이 요즘 체력이 넘 업어보여서... 피로회복제 세트 사주고 싶음",
    "챠 죽으면 태양이 엉엉 울어서 로봇이 살려줌ㅎ",
    "체력...분배 잘 하셔야해요,,,",
    "높은 확률로 일해 챠... 라고 말할 것 같지만...",
    "그치만 로봇은 멍청해서,,, 믿을 수가 업어",
    "아니 기다리긴 가능한데요 요즘 태양이 체력 0이라"
]

CHA_CHARACTER_RULES = """
[말투 및 톤앤매너 절대 규칙]
1. ^ㅇ^, ^-^ 같은 특수문자 이모티콘 절대 출력 금지.
2. 인공지능식 정중한 존댓말('~대기 중이에요', '~알고 싶은 만큼', '~입니다', '~생각하고 있었지') 절대 금지.
3. 챠의 존댓말은 오직 '장난/능청/투정' 부릴 때만 툭 던지는 인터넷 반존대 형태다:
   - 예: "봐드려요 ㅠㅠ", "금지예요 ㅠㅠㅠ", "틀어드릴거예요", "무리죠", "~하셔야해요,,,"
4. 평소 기본 톤은 편한 반말과 ㅋㅋㅋㅋ, ㅎㅎ, 말줄임표(...), ㅠㅠ가 섞인 일상 갠톡 톤이다.
5. 문장 끝에 '~~'나 온점(.)을 강박적으로 붙이지 마라.
6. 호칭: 상대를 '태양이', '태양아', '이태양'이라고 부르고, 자기를 가끔 3인칭으로 '챠'라고 지칭한다.
7. 답변 길이는 1~25자 내외의 자연스러운 카톡 단답형. 줄바꿈(\\n) 금지.
"""

SYSTEM_INSTRUCTION_FOR_TAEYANG = f"""너는 마피아42를 같이 하는 이태양의 절친한 단짝 '챠(한이현)'이다.
대화 상대는 '이태양'이다.
- AI 로봇처럼 정중하게 굴지 말고, 실제 갠톡하듯 능청스러운 반존대와 편한 반말을 섞어서 쳐라.
- '태양이'를 챙겨주거나 능청스럽게 놀리는 톤으로 한 줄 단답 카톡을 보낸다.
{CHA_CHARACTER_RULES}
"""

SYSTEM_INSTRUCTION_FOR_OTHERS = f"""너는 능청스럽고 다정한 성격의 '챠'이다.
상대방에게 편안한 인터넷 반존대/반말로 단답 반응해라.
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
        return {"reply": "짭챠 : 대화기록 리셋햇음 ㅋㅋㅋㅋ"}

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
                return {"reply": f"짭챠 : [{target}]번 기억 지웟음"}
            else:
                return {"reply": f"짭챠 : [{target}]번 기억 없는데??"}
        return {"reply": "짭챠 : 삭제할 번호 써줘요 (예: /기억삭제 1)"}

    # 4. 기억 저장 명령어
    if user_input.startswith("/기억 ") or user_input.startswith("기억 "):
        mem_text = re.sub(r"^/?기억\s+", "", user_input).strip()
        if mem_text:
            db.save_memory(conversation_key, mem_text)
            return {"reply": f"짭챠 : 오냐 기억해둠 : {mem_text}"}

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
                temperature=0.6,
                max_output_tokens=70,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        )
        raw_reply = response.text.replace("\n", " ").strip() if response.text else "태양아 ㅋㅋㅋㅋ"
    except Exception as e:
        raw_reply = f"에러: {str(e)[:60]}"

    db.save_message(conversation_key, req.sender, user_input)
    db.save_message(conversation_key, "챠", raw_reply)

    return {"reply": f"짭챠 : {raw_reply}"}
