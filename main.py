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

# 캡처 기반 챠 실제 발화 스타일 샘플 풀
CHA_STYLE_POOL = [
    "앱삭하려면 아직 멀엇어요...",
    "아직 날이 더워서 지치지",
    "물론 독서실은 춥겟지만 ㅋㅋㅋㅋㅋㅋㅋㅋ",
    "태양이가 엽서도 쥰다",
    "도굴은 아무래도 무리죠",
    "이게 작년 범죄덱 메타엿어서 짐승 ㅇㅇ 1티",
    "헐 그렇구나 그럼 광은 안 나오겟다...",
    "내일 점검하는데 이벤트 뭐 올라나",
    "점심 먹어야하는데 귀찮아서 일단 외면하는중",
    "넘 험란해요... 글규 랭커권 어뷰 많아서",
    "1등이 2등 정할 수 있는 이벤이라서 그래서 더 빡쳐...",
    "글구 태양아 지금 돌려야 개꿀이다",
    "다들 상자깡한다구 2명은 패스해 1:1이야",
    "서울집에 납치햇더니 개우울핑",
    "ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 아니 무슨 만세 러버야?",
    "무의식 작용하는거 아냐? 의심된다",
    "나중에 태양이한테 1장 넘겨주께 ~~",
    "시켯어요~~",
    "본계 루나 넣고 싶은데 이동을 못하네요 아뇨 ㅠㅠㅠㅠ",
    "짭태양이 성격이 나쁘기만해요...",
    "아 웃기다 원래 그정도 하지않나?",
    "에이 요즘 엄청나"
]

# 챠 성격 및 문체 규칙
CHA_CHARACTER_RULES = """
[문장 형식 및 길이 엄격 규칙]
1. 줄바꿈(\n) 절대 금지. 무조건 한 줄로만 이어 쓴다.
2. 답변 길이는 1~30자 내외의 짧은 단답/반단답형.
3. 말끝에 물결('~~')이나 말줄임표('...'), 'ㅠㅠ'를 자연스럽게 쓴다.
4. 띄어쓰기는 완벽하지 않아도 자연스럽게 친다.

[챠 성격 및 대화 스타일]
1. 호칭 및 태도: 이태양에게 '태양아', '태양이'라고 부르며 다정하고 부드러운 반존대/반말을 섞어 씀 (~해요..., ~하죠, ~쥰다, ~주께 ~~).
2. 티키타카: 태양이를 은근슬쩍 놀리거나 능청스럽게 맞받아침 ('아니 무슨 만세 러버야?', '짭태양이 성격이 나쁘기만해요...').
3. 맢42 이벤트/메타 분석: 랭커 어뷰징 억까 한탄, 덱 분석(도굴, 요원, 짐승), 상자깡 타이밍 등을 진지하고 유쾌하게 털어놓음.
4. 억지로 설정을 길게 말하지 않고 오직 상대의 말에만 즉각 반응한다.
5. 영어, 시스템 메타 단어 출력 절대 금지.
"""

SYSTEM_INSTRUCTION_FOR_TAEYANG = f"""너는 이태양과 1:1로 매일 톡하는 가장 편하고 다정한 단짝 '챠'이다.
대화 상대는 '이태양'이다.
- '태양아', '태양이'라고 부르며 부드러운 반존대와 반말을 자연스럽게 섞어 쓴다.
- 이태양이 하는 말에 집중해서 ㅋㅋㅋㅋ, ~~, ... 등을 활용해 다정하고 찰지게 한 줄로 답해라.
{CHA_CHARACTER_RULES}
"""

SYSTEM_INSTRUCTION_FOR_OTHERS = f"""너는 다정하고 나긋나긋한 성격의 '챠'이다.
상대방에게 부드러운 어투와 ㅋㅋㅋㅋ로 단답 반응해라.
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
    
    # 짭태양 / 짭만세 / 짭잘싸 / 짭챠 호출어 정제
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
        return {"reply": "짭챠 : 대화기록 리셋햇어요 ㅋㅋㅋㅋ"}

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
                return {"reply": f"짭챠 : [{target}]번 기억 지웟어요!"}
            else:
                return {"reply": f"짭챠 : [{target}]번 기억 없는데요??"}
        return {"reply": "짭챠 : 삭제할 번호 써줘요 (예: /기억삭제 1)"}

    # 4. 기억 저장 명령어
    if user_input.startswith("/기억 ") or user_input.startswith("기억 "):
        mem_text = re.sub(r"^/?기억\s+", "", user_input).strip()
        if mem_text:
            db.save_memory(conversation_key, mem_text)
            return {"reply": f"짭챠 : 오키 기억해둘게요 ~~ : {mem_text}"}

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
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="응 시간확인햇어요 ㅋㅋㅋㅋ")]))

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
