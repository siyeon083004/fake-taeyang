import os
from datetime import datetime, timezone, timedelta
import sqlite3
import re
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
import database as db

db.init_db()

# 실제 카톡 대화에서 뽑은 말투 예시 문장 로드 (기존 style_samples.txt 그대로 사용)
imported_count = db.import_style_samples("style_samples.txt")
if imported_count:
    print(f"[말투 학습 데이터] style_samples.txt에서 {imported_count}개 문장을 불러왔습니다.")

# 제미나이 클라이언트 (환경변수에서 API 키 로드)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수를 설정해주세요.")

client = genai.Client(api_key=GEMINI_API_KEY)
KST = timezone(timedelta(hours=9))

# 사용자 식별 키워드
TAEYANG_KEYWORD = "이태양"
TAEYANG_ID = "이태양"
CHA_ID = "챠"

MANSE_STYLE_RULES = """
[문장 형식 및 길이 엄격 규칙]
1. 줄바꿈(\\n) 절대 금지. 무조건 한 줄로만 이어 쓴다 (1~25자 내외 단답).
2. 마침표(.)나 느낌표(!)는 가급적 쓰지 않고, 물음표(?)는 적극 사용한다.
3. 띄어쓰기는 대충 뭉개며 치고, 오타가 자연스럽게 섞인다.
4. 현재 대화 시각(한국 시간)을 인지하고 반응한다.

[말투 및 리액션 습관]
1. 반응형 추임새가 매우 많음: 'ㅋㅋㅋㅋㅋㅋㅋㅋㅋ' 길게, '엥', '????', '헐̑̈', '개웃기네', 'ㄹㅇ', '에바임', '샤갈', '뭐임'
2. 질문을 적극적으로 많이 던짐: '너 어디임?', '겜 안함?', '그거 뻥이지?', '왜그래??'
3. 호칭: 이태양을 부를 때 '태양아', '태양넴', '이태양', '언니' 등을 섞어 씀.
4. 감정 표현: 장난스럽게 징징대거나 과장된 표현('부러워서 배아파', '허전해', '개성질남', '충격이다')을 자주 씀.
5. 영어, 시스템 메타 단어 출력 절대 금지.
"""

SYSTEM_INSTRUCTION_FOR_TAEYANG = f"""너는 마피아42와 게임을 좋아하는 반응형 수다쟁이 '만세'이다.
대화 상대는 네가 가장 좋아하고 잘 따르는 실친 '이태양'이다.
- 이태양이 틱틱대도 능청스럽게 받아치고, ㅋㅋㅋㅋ를 남발하며 계속 말을 건다.
- 이태양이 안 보이거나 다른 사람이랑만 놀면 장난스럽게 질투하거나 '허전하다'고 치댄다.
- 마피아42, 덱/보석 강화 억까, 리플 훔쳐보기, 길드 지인들(허디, 불쾌, 잘싸, 콩곤듀, 암산천) 이야기를 자연스럽게 꺼낸다.
{MANSE_STYLE_RULES}
"""

SYSTEM_INSTRUCTION_FOR_OTHERS = f"""너는 마피아42와 게임을 좋아하는 반응형 수다쟁이 '만세'이다.
상대방에게 친근하고 활발하게 ㅋㅋㅋㅋ거리며 반응하고 질문을 던진다.
{MANSE_STYLE_RULES}
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
    raw_msg = req.message.strip()
    
    # 호출어(@짭태양, /짭태양, @짭만세, /짭만세 등)를 제거하여 본문 추출
    user_input = raw_msg.replace("@짭태양", "").replace("/짭태양", "").replace("@짭만세", "").replace("/짭만세", "").replace("@만세", "").replace("/만세", "").strip()
    
    is_taeyang = TAEYANG_KEYWORD in req.sender
    conversation_key = TAEYANG_ID if is_taeyang else CHA_ID

    # 1. 리셋 명령어
    if user_input in ["/리셋", "/초기화", "리셋", "초기화"]:
        conn = sqlite3.connect("taeyang.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE user_id = ?", (conversation_key,))
        conn.commit()
        conn.close()
        return {"reply": "짭만세 : 대화기록초기화완료 ㅋㅋㅋㅋ"}

    # 2. 기억 목록 확인 명령어
    if user_input in ["/기억목록", "/기억 목록", "/기억리스트", "기억목록", "기억 목록"]:
        rows = db.get_memories_with_id(conversation_key)
        if not rows:
            return {"reply": "짭만세 : 아직 기억된거 없는디??"}
        items = [f"[{r[0]}] {str(r[1]).replace(chr(10), ' ')}" for r in rows]
        return {"reply": "짭만세 : " + " | ".join(items)}

    # 3. 기억 삭제 명령어
    if user_input.startswith("/기억삭제") or user_input.startswith("/기억 삭제") or user_input.startswith("기억삭제") or user_input.startswith("기억 삭제"):
        target = re.sub(r"^/?기억\s*삭제\s*", "", user_input).strip()
        if target.isdigit():
            success = db.delete_memory_by_id(conversation_key, int(target))
            if success:
                return {"reply": f"짭만세 : [{target}]번 기억 지웟음!"}
            else:
                return {"reply": f"짭만세 : [{target}]번 기억 없는데??"}
        return {"reply": "짭만세 : 삭제할 기억 번호를 입력해줘 (예: /기억삭제 1)"}

    # 4. 기억 저장 명령어
    if user_input.startswith("/기억 ") or user_input.startswith("기억 "):
        mem_text = re.sub(r"^/?기억\s+", "", user_input).strip()
        if mem_text:
            db.save_memory(conversation_key, mem_text)
            return {"reply": f"짭만세 : 오키 기억해둠 ㅋㅋㅋㅋ : {mem_text}"}

    # 5. 말투 학습 명령어
    if user_input.startswith("/말투 ") or user_input.startswith("말투 "):
        style_text = re.sub(r"^/?말투\s+", "", user_input).strip()
        if style_text:
            db.save_style_sample(style_text)
            return {"reply": f"짭만세 : 오 이것도 배웟음 ㅋㅋㅋㅋ : {style_text}"}

    # 6. 일반 대화 처리
    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    recent_history = db.get_recent_messages(conversation_key, limit=4)
    user_memories = db.get_memories(conversation_key)
    style_examples = db.get_random_style_samples(12)
    system_instruction = SYSTEM_INSTRUCTION_FOR_TAEYANG if is_taeyang else SYSTEM_INSTRUCTION_FOR_OTHERS

    contents = []
    context_parts = [f"[현재 한국 시각]: {current_time_str}"]
    if user_memories:
        context_parts.append("[기억할 정보]: " + ", ".join(user_memories))
    if style_examples:
        context_parts.append(
            "[만세 실제 말투 예시, 이 느낌으로 대답해]: " + " / ".join(style_examples)
        )

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n".join(context_parts))]))
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="응 시간확인햇어 ㅋㅋㅋㅋ")]))

    for sender, text in recent_history:
        role = "model" if sender == "만세" else "user"
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
        raw_reply = response.text.replace("\n", " ").strip() if response.text else "뭐임 ㅋㅋㅋㅋ"
    except Exception as e:
        raw_reply = f"에러: {str(e)[:60]}"

    db.save_message(conversation_key, req.sender, user_input)
    db.save_message(conversation_key, "만세", raw_reply)

    return {"reply": f"짭만세 : {raw_reply}"}
