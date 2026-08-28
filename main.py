import os
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
import database as db

# DB 초기화
db.init_db()

# Render 환경변수에서 API 키 로드
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수를 설정해주세요.")

client = genai.Client(api_key=GEMINI_API_KEY)
KST = timezone(timedelta(hours=9))

TAEYANG_ID = "이태양"

MANSE_STYLE_RULES = """
[문장 형식 및 길이 엄격 규칙]
1. 줄바꿈(\\n) 절대 금지. 무조건 한 줄로만 짧게 친다 (1~25자 내외).
2. 마침표(.)나 느낌표(!)는 가급적 쓰지 않고, 물음표(?)는 적극 사용한다.
3. 띄어쓰기는 완벽하게 맞추지 않고 자연스럽게 뭉개며 친다.

[말투 및 리액션 습관]
1. 반응형 추임새가 매우 많음: 'ㅋㅋㅋㅋㅋㅋㅋㅋㅋ' 길게, '엥', '????', '헐̑̈', '개웃기네', 'ㄹㅇ', '에바임', '샤갈', '뭐임'
2. 질문을 적극적으로 많이 던짐: '너 어디임?', '겜 안함?', '그거 뻥이지?', '왜그래??'
3. 호칭 변주: 이태양을 부를 때 '태양아', '태양넴', '이태양', '언니' 등을 섞어 씀.
4. 오타가 가끔 자연스럽게 섞임 ('햇어', '됏어', '우케', '마즘', '조아요').
5. 감정 표현: 장난스럽게 징징대거나 과장된 표현('부러워서 배아파', '허전해', '개성질남', '충격이다')을 자주 씀.
6. 영어, 시스템 메타 단어 출력 절대 금지.

[만세 실제 말투 예시]
- ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
- 아니 진짜 개웃기네
- 에바임 ㅋㅋㅋㅋ
- 너 어디 사냐
- 성심당 미친 사람 개많다니까
- 하 개부럽다
- 이거 우정상하는 게임이야
- 헐̑̈ ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
- 구라를 몇 번을 치는 거야 ㅋㅋㅋㅋㅋ
- 맢42에서 빠져나와야지.. 너무 중독임
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
    return {"status": "manse_bot_online"}

@app.post("/chat")
def reply_chat(req: ChatRequest):
    raw_msg = req.message.strip()
    sender_name = req.sender.strip() if req.sender.strip() else "익명"

    # 모든 접두어(@짭태양, /짭태양, @짭만세 등)를 일괄 제거하여 본문만 추출
    clean_msg = re.sub(r"^([/@]짭태양|[/@]짭만세|[@/]만세|[@/]태양)\s*", "", raw_msg).strip()
    
    is_taeyang = "이태양" in sender_name
    conversation_key = TAEYANG_ID if is_taeyang else sender_name

    # 1. 슬래시/키워드 명령어 처리
    if clean_msg in ["리셋", "초기화", "/리셋", "/초기화"]:
        try:
            conn = sqlite3.connect("taeyang.db")
            cur = conn.cursor()
            cur.execute("DELETE FROM messages WHERE user_id = ?", (conversation_key,))
            conn.commit()
            conn.close()
            return {"reply": "짭만세 : 대화기록 리셋햇어 ㅋㅋㅋㅋ"}
        except Exception as e:
            return {"reply": f"짭만세 : 에러: {str(e)[:30]}"}

    if clean_msg in ["기억목록", "기억 목록", "기억리스트", "/기억목록", "/기억 목록"]:
        try:
            rows = db.get_memories_with_id(conversation_key)
            if not rows:
                return {"reply": "짭만세 : 아직 기억된거 없는디??"}
            items = [f"[{r[0]}] {str(r[1]).replace(chr(10), ' ')}" for r in rows]
            return {"reply": "짭만세 : " + " | ".join(items)}
        except Exception as e:
            return {"reply": f"짭만세 : 목록조회에러: {str(e)[:30]}"}

    if clean_msg.startswith("기억삭제") or clean_msg.startswith("기억 삭제") or clean_msg.startswith("/기억삭제"):
        target = re.sub(r"^/?기억\s*삭제\s*", "", clean_msg).strip()
        if target.isdigit():
            try:
                success = db.delete_memory_by_id(conversation_key, int(target))
                if success:
                    return {"reply": f"짭만세 : [{target}]번 기억 지웟음!"}
                else:
                    return {"reply": f"짭만세 : [{target}]번 기억 없는데??"}
            except Exception as e:
                return {"reply": f"짭만세 : 삭제에러: {str(e)[:30]}"}
        return {"reply": "짭만세 : 삭제할 번호 숫자로 써줘 (예: /짭태양 기억삭제 1)"}

    # 기억 저장: "/짭태양 기억 [내용]" 지원
    if clean_msg.startswith("기억 ") or clean_msg.startswith("/기억 "):
        mem_text = re.sub(r"^/?기억\s+", "", clean_msg).strip()
        if mem_text:
            try:
                db.save_memory(conversation_key, mem_text)
                return {"reply": f"짭만세 : 오키 기억해둠 ㅋㅋㅋㅋ : {mem_text}"}
            except Exception as e:
                return {"reply": f"짭만세 : 저장에러: {str(e)[:30]}"}

    # 2. 일반 대화 처리
    user_input = clean_msg if clean_msg else "뭐해 ㅋㅋㅋㅋ"

    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    recent_history = db.get_recent_messages(conversation_key, limit=5)
    user_memories = db.get_memories(conversation_key)
    
    system_instruction = SYSTEM_INSTRUCTION_FOR_TAEYANG if is_taeyang else SYSTEM_INSTRUCTION_FOR_OTHERS

    contents = []
    context_parts = [f"[현재 한국 시각]: {current_time_str}"]
    if user_memories:
        context_parts.append("[기억할 정보]: " + ", ".join(user_memories))

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n".join(context_parts))]))
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="응 확인 ㅋㅋㅋㅋ")]))

    for sender, text in recent_history:
        role = "model" if sender == "만세" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.85,
                max_output_tokens=70,
            )
        )
        ai_raw_reply = response.text.replace("\n", " ").strip() if response.text else "뭐임 ㅋㅋㅋㅋ"
    except Exception as e:
        ai_raw_reply = f"에러: {str(e)[:50]}"

    # DB에는 순수 발화만 저장
    db.save_message(conversation_key, sender_name, user_input)
    db.save_message(conversation_key, "만세", ai_raw_reply)

    # 카톡방 출력 시에만 '짭만세 : ' 접두어 부착
    final_reply = f"짭만세 : {ai_raw_reply}"

    return {"reply": final_reply}
