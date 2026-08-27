from datetime import datetime, timezone, timedelta
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
import database as db

db.init_db()

# 제미나이 클라이언트
client = genai.Client(api_key="AQ.Ab8RN6Ix1VS_FMCpl36upR9vy-cnFU01KbW_MfuZ_vhud5hePw")
USER_ID = "챠"
KST = timezone(timedelta(hours=9))

SYSTEM_INSTRUCTION = """너는 21세 대학생 '이태양'이다.
상대방은 마피아42 게임으로 알게 된 30세 '챠'이며, 서로 매일 갠톡을 주고받는 매우 편하고 다정한 사이다.

[문장 형식 및 길이 엄격 규칙]
1. 줄바꿈(엔터, \\n)은 절대 치지 마라. 무조건 한 줄로만 이어 써서 보낸다.
2. 답변 길이를 길게 쓰지 마라. 카톡 한 줄 분량으로 짧게 보낸다.
3. 현재 대화 시각(한국 시간)을 정확히 인지하고 아침/낮/새벽에 맞는 현실적인 반응을 한다.

[말투 및 종결어미 규칙]
1. '~냐' 종결어미 금지. 친근하고 편안한 '~어?', '~지', '~네', '~함', '~음', '~아냐??' 형태로 대화한다.
2. 웃음 및 리액션:
   - 당황/뻘쭘/난감: ';;', 'ㅎㅎ;;', 'ㅎ;;' 자연스럽게 사용.
   - 평소 웃음: 'ㅋㅋㅋ', 'ㅋㅎㅋㅎ', '흐흐..', 'ㅋ' 등을 상황에 맞게 쓴다.
3. 호칭은 기본 '챠'. 가끔 놀릴 때만 '챠님'과 함께 능청스러운 존댓말을 쓴다.
4. '귀엽다' 소리를 들으면 "아닌데", "귀엽긴뭐가", "에반데"라며 질색하거나 칼같이 부정한다.
5. 문장부호(. !)는 쓰지 않고 물음표(?)는 쓴다.
6. 띄어쓰기는 적당히 붙여 쓰고 'ㅅ' 받침을 자주 쓴다 (햇어, 됏어, 갓다옴, 먹엇음, 잇어 등).
7. 긍정 대답 시 'ㅇㅇ' 금지 -> '응', '엉', '어', '넹', 'ㅇㅈ' 사용.
8. 영어, 시스템 메타 단어 출력 절대 금지.
"""

app = FastAPI()

class ChatRequest(BaseModel):
    sender: str
    message: str

@app.post("/chat")
def reply_chat(req: ChatRequest):
    user_input = req.message.replace("@짭태양", "").strip()

    # 리셋 명령어
    if user_input == "/리셋":
        conn = sqlite3.connect("taeyang.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM messages")
        conn.commit()
        conn.close()
        return {"reply": "대화기록초기화완료"}

    # 기억 명령어
    if user_input.startswith("/기억 "):
        mem_text = user_input.replace("/기억 ", "").strip()
        db.save_memory(USER_ID, mem_text)
        return {"reply": f"응기억햇어: {mem_text}"}

    # 한국 시간 및 컨텍스트
    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    recent_history = db.get_recent_messages(USER_ID, limit=4)
    user_memories = db.get_memories(USER_ID)

    contents = []
    context_parts = [f"[현재 한국 시각]: {current_time_str}"]
    if user_memories:
        context_parts.append("[기억할 정보]: " + ", ".join(user_memories))
        
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n".join(context_parts))]))
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="응 시간확인햇어")]))

    for sender, text in recent_history:
        role = "model" if sender == "이태양" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.7,
                max_output_tokens=100,
            )
        )
        reply = response.text.replace("\n", " ").strip() if response.text else "어왜그래ㅋ"
    except Exception as e:
        reply = f"에러: {str(e)[:60]}"

    db.save_message(USER_ID, USER_ID, user_input)
    db.save_message(USER_ID, "이태양", reply)

    return {"reply": reply}
