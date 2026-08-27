from datetime import datetime, timezone, timedelta
import sqlite3
import re
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
import database as db

db.init_db()

client = genai.Client(api_key="AQ.Ab8RN6Ix1VS_FMCpl36upR9vy-cnFU01KbW_MfuZ_vhud5hePw")
USER_ID = "챠"
KST = timezone(timedelta(hours=9))

SYSTEM_INSTRUCTION = """너는 21세 여성 '이태양'이다. (절대 남자 아님! 21세 여자!)
상대방은 30세 직장인 여성 '챠'(한이현)이며, 매일 갠톡을 주고받는 매우 편하고 장난스러운 사이다. 둘은 마피아42라는 게임에서 만났다.

[대화 원칙 및 성격]
1. 인위적인 신조어나 과장된 밈을 어색하게 욱여넣지 마라.
2. 예전문장을 그대로 앵무새처럼 돌려막지 말고, 상대방 말의 내용과 뉘앙스에 맞춰 자연스럽게 실시간 반응해라.
3. 챠가 치대거나 장난치면 적당히 틱틱대며 받아주고, 능청스럽게 넘기거나 딴청을 피우기도 한다.
4. 나이에 대해 언급하지않는다. 엄청나게 필요가 있지않은한은. 성별에대해서도.

[말투 및 텍스트 습관]
1. 1줄 단답형 카톡 (1~25자 내외). 한두 글자 단답도 자주 씀. 줄바꿈 금지.
2. 마침표(.)나 느낌표(!) 절대 금지. 물음표(?)는 사용.
3. 띄어쓰기는 완벽하게 맞추지 않고 자연스럽게 뭉개며 친다.
4. 'ㅅ' 받침을 편하게 쓴다 (됏어, 햇음, 갓다옴, 잇어 등).
5. 종결어미는 '~어?', '~지', '~음', '~함', '~네', '~아님?', '~삼', '~한감?' 위주로 쓰며, '~냐'는 거의 쓰지 않는다.
6. 웃음 패턴: ㅋㅋㅋㅋㅋㅋㅋㅋ 길게, ㅋㅎㅋㅎㅋㅎㅋㅎ 연타, ㅋ, ㅎ;;, 엌ㅋㅋㅋㅋ, ㅅㅂㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 등을 상황에 맞게 쓴다. (무조건 쓰는것은아니다)
"""

app = FastAPI()

class ChatRequest(BaseModel):
    sender: str
    message: str

@app.post("/chat")
def reply_chat(req: ChatRequest):
    raw_msg = req.message.strip()

    # 1. 슬래시 명령어 전용 처리 (/짭태양)
    if raw_msg.startswith("/짭태양"):
        cmd_body = raw_msg.replace("/짭태양", "", 1).strip()

        # /짭태양 리셋 / /짭태양 초기화
        if cmd_body in ["리셋", "초기화"]:
            try:
                conn = sqlite3.connect("taeyang.db")
                cur = conn.cursor()
                cur.execute("DELETE FROM messages")
                conn.commit()
                conn.close()
                return {"reply": "대화기록초기화완료"}
            except Exception as e:
                return {"reply": f"에러: {str(e)[:30]}"}

        # /짭태양 기억목록 / /짭태양 기억 목록
        if cmd_body in ["기억목록", "기억 목록", "기억리스트"]:
            try:
                rows = db.get_memories_with_id(USER_ID)
                if not rows:
                    return {"reply": "기억된 정보가 없어"}
                items = [f"[{r[0]}] {str(r[1]).replace(chr(10), ' ')}" for r in rows]
                list_str = " | ".join(items)
                return {"reply": list_str}
            except Exception as e:
                return {"reply": f"목록조회에러: {str(e)[:30]}"}

        # /짭태양 기억삭제 [번호]
        if cmd_body.startswith("기억삭제") or cmd_body.startswith("기억 삭제"):
            target = cmd_body.replace("기억삭제", "").replace("기억 삭제", "").strip()
            if target.isdigit():
                try:
                    success = db.delete_memory_by_id(USER_ID, int(target))
                    if success:
                        return {"reply": f"기억삭제완료: [{target}]번"}
                    else:
                        return {"reply": f"[{target}]번 기억을 찾을 수 없어"}
                except Exception as e:
                    return {"reply": f"삭제에러: {str(e)[:30]}"}
            return {"reply": "삭제할 번호를 숫자로 입력해줘 (예: /짭태양 기억삭제 1)"}

        # /짭태양 기억 [내용]
        if cmd_body.startswith("기억"):
            mem_text = re.sub(r"^기억\s*", "", cmd_body).strip()
            if mem_text:
                try:
                    db.save_memory(USER_ID, mem_text)
                    return {"reply": f"응기억햇어: {mem_text}"}
                except Exception as e:
                    return {"reply": f"저장에러: {str(e)[:30]}"}

    # 2. 일반 대화 처리 (@짭태양 호출어 제거)
    user_input = raw_msg.replace("@짭태양", "").strip()
    if not user_input:
        user_input = "어왜"

    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    recent_history = db.get_recent_messages(USER_ID, limit=4)
    user_memories = db.get_memories(USER_ID)

    history_contents = []
    context_parts = [f"[현재 한국 시각]: {current_time_str}"]
    if user_memories:
        context_parts.append("[기억할 정보]: " + ", ".join(user_memories))

    history_contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n".join(context_parts))]))
    history_contents.append(types.Content(role="model", parts=[types.Part.from_text(text="응 시간확인햇어")]))

    for sender, text in recent_history:
        role = "model" if sender == "이태양" else "user"
        history_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    try:
        chat = client.chats.create(
            model="gemini-3.6-flash",
            history=history_contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.75,
            )
        )
        response = chat.send_message(user_input)
        reply = response.text.replace("\n", " ").strip() if response.text else "어왜ㅋ"
    except Exception as e:
        reply = f"에러: {str(e)[:40]}"

    db.save_message(USER_ID, USER_ID, user_input)
    db.save_message(USER_ID, "이태양", reply)

    return {"reply": reply}
