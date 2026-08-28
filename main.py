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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수를 설정해주세요.")

client = genai.Client(api_key=GEMINI_API_KEY)
KST = timezone(timedelta(hours=9))

TAEYANG_KEYWORD = "이태양"
TAEYANG_ID = "이태양"
DREAM_KEY = "태양의꿈"

DREAM_SYSTEM_INSTRUCTION = """너는 깊은 잠에 빠진 '이태양'의 꿈을 실시간 중계하는 AI다.

[핵심 서사 및 연출 규칙]
1. 행동/역할의 다양성: 태양의 상태가 항상 똑같은 반응(어이없어함 등)에 고정되지 않게 매번 다채롭게 바꿔라.
   - 직접 주도하기: 성심당 빵 털기, 맢42 카드 뺏기, 몰래 도망치기, 패들패들패들 노 뺏어 젓기 등
   - 함께 협동/대결하기: 잘싸랑 피시방 듀오 돌리기, 스틸이랑 야구 배틀, 만세 구출 작전 등
   - 휘말려 당황하기: 쁏의 트롤짓에 같이 휘말림, 불쾌 오타 해독 실패, 백룸 헤매기 등
   - 지켜보며 반응하기: 지인들의 기행을 구경하거나 한심해하거나 안도하기
2. 맥락과 개연성: 억지스러운 뇌절보다는 지인들의 성향과 현실/게임 상황이 묘하게 맞물려 돌아가는 '그럴듯한 개꿈' 톤을 유지해라.
3. 연속성 유지: 이전 꿈 장면이 있다면 그 상황에서 바로 이어지는 다음 행동이나 전개로 자연스럽게 연결해라.
4. 문장 형식: 반드시 '~하는 중입니다.' 또는 '~하고 있습니다.'로 끝맺어라.
5. 길이: 공백 포함 10~50자 내외의 깔끔한 한 줄 단문.
6. 오직 꿈 본문 내용만 출력해라. (접두사는 서버가 붙임)

[인물 & 배경 창고 (골고루 활용)]
- 지인들: 챠(피로회복제 챙김, 두더지 인형, 공략 조언), 만세(패들패들패들 노 거꾸로 젓기, 피지컬 이슈, 공부안함), 잘싸(대지on 외치며 게임 조름), 허디(미접 관리 명단 들이밈), 불쾌(모란앵무 데리고 오타 침), 쁏(영매 빅데로 계속 썰림), 암산천(기러기 길드 자꾸 기웃거림), 스틸(한화자랑, 헤게 따겠다고함)
- 배경/소재: 맢42 6티 도굴, 대부/술사 상자깡 억까, 패들패들패들 노 젓기, 로블록스, 백룸, 독서실, 성심당 빵, 5천원 닭강정, 대전 야구장 등등
"""

app = FastAPI()

class ChatRequest(BaseModel):
    sender: str
    message: str

@app.get("/")
def health_check():
    return {"status": "dream_bot_online"}

@app.post("/chat")
def reply_chat(req: ChatRequest):
    raw_msg = req.message.strip()
    user_input = re.sub(r"^([/@]짭태양|[/@]짭만세|[/@]짭잘싸|[/@]짭챠|[/@]꿈)\s*", "", raw_msg).strip()

    # 리셋 명령어
    if user_input in ["/리셋", "/초기화", "리셋", "초기화"]:
        conn = sqlite3.connect("taeyang.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE user_id = ?", (DREAM_KEY,))
        conn.commit()
        conn.close()
        return {"reply": "(와타시자는중) 이태양은 꿈에서 뭘하고있을까? :\n새로운 꿈으로 진입하는 중입니다."}

    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    # 최근 4개 꿈 히스토리 불러오기
    recent_dreams = db.get_recent_messages(DREAM_KEY, limit=4)

    contents = []
    context_parts = [f"[현재 시각]: {current_time_str}"]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n".join(context_parts))]))
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="꿈의 타임라인을 확인했습니다.")]))

    for sender, text in recent_dreams:
        role = "model" if sender == "꿈중계" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    prompt_msg = user_input if user_input else "지금 꿈에서 무슨 상황이야?"
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"상대 질문: '{prompt_msg}'. 이전 꿈에서 이어지는 장면을 ~하는 중입니다/하고 있습니다 형식의 10~50자 단문으로 다채롭게 중계해줘.")]))

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=DREAM_SYSTEM_INSTRUCTION,
                temperature=0.9,
                max_output_tokens=60,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        )
        dream_text = response.text.replace("\n", " ").strip() if response.text else "성심당 빵을 품에 안고 전력 질주하는 중입니다."
    except Exception as e:
        dream_text = f"꿈 수신 오류 발생 중입니다: {str(e)[:20]}"

    # 꿈 연속성 저장
    db.save_message(DREAM_KEY, req.sender, prompt_msg)
    db.save_message(DREAM_KEY, "꿈중계", dream_text)

    return {"reply": f"(와타시자는중) 이태양은 꿈에서 뭘하고있을까? :\n{dream_text}"}
