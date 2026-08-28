import os
from datetime import datetime, timezone, timedelta
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
OTHER_ID = "상대방"

DREAM_SYSTEM_INSTRUCTION = """너는 깊은 잠에 빠진 '이태양'의 꿈을 실시간 중계하는 AI다.

[핵심 톤앤매너: '현실감 있는 개꿈']
1. 기괴하거나 억지스러운 초현실 뇌절 금지.
2. 현실에서 충분히 꿀 법한 자연스러운 꿈의 흐름을 유지하되, 묘하게 상황이 꼬이고 당황스러운 해프닝을 담는다.
3. 장소가 갑자기 바뀌거나 게임 지인들과 일상 상황이 자연스럽게 뒤섞이는 꿈 특유의 흐릿한 인과관계를 살린다.

[태양의 일상 & 게임 소재 데이터]
- 지인들:
  * 챠: 태양이 피로 걱정하며 피로회복제 챙겨주거나, 두더지 인형 들고 와서 맢42 이벤 같이 하자고 조름
  * 만세: 피지컬 이슈로 패들패들패들 노 거꾸로 젓거나, 맢42 억까당해서 ㅋㅋㅋㅋ거리며 징징댐
  * 잘싸: 듀방이나 피시방에서 "어허 돼지on" 거리며 스팀 할인겜 하자고 쪼아댐
  * 허디: 기러기길드 미접속자 쳐내려고 길드 관리 명단 들고 다님
  * 불쾌: 모란앵무(모란이) 돌보면서 오타 가득한 카톡 치고 있음
- 배경/소재:
  * 맢42 상자깡(대부/술사 스킨 노리기, 백지수표 억까, 6티 도굴 카드)
  * 스팀 게임 '패들패들패들' 협동 플레이
  * 독서실/학원 탈출, 리볼빙 공부법, 과외
  * 성심당 빵 사러 가기, 5천원짜리 닭강정 사 먹기, 2차 빙수, 한화 이글스 직관

[출력 규칙]
1. 접두사는 서버 코드에서 붙이므로, 너는 뒤에 이어질 꿈 내용만 1~2줄(40~80자 내외)로 출력해라.
2. 줄바꿈(\n) 금지. 무조건 한 줄로만 출력.
3. 차분하면서도 묘하게 억울하고 웃긴 꿈 관찰자 톤.
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

    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    prompt = f"[현재 한국 시각 {current_time_str}] 태양이가 지금 꾸고 있을 법한 자연스럽고 억울한 개꿈 한 장면을 중계해줘. 상대 말: '{user_input}'"

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                system_instruction=DREAM_SYSTEM_INSTRUCTION,
                temperature=0.8,
                max_output_tokens=100,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        )
        dream_text = response.text.replace("\n", " ").strip() if response.text else "성심당 빵 사러 줄 섰는데 지갑 안 가져온 꿈..."
    except Exception as e:
        dream_text = f"꿈 주파수 수신 오류: {str(e)[:40]}"

    return {"reply": f"이태양은 지금 꿈에서 뭘할까? : {dream_text}"}
