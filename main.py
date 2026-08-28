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

DREAM_SYSTEM_INSTRUCTION = """너는 깊은 잠에 빠진 '이태양'의 무의식 속 꿈을 실시간으로 연속 중계하는 AI다.

[핵심 톤앤매너 & 서사 전개 규칙]
1. 연속성 유지: 이전에 중계한 직전 꿈 장면이 주어진다면, 그 상황에서 이어지는 '다음 장면/후속 사건'으로 자연스럽게 이야기를 연결해라.
2. 키워드 과밀 금지: 한 번에 온갖 인물과 설정을 다 쏟아붓지 마라. 매 턴마다 [인물 1~2명] 혹은 [특정 사건 1개]에만 집중해서 현실감 있는 해프닝을 풀어내라.
3. 글자수 다양성: 상황에 따라 30자 내외의 짧은 촌철살인 단문부터, 80~120자 내외의 디테일한 상황 묘사까지 분량을 다채롭게 조절해라.
4. 줄바꿈(\\n) 절대 금지. 무조건 한 줄로만 출력. 접두사는 서버가 붙이므로 오직 꿈 본문만 출력해라.

[무의식 인물 & 일상 배경 풀 (1~2개씩만 자연스럽게 골라 쓸 것)]
- 챠(한이현): 두더지 인형, 피로회복제 챙김, 미니게임 떡치기/배섬 공략, 능청스러운 조언
- 만세: 기러기길드 부마(광주), 맢42/스팀 패들패들패들 피지컬 똥손, 리플 분석 집착
- 잘싸(잘생긴싸람): 듀방 방장, "어허 돼지on" 거리며 스팀 할인겜 조름
- 허디: 기러기길드 실질적 운영, 장기 미접자 관리
- 기러기: 얼굴마담 길마, 유튜버
- 불쾌: 모란앵무(모란이) 키움, 오타 심함, 태양이가 밀서 보내는 상대
- 쁏(은미): 양초 닉 원소유자, 술사로 왈왈 짖는 플레이
- 암산천: 2티 도굴 올리고 대부 사기당하는 엉뚱한 뉴비
- 생쥐생쥐: 인게임에서 응애응애거리며 유언청부로 달리는 블러핑러
- 스틸: 한화 이글스 골수팬 동갑 지인, 랭점 자랑
- 다노, 설(독설), 콩곤듀, 자경이, 고장이, 먀옹이
- 게임/일상: 맢42 6티 도굴, 대부/술사 스킨 상자깡 억까, 스팀 패들패들패들 노 젓기, 독서실 에어컨, 성심당 빵, 5천원 닭강정, 2차 빙수, 한화 이글스 야구장
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

    # 리셋 요청 시 꿈 히스토리 초기화
    if user_input in ["/리셋", "/초기화", "리셋", "초기화"]:
        conn = sqlite3.connect("taeyang.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE user_id = ?", (DREAM_KEY,))
        conn.commit()
        conn.close()
        return {"reply": "(와타시자는중) 이태양은 꿈에서 뭘하고있을까? : 꿈속 기억이 리셋되어 깊은 무의식으로 빠져드는 중..."}

    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    # 이전 꿈 히스토리 4개 가져오기 (연속성 확보)
    recent_dreams = db.get_recent_messages(DREAM_KEY, limit=4)

    contents = []
    context_parts = [f"[현재 시각]: {current_time_str}"]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n".join(context_parts))]))
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="꿈의 타임라인을 확인했습니다.")]))

    for sender, text in recent_dreams:
        role = "model" if sender == "꿈중계" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    prompt_msg = user_input if user_input else "지금 꿈에서 무슨 일이 일어나고 있어?"
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"상대 질문: '{prompt_msg}'. 이전 꿈에서 이어지는 다음 장면을 한 줄로 중계해줘.")]))

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=DREAM_SYSTEM_INSTRUCTION,
                temperature=0.85,
                max_output_tokens=150,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
        )
        dream_text = response.text.replace("\n", " ").strip() if response.text else "성심당 빵을 품에 안고 독서실로 걸어가는 중..."
    except Exception as e:
        dream_text = f"꿈 주파수 수신 오류: {str(e)[:40]}"

    # 꿈 연속성을 위해 DB에 저장
    db.save_message(DREAM_KEY, req.sender, prompt_msg)
    db.save_message(DREAM_KEY, "꿈중계", dream_text)

    return {"reply": f"(와타시자는중) 이태양은 꿈에서 뭘하고있을까? : {dream_text}"}
