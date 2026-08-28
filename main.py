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

[창작 자유도 및 톤앤매너]
1. 제약 없는 상상력: 로블록스 맵, 무한한 노란 벽의 백룸(Backrooms), 우주, 사이버펑크, 이세계 등 온갖 기괴하고 황당한 배경을 자유롭게 투입해라.
2. 자연스러운 결합: 태양의 일상/게임 요소와 엉뚱한 배경을 자유자재로 버무려 생생한 꿈 장면을 만들어라.
3. 연속성 유지: 이전 꿈 내용이 있다면 그 상황에서 이어지는 다음 행동이나 엉뚱한 사건으로 자연스럽게 전개해라.
4. 문장 형식: 반드시 '~하는 중입니다.' 또는 '~하고 있습니다.'로 끝맺어라.
5. 길이: 공백 포함 10~50자 내외의 깔끔한 한 줄 단문.
6. 오직 꿈 본문 내용만 출력해라. (접두사는 서버가 붙임)

[추가된 핵심 무의식 키워드 창고 (자유롭게 1~2개 조합)]
- 로블록스 & 백룸: 로블록스 점프맵(오비), 도어즈(DOORS), 타워오브헬, 끝없는 백룸(노란 벽지/형광등 소리/엔티티 탈출)
- 마피아42 심화 용어: 확직, 홀경, 맞경, 첫맢, 꽁승, 유언청부, 직공, 계망(계약망령), 맢블(마피아 블러핑), 특직, 밤투표, 엽서 테러(마엽/깜엽), 랭포 억까, 접막, 신분세탁
- 지인/인물: 챠(두더지 인형, 피로회복제), 만세(패들패들패들 노 거꾸로 젓기, 피지컬 똥손), 잘싸(돼지on, 스팀 할인겜 조르기), 허디(미접 관리), 불쾌(모란앵무, 오타), 쁏(술사 왈왈), 암산천(2티 도굴 사기), 생쥐생쥐(응애응애 유언청부), 스틸(한화팬)
- 일상/배경: 독서실 에어컨 추위, 성심당 빵, 5천원 닭강정, 2차 빙수, 한화 이글스 야구장
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
        return {"reply": "(와타시자는중) 이태양은 꿈에서 뭘하고있을까? :\n새로운 무의식 차원으로 진입하는 중입니다."}

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

    prompt_msg = user_input if user_input else "지금 꿈에서 뭘 하고 있어?"
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"상대 질문: '{prompt_msg}'. 이전 꿈에서 이어지는 기발한 다음 장면을 ~하는 중입니다/하고 있습니다 형식의 10~50자 단문으로 중계해줘.")]))

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
        dream_text = response.text.replace("\n", " ").strip() if response.text else "백룸 0레벨 노란 벽지 사이에서 맢42 홀경을 주장하는 중입니다."
    except Exception as e:
        dream_text = f"꿈 수신 오류 발생 중입니다: {str(e)[:20]}"

    # 꿈 연속성 저장
    db.save_message(DREAM_KEY, req.sender, prompt_msg)
    db.save_message(DREAM_KEY, "꿈중계", dream_text)

    return {"reply": f"(와타시자는중) 이태양은 꿈에서 뭘하고있을까? :\n{dream_text}"}

