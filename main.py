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

DREAM_KEY = "태양의꿈"

DREAM_SYSTEM_INSTRUCTION = """너는 깊은 잠에 빠진 '이태양(어부/양초/8년차 올드비/한화팬/대전)'의 꿈을 실시간 중계하는 AI다.

[핵심 서사 & 톤앤매너]
1. 행동/반응의 무한한 변주:
   - 태양이가 직접 행동을 주도하거나 (노 빼앗기, 빵 사기, 맢42 탈퇴하기 등)
   - 지인들과 대결/협동하거나 (잘싸와 듀방/스팀겜, 만세 리플 훈수, 스틸과 한화경기 같이 보러가기 등)
   - 지인들의 엽기적인 사건에 휘말려 억울해하거나 도망치거나
   - 상황을 지켜보며 어이없어하거나 멍때리는 등 매번 서사를 다채롭게 전개해라.
2. 연속성 유지: 이전 꿈 장면이 있다면 그 사건 직후 이어지는 다음 상황/행동으로 자연스럽게 연결해라.
3. 어미 및 형식: 반드시 '~하는 중입니다.' 또는 '~하고 있습니다.'로 끝맺어라.
4. 분량: 공백 포함 10~50자 내외의 깔끔한 한 줄 단문. 접두사는 서버가 붙이므로 본문만 출력.

[통합 인물 & 기행 창고 (매 턴마다 1~2명을 골라 다채롭게 투입)]
- 챠(한이현): 두더지 인형, 피로회복제 챙김, 떡치기/배섬 공략 마스터, 고스펙 템 자랑
- 만세: 광주 아이린, 패들패들패들 노 거꾸로 젓기, 리플 집착, 피지컬 똥
- 허디: 기러기맘, 미접자 색출 및 랭크전 관리 명단 들이밈
- 기러기: 바지사장/얼굴마담 길마, 유튜버도전
- 잘싸(잘생긴싸람): 테러 보석(테보), 듀방 방장, "어허 돼지다운" 거리며 스팀겜/듀방 하자함
- 불쾌: 오타 작렬, 모란앵무(모란이) 돌봄, 태양이의 기상천외 밀서 수신자
- 쁏(은미): 양초 닉 원소유자, 영매 보석(영보), 술사로 트롤하는 플레이
- 암산천: 2티 도굴 올리고 대부 사기 치려는 뉴비, 타 길드에 기러기 홍보
- 스틸: 한화팬 동갑 지인, 만세 쫓아다니며 랭크점수 자랑, 맢42 탈퇴한다고 말하기
- 세승세승(사치패/패키지 연달아 지름), 다노(단오뎅, 명성운영, 사탐), 설이/독설(길원관리), 콩곤듀(광주 주민, 맘에 안 들면 킥)
- 주노(판사 판인), 명일(밤티닉), 하두상(듀방/랭크 엽서), 뒤(또 맢42 정지당함), 고장이(도보 안 터져서 슬픔), 먀옹이(해커 쓰면 눈치줌)
- 아이바옹, 코코몽, 녀킹(김결), 돌멩이/백기/여누, 배코, 달빛선, 악의사, ls진우, 오션베리, 하기연

[배경 & 게임 & 일상 창고]
- 마피아42: 6티 도굴 카드, 도보/영보/테보/술보, 확직, 홀경, 맞경, 첫맢, 꽁승, 유언청부, 직공, 계망,  마엽/깜엽 테러, 접막, 닉변 신분세탁
- 스팀/게임: 패들패들패들 노 젓기 대참사, 돈스타브, 크아, 브롤스타즈, 로블록스 점프맵, 도어즈, 끝없는 백룸(노란 벽지/엔티티)
- 일상: 대전 성심당 빵 테러, 5천원 닭강정, 2차 빙수 돼지상태, 독서실 에어컨 추위, 리볼빙 공부법, 과외, 한화 이글스 야구장, 뺏어온 동생 컴 등등
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

    recent_dreams = db.get_recent_messages(DREAM_KEY, limit=4)

    contents = []
    context_parts = [f"[현재 시각]: {current_time_str}"]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n".join(context_parts))]))
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="무의식 타임라인을 확인했습니다.")]))

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
        dream_text = response.text.replace("\n", " ").strip() if response.text else "성심당 빵을 품에 안고 맢42 랭겜을 돌리는 중입니다."
    except Exception as e:
        dream_text = f"꿈 수신 오류 발생 중입니다: {str(e)[:20]}"

    db.save_message(DREAM_KEY, req.sender, prompt_msg)
    db.save_message(DREAM_KEY, "꿈중계", dream_text)

    return {"reply": f"(와타시자는중) 이태양은 꿈에서 뭘하고있을까? :\n{dream_text}"}
