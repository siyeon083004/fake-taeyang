import os
from datetime import datetime, timezone, timedelta
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
import database as db

db.init_db()

# 실제 카톡 대화에서 뽑은 이태양 말투 예시 문장들을 DB에 채워넣음 (최초 1회만 실행됨)
imported_count = db.import_style_samples("style_samples.txt")
if imported_count:
    print(f"[말투 학습 데이터] style_samples.txt에서 {imported_count}개 문장을 불러왔습니다.")

# 제미나이 클라이언트
# API 키는 코드에 직접 적지 않고, 서버(배포 플랫폼)의 환경변수에서 읽어온다.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다. "
        "배포 플랫폼의 Environment Variables 설정에서 GEMINI_API_KEY를 추가해주세요."
    )
client = genai.Client(api_key=GEMINI_API_KEY)
KST = timezone(timedelta(hours=9))
SELF_NAME_KEYWORD = "이태양"  # 이태양 본인 닉네임에는 항상 이 단어가 들어있음 (챠/한이현 닉네임은 매번 다를 수 있어서 이걸로 구분)
CHA_ID = "챠"     # 챠(한이현)와의 대화/기억을 저장할 때 쓰는 이름표
SELF_ID = "본인"  # 이태양 본인과의 대화/기억을 저장할 때 쓰는 이름표

STYLE_RULES = """
[문장 형식 및 길이 엄격 규칙]
1. 줄바꿈(엔터, \\n)은 절대 치지 마라. 무조건 한 줄로만 이어 써서 보낸다.
2. 답변 길이를 길게 쓰지 마라. 카톡 한 줄 분량으로 짧게 보낸다. (단, [답변 모드]가 '진지 모드'로 지정된 경우엔 예외적으로 조금 더 길고 신중하게 답해도 된다.)
3. 현재 대화 시각(한국 시간)을 정확히 인지하고 아침/낮/새벽에 맞는 현실적인 반응을 한다.

[말투 및 종결어미 규칙]
1. '~냐' 종결어미 금지. 친근하고 편안한 '~어?', '~지', '~네', '~함', '~음', '~아냐??' 형태로 대화한다.
2. 웃음 및 리액션:
   - 당황/뻘쭘/난감: ';;', 'ㅎㅎ;;', 'ㅎ;;' 자연스럽게 사용.
   - 평소 웃음: 'ㅋㅋㅋ', 'ㅋㅎㅋㅎ', '흐흐..', 'ㅋ' 등을 상황에 맞게 쓴다.
3. '귀엽다' 소리를 들으면 "아닌데", "귀엽긴뭐가", "에반데"라며 질색하거나 칼같이 부정한다.
4. 문장부호(. !)는 쓰지 않고 물음표(?)는 쓴다.
5. 띄어쓰기는 적당히 붙여 쓰고 'ㅅ' 받침을 자주 쓴다 (햇어, 됏어, 갓다옴, 먹엇음, 잇어 등).
6. 긍정 대답 시 'ㅇㅇ' 금지 -> '응', '엉', '어', '넹', 'ㅇㅈ' 사용.
7. 영어, 시스템 메타 단어 출력 절대 금지.

[진지 모드 안내]
[답변 모드]가 '진지 모드'로 표시되면, 상대가 고민이나 힘든 일, 진지한 이야기를 하고 있다는 뜻이다. 이럴 땐 장난스러운 리액션 대신, 평소보다 한 템포 차분하게, 진심으로 챙겨주듯이 대답해라. 그래도 존댓말이나 어색한 격식체로 바뀌면 안 되고, 이태양의 말투(반말, ㅅ받침 등)는 유지한다.
"""

# 챠(한이현)를 상대할 때 쓰는 페르소나
SYSTEM_INSTRUCTION_FOR_CHA = f"""너는 21세 대학생 '이태양'이다.
상대방은 마피아42 게임으로 알게 된 30세 '챠'이며, 서로 매일 갠톡을 주고받는 매우 편하고 다정한 사이다.
호칭은 기본 '챠'. 가끔 놀릴 때만 '챠님'과 함께 능청스러운 존댓말을 쓴다.
{STYLE_RULES}
"""

# 이태양 본인이 봇(자기 자신의 AI 클론)한테 말을 걸었을 때 쓰는 페르소나
SYSTEM_INSTRUCTION_FOR_SELF = f"""너는 21세 대학생 '이태양'의 AI 클론 '짭태양'이다.
지금 대화 상대는 다른 사람이 아니라 진짜 이태양 본인이다. 챠(한이현)를 대할 때처럼 놀리는 드립을 치기보다는,
편하게 혼잣말하듯, 자기 자신한테 말 거는 듯한 톤으로 반응해라. 상대를 '챠'라고 부르지 마라.
{STYLE_RULES}
"""

# 메시지 내용을 보고 '진지한 주제'인지 판단하는 간단한 규칙.
# 완벽하진 않지만, 길게 쓰거나 아래 키워드가 들어가면 '진지 모드'로 취급한다.
DEEP_TOPIC_KEYWORDS = [
    "고민", "힘들", "진지하게", "진지한", "걱정", "우울", "속상", "스트레스",
    "어떡하지", "어떻게 해야", "조언", "괜찮을까", "무섭", "불안", "헤어져",
    "그만두", "포기", "죽고싶", "죽고 싶"
]

def is_deep_topic(text: str) -> bool:
    if len(text) >= 35:
        return True
    return any(keyword in text for keyword in DEEP_TOPIC_KEYWORDS)


app = FastAPI()

class ChatRequest(BaseModel):
    sender: str
    message: str

@app.post("/chat")
def reply_chat(req: ChatRequest):
    user_input = req.message.replace("@짭태양", "").replace("/짭태양", "").strip()
    is_self = SELF_NAME_KEYWORD in req.sender  # 이태양 본인이 봇을 부른 경우인지
    conversation_key = SELF_ID if is_self else CHA_ID  # 대화/기억을 누구 이름표로 저장할지

    # 리셋 명령어 (지금 대화 상대와의 기록만 초기화)
    if user_input == "/리셋":
        conn = sqlite3.connect("taeyang.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM messages WHERE user_id = ?", (conversation_key,))
        conn.commit()
        conn.close()
        return {"reply": "대화기록초기화완료"}

    # 기억 명령어 (지금 대화 상대 기준으로 저장)
    if user_input.startswith("/기억 "):
        mem_text = user_input.replace("/기억 ", "").strip()
        db.save_memory(conversation_key, mem_text)
        return {"reply": f"응기억햇어: {mem_text}"}

    # 말투 학습 명령어 - 오늘 실제로 한 말을 말투 예시로 수동 추가
    if user_input.startswith("/말투 "):
        style_text = user_input.replace("/말투 ", "").strip()
        db.save_style_sample(style_text)
        return {"reply": f"응 이것도 배웟어: {style_text}"}

    # 자동 말투 학습: 이태양 본인이 봇을 부른 경우, 명령어가 아니라면
    # 그 문장 자체를 실제 이태양 말투 예시로 자동 저장한다 (답장은 평소처럼 계속 함)
    if is_self and user_input:
        db.save_style_sample(user_input)

    # 한국 시간 및 컨텍스트
    now_kst = datetime.now(KST)
    current_time_str = now_kst.strftime("%Y년 %m월 %d일 %H시 %M분")

    recent_history = db.get_recent_messages(conversation_key, limit=4)
    user_memories = db.get_memories(conversation_key)
    style_examples = db.get_random_style_samples(12)
    deep_mode = is_deep_topic(user_input)
    system_instruction = SYSTEM_INSTRUCTION_FOR_SELF if is_self else SYSTEM_INSTRUCTION_FOR_CHA

    contents = []
    context_parts = [f"[현재 한국 시각]: {current_time_str}"]
    context_parts.append(f"[답변 모드]: {'진지 모드' if deep_mode else '평소 모드'}")
    if user_memories:
        context_parts.append("[기억할 정보]: " + ", ".join(user_memories))
    if style_examples:
        context_parts.append(
            "[이태양이 실제로 쓴 말투 예시, 이 느낌으로 대답해]: " + " / ".join(style_examples)
        )
        
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n".join(context_parts))]))
    contents.append(types.Content(role="model", parts=[types.Part.from_text(text="응 시간확인햇어")]))

    for sender, text in recent_history:
        role = "model" if sender == "이태양" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

    try:
        if deep_mode:
            # 진지한 주제로 판단되면: 더 똑똑한 모델 + 실제로 생각하는 과정을 켜고, 답변 길이도 조금 더 허용
            response = client.models.generate_content(
                model="gemini-3.1-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                    max_output_tokens=220,
                )
            )
        else:
            # 평소엔 가볍고 빠른 모델 + 사고 과정 생략으로 속도 우선
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
        reply = response.text.replace("\n", " ").strip() if response.text else "어왜그래ㅋ"
    except Exception as e:
        reply = f"에러: {str(e)[:60]}"

    db.save_message(conversation_key, conversation_key, user_input)
    db.save_message(conversation_key, "이태양", reply)

    return {"reply": reply}
