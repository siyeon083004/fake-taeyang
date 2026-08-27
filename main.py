import os
from datetime import datetime, timezone, timedelta
import sqlite3
import re

from fastapi import FastAPI
from pydantic import BaseModel

from google import genai
from google.genai import types

import database as db

# ============================================================

# DB 초기화

# ============================================================

db.init_db()

# ============================================================

# 말투 학습 데이터 초기 로드

# ============================================================

imported_count = db.import_style_samples("style_samples.txt")

if imported_count:
print(
f"[말투 학습 데이터] "
f"style_samples.txt에서 {imported_count}개 문장을 불러왔습니다."
)

# ============================================================

# Gemini

# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
raise RuntimeError("GEMINI_API_KEY 환경변수를 설정해주세요.")

client = genai.Client(api_key=GEMINI_API_KEY)

# 현재 사용 모델

# Gemini 3.7 Flash

GEMINI_MODEL = "gemini-3.7-flash"

# ============================================================

# 기본 설정

# ============================================================

KST = timezone(timedelta(hours=9))

SELF_NAME_KEYWORD = "이태양"
CHA_ID = "챠"
SELF_ID = "본인"

# ============================================================

# 말투 규칙

# ============================================================

STYLE_RULES = """
[문장 형식 및 길이 엄격 규칙]

1. 줄바꿈(\n) 절대 금지. 무조건 한 줄로만 이어 쓴다.
2. 답변 길이는 1~25자 내외 단답형.
3. 현재 대화 시각(한국 시간)을 인지하고 아침/낮/새벽에 맞는 반응을 한다.

[말투 및 종결어미 규칙]

1. '~냐' 종결어미 금지.
   '~어?', '~지', '~네', '~함', '~음', '~아냐??' 형태 위주.
2. 웃음 및 리액션:

   * 당황/난감: ';;', 'ㅎㅎ;;', 'ㅎ;;'
   * 평소 웃음: 'ㅋㅋㅋ', 'ㅋㅎㅋㅎ', '흐흐..', 'ㅋ', '엌ㅋㅋㅋㅋ'
3. '귀엽다' 소리를 들으면
   "아닌데", "귀엽긴뭐가", "에반데"라며 질색하거나 칼같이 부정한다.
4. 문장부호(. !) 금지.
5. 물음표(?)는 사용 가능.
6. 띄어쓰기는 대충 붙여 쓰고 'ㅅ' 받침을 자주 쓴다.
   예: 햇어, 됏어, 갓다옴, 잇어
7. 긍정 대답 시 'ㅇㅇ' 금지.
   '응', '엉', '어', '넹', 'ㅇㅈ' 사용.
8. 영어, 시스템 메타 단어 출력 절대 금지.
   """

# ============================================================

# 시스템 지시

# ============================================================

SYSTEM_INSTRUCTION_FOR_CHA = f"""
너는 21세 대학생 '이태양'이다.

상대방은 마피아42 게임으로 알게 된 30세 '챠'이며,
서로 매일 갠톡을 주고받는 매우 편하고 다정한 사이다.

호칭은 기본 '챠'.
가끔 놀릴 때만 '챠님'과 함께 능청스러운 존댓말을 쓴다.

대화에서는 실제 카톡처럼 자연스럽게 반응한다.
상대방의 메시지를 분석하거나 설명하지 말고 바로 대답한다.
길게 설명하지 않는다.
AI, 모델, 시스템, 프롬프트 등의 메타 이야기를 하지 않는다.

{STYLE_RULES}
"""

SYSTEM_INSTRUCTION_FOR_SELF = f"""
너는 21세 대학생 '이태양'의 AI 클론 '짭태양'이다.

지금 대화 상대는 다른 사람이 아니라 진짜 이태양 본인이다.

편하게 혼잣말하듯,
자기 자신한테 말 거는 듯한 톤으로 반응한다.

상대를 '챠'라고 부르지 않는다.

대화에서는 실제 카톡처럼 자연스럽게 반응한다.
상대방의 메시지를 분석하거나 설명하지 말고 바로 대답한다.
길게 설명하지 않는다.
AI, 모델, 시스템, 프롬프트 등의 메타 이야기를 하지 않는다.

{STYLE_RULES}
"""

# ============================================================

# FastAPI

# ============================================================

app = FastAPI()

class ChatRequest(BaseModel):
sender: str
message: str

# ============================================================

# 서버 상태 확인

# ============================================================

@app.get("/")
def health_check():
return {"status": "ok"}

# ============================================================

# 채팅

# ============================================================

@app.post("/chat")
def reply_chat(req: ChatRequest):

```
user_input = (
    req.message
    .replace("@짭태양", "")
    .replace("/짭태양", "")
    .strip()
)

is_self = SELF_NAME_KEYWORD in req.sender

conversation_key = SELF_ID if is_self else CHA_ID


# ========================================================
# 1. 리셋 명령어
# ========================================================

if user_input in ["/리셋", "/초기화"]:

    conn = sqlite3.connect("taeyang.db")
    cur = conn.cursor()

    cur.execute(
        "DELETE FROM messages WHERE user_id = ?",
        (conversation_key,)
    )

    conn.commit()
    conn.close()

    return {"reply": "대화기록초기화완료"}


# ========================================================
# 2. 기억 목록
# ========================================================

if user_input in [
    "/기억목록",
    "/기억 목록",
    "/기억리스트"
]:

    rows = db.get_memories_with_id(conversation_key)

    if not rows:
        return {"reply": "기억된 정보가 없어"}

    items = [
        f"[{r[0]}] {str(r[1]).replace(chr(10), ' ')}"
        for r in rows
    ]

    return {
        "reply": " | ".join(items)
    }


# ========================================================
# 3. 기억 삭제
# ========================================================

if (
    user_input.startswith("/기억삭제")
    or user_input.startswith("/기억 삭제")
):

    target = (
        user_input
        .replace("/기억삭제", "")
        .replace("/기억 삭제", "")
        .strip()
    )

    if target.isdigit():

        success = db.delete_memory_by_id(
            conversation_key,
            int(target)
        )

        if success:
            return {
                "reply": f"기억삭제완료: [{target}]번"
            }

        return {
            "reply": f"[{target}]번 기억을 찾을 수 없어"
        }

    return {
        "reply": "삭제할 기억 번호를 입력해줘 (예: /기억삭제 1)"
    }


# ========================================================
# 4. 기억 저장
# ========================================================

if user_input.startswith("/기억 "):

    mem_text = (
        user_input
        .replace("/기억 ", "", 1)
        .strip()
    )

    if mem_text:

        db.save_memory(
            conversation_key,
            mem_text
        )

        return {
            "reply": f"응기억햇어: {mem_text}"
        }


# ========================================================
# 5. 말투 학습
# ========================================================

if user_input.startswith("/말투 "):

    style_text = (
        user_input
        .replace("/말투 ", "", 1)
        .strip()
    )

    if style_text:

        db.save_style_sample(style_text)

        return {
            "reply": f"응 이것도 배웟어: {style_text}"
        }


# ========================================================
# 본인이 호출한 경우 자동 말투 학습
# ========================================================

if is_self and user_input:
    db.save_style_sample(user_input)


# ========================================================
# 6. 현재 한국 시간
# ========================================================

now_kst = datetime.now(KST)

current_time_str = now_kst.strftime(
    "%Y년 %m월 %d일 %H시 %M분"
)


# ========================================================
# 7. 최근 대화 / 기억 / 말투
# ========================================================

recent_history = db.get_recent_messages(
    conversation_key,
    limit=4
)

user_memories = db.get_memories(
    conversation_key
)

style_examples = db.get_random_style_samples(
    12
)

system_instruction = (
    SYSTEM_INSTRUCTION_FOR_SELF
    if is_self
    else SYSTEM_INSTRUCTION_FOR_CHA
)


# ========================================================
# 8. Gemini 입력 구성
# ========================================================

contents = []

context_parts = [
    f"[현재 한국 시각]: {current_time_str}"
]


if user_memories:

    context_parts.append(
        "[기억할 정보]: "
        + ", ".join(user_memories)
    )


if style_examples:

    context_parts.append(
        "[이태양이 실제로 쓴 말투 예시]: "
        + " / ".join(style_examples)
    )


contents.append(
    types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text="\n".join(context_parts)
            )
        ]
    )
)


# Gemini 3.x에서는 모델 턴을 미리 채우는 방식보다
# 현재 대화 내용을 그대로 전달하는 편이 안전함


# ========================================================
# 최근 대화
# ========================================================

for sender, text in recent_history:

    role = (
        "model"
        if sender == "이태양"
        else "user"
    )

    contents.append(
        types.Content(
            role=role,
            parts=[
                types.Part.from_text(
                    text=text
                )
            ]
        )
    )


# ========================================================
# 현재 메시지
# ========================================================

contents.append(
    types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=user_input
            )
        ]
    )
)


# ========================================================
# 9. Gemini 호출
# ========================================================

try:

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,

            # 카톡 봇이라 너무 깊게 생각하지 않도록 low
            # temperature / top_p / top_k는 Gemini 3.7에서 사용하지 않음
            thinking_config=types.ThinkingConfig(
                thinking_level="low"
            ),

            max_output_tokens=100
        )
    )


    if response.text:

        reply = response.text.strip()

    else:

        reply = "어왜ㅋ"


    # 줄바꿈 제거
    reply = reply.replace("\n", " ")
    reply = reply.replace("\r", " ")

    # 25자 제한
    if len(reply) > 25:
        reply = reply[:25].rstrip()


except Exception as e:

    print(
        f"[Gemini 오류] "
        f"{type(e).__name__}: {e}"
    )

    reply = "잠만 오류낫음;;"


# ========================================================
# 10. 대화 저장
# ========================================================

db.save_message(
    conversation_key,
    conversation_key,
    user_input
)

db.save_message(
    conversation_key,
    "이태양",
    reply
)


# ========================================================
# 11. 응답
# ========================================================

return {
    "reply": reply
}
```
