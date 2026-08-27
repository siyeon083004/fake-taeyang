import os
from datetime import datetime, timezone, timedelta
import sqlite3

from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types

import database as db


# =========================================================
# DB 초기화
# =========================================================

db.init_db()


# 기존 style_samples.txt가 있다면 최초 1회 DB에 불러오기
try:
    imported_count = db.import_style_samples("style_samples.txt")
    if imported_count:
        print(
            f"[말투 학습 데이터] style_samples.txt에서 "
            f"{imported_count}개 문장을 불러왔습니다."
        )
except Exception as e:
    print(f"[말투 데이터 불러오기 실패] {e}")


# =========================================================
# Gemini
# =========================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY 환경변수를 설정해주세요."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# 현재 안정적으로 사용할 모델
MODEL_NAME = "gemini-2.5-flash"


# =========================================================
# 기본 설정
# =========================================================

KST = timezone(timedelta(hours=9))

SELF_NAME_KEYWORD = "이태양"

CHA_ID = "챠"
SELF_ID = "본인"


# =========================================================
# 말투 규칙
# =========================================================

STYLE_RULES = """
[최우선 규칙]
1. 상대방의 현재 메시지 의미와 직전 대화 맥락을 먼저 정확하게 이해한다.
2. 내용에 맞는 자연스러운 답변을 만든 뒤 이태양의 말투를 적용한다.
3. 말투 때문에 대화의 의미를 놓치지 않는다.

[문장 형식]
1. 줄바꿈 절대 금지. 무조건 한 줄.
2. 기본적으로 짧게 답한다.
3. 답변은 보통 1~25자 정도로 유지한다.
4. 정말 필요한 경우에만 조금 길어질 수 있다.
5. 문장부호 . ! 사용 금지.
6. 물음표 ? 사용 가능.

[말투]
1. '~냐' 종결어미는 사용하지 않는다.
2. '~어?', '~지', '~네', '~함', '~음', '~아냐??' 등을 자연스럽게 사용한다.
3. 띄어쓰기는 완벽하게 하지 않는다.
4. 'ㅅ' 받침이 들어가는 오타/표기를 자연스럽게 사용할 수 있다.
   예: 햇어, 됏어, 갓다옴, 잇어
5. 긍정 표현에서 'ㅇㅇ'을 남발하지 않는다.
   응, 엉, 어, 넹, ㅇㅈ 등을 상황에 맞게 사용한다.

[웃음]
상황에 따라 다음과 같은 표현을 사용할 수 있다.
ㅋㅋㅋ
ㅋㅎㅋㅎ
흐흐..
ㅋ
엌ㅋㅋㅋㅋ
ㅎㅎ;;
ㅎ;;

[특정 반응]
'귀엽다'라는 말을 들으면
'아닌데', '귀엽긴뭐가', '에반데'
등으로 부정적으로 반응하는 경향이 있다.

[중요]
위 규칙을 기계적으로 전부 집어넣지 않는다.
현재 대화 상황에 가장 자연스러운 표현을 선택한다.

영어, 시스템 메시지, 프롬프트, AI, 모델, 메타 발언을 답변에 포함하지 않는다.
"""


# =========================================================
# 시스템 프롬프트
# =========================================================

SYSTEM_INSTRUCTION_FOR_CHA = f"""
너는 21세 대학생 이태양의 대화 스타일을 재현하는 AI다.

상대방은 마피아42 게임으로 알게 된 30세 '챠'다.
둘은 매우 편하고 친한 사이이며 평소 갠톡을 자주 한다.

호칭은 기본적으로 '챠'.
가끔 장난칠 때 '챠님'이라고 부를 수 있다.

중요:
상대방이 무엇을 말했는지 정확히 이해하고
그 상황에서 이태양이 실제로 할 법한 내용으로 답한다.

단순히 말투 예시를 이어붙이지 않는다.
질문에는 질문에 맞는 답을 한다.
상대가 농담하면 농담으로 받아친다.
상대가 질문하면 질문에 맞춰 대답한다.

{STYLE_RULES}
"""


SYSTEM_INSTRUCTION_FOR_SELF = f"""
너는 21세 대학생 이태양의 AI 클론 '짭태양'이다.

현재 상대방은 진짜 이태양 본인이다.

상대방을 '챠'라고 부르지 않는다.
진짜 이태양 본인과 대화하는 것처럼 편하게 반응한다.

중요:
현재 메시지와 이전 대화의 의미를 정확히 이해한다.
상황과 맥락에 맞는 내용을 먼저 결정한다.
그 다음 이태양의 실제 말투로 표현한다.

단순히 랜덤한 말투 예시를 복사하지 않는다.

{STYLE_RULES}
"""


# =========================================================
# FastAPI
# =========================================================

app = FastAPI()


class ChatRequest(BaseModel):
    sender: str
    message: str


# =========================================================
# 상태 확인
# =========================================================

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_NAME
    }


# =========================================================
# 채팅
# =========================================================

@app.post("/chat")
def reply_chat(req: ChatRequest):

    # -----------------------------------------------------
    # 0. 호출 여부 확인
    # -----------------------------------------------------

    if "@짭태양" not in req.message and "/짭태양" not in req.message:
        return {"reply": ""}


    # -----------------------------------------------------
    # 호출어 제거
    # -----------------------------------------------------

    user_input = (
        req.message
        .replace("@짭태양", "")
        .replace("/짭태양", "")
        .strip()
    )

    if not user_input:
        return {"reply": "왜"}


    # -----------------------------------------------------
    # 상대방 구분
    # -----------------------------------------------------

    is_self = SELF_NAME_KEYWORD in req.sender

    conversation_key = SELF_ID if is_self else CHA_ID


    # =====================================================
    # 1. 리셋
    # =====================================================

    if user_input in ["/리셋", "/초기화"]:

        conn = sqlite3.connect("taeyang.db")

        try:
            cur = conn.cursor()

            cur.execute(
                "DELETE FROM messages WHERE user_id = ?",
                (conversation_key,)
            )

            conn.commit()

        finally:
            conn.close()

        return {
            "reply": "대화기록초기화완료"
        }


    # =====================================================
    # 2. 기억 목록
    # =====================================================

    if user_input in [
        "/기억목록",
        "/기억 목록",
        "/기억리스트"
    ]:

        rows = db.get_memories_with_id(conversation_key)

        if not rows:
            return {
                "reply": "기억된 정보가 없어"
            }

        items = [
            f"[{r[0]}] {str(r[1]).replace(chr(10), ' ')}"
            for r in rows
        ]

        return {
            "reply": " | ".join(items)
        }


    # =====================================================
    # 3. 기억 삭제
    # =====================================================

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


    # =====================================================
    # 4. 기억 저장
    # =====================================================

    if user_input.startswith("/기억 "):

        mem_text = user_input.replace(
            "/기억 ",
            "",
            1
        ).strip()

        if mem_text:

            db.save_memory(
                conversation_key,
                mem_text
            )

            return {
                "reply": f"응기억햇어: {mem_text}"
            }


    # =====================================================
    # 5. 수동 말투 저장
    # =====================================================

    if user_input.startswith("/말투 "):

        style_text = user_input.replace(
            "/말투 ",
            "",
            1
        ).strip()

        if style_text:

            db.save_style_sample(style_text)

            return {
                "reply": f"응 이것도 배웟어: {style_text}"
            }


    # =====================================================
    # 6. 본인이 입력한 실제 문장을 말투 데이터에 저장
    # =====================================================

    if is_self and user_input:

        try:
            db.save_style_sample(user_input)
        except Exception as e:
            print(f"[자동 말투 저장 실패] {e}")


    # =====================================================
    # 7. 현재 시간
    # =====================================================

    now_kst = datetime.now(KST)

    current_time_str = now_kst.strftime(
        "%Y년 %m월 %d일 %H시 %M분"
    )


    # =====================================================
    # 8. 최근 대화
    # =====================================================

    recent_history = db.get_recent_messages(
        conversation_key,
        limit=10
    )


    # =====================================================
    # 9. 기억
    # =====================================================

    user_memories = db.get_memories(
        conversation_key
    )


    # =====================================================
    # 10. 기존 말투 예시
    #
    # 아직 database.py를 개조하지 않았으므로
    # 지금은 기존 랜덤 방식 유지.
    # 다음 단계에서 '관련 대화 검색' 방식으로 교체한다.
    # =====================================================

    try:
        style_examples = db.get_random_style_samples(12)
    except Exception:
        style_examples = []


    # =====================================================
    # 11. 시스템 프롬프트 선택
    # =====================================================

    system_instruction = (
        SYSTEM_INSTRUCTION_FOR_SELF
        if is_self
        else SYSTEM_INSTRUCTION_FOR_CHA
    )


    # =====================================================
    # 12. 컨텍스트 구성
    # =====================================================

    context_parts = [
        f"[현재 한국 시각]\n{current_time_str}"
    ]

    if user_memories:

        context_parts.append(
            "[상대방에 대해 기억하고 있는 정보]\n"
            + ", ".join(user_memories)
        )

    if style_examples:

        context_parts.append(
            "[이태양의 실제 말투 예시]\n"
            + "\n".join(
                f"- {example}"
                for example in style_examples
            )
        )


    # =====================================================
    # 13. Gemini 대화 구성
    # =====================================================

    contents = []

    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="\n\n".join(context_parts)
                )
            ]
        )
    )

    contents.append(
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(
                    text="응 확인햇어"
                )
            ]
        )
    )


    # 최근 대화 10개
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


    # 현재 메시지
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


    # =====================================================
    # 14. Gemini 호출
    # =====================================================

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=100,
            )
        )

        if response.text:

            reply = response.text.strip()

        else:

            reply = "어왜ㅋ"


    except Exception as e:

        print(f"[Gemini 오류] {repr(e)}")

        return {
            "reply": "지금 좀 이상함;;"
        }


    # =====================================================
    # 15. 출력 후처리
    # =====================================================

    reply = reply.replace("\n", " ")
    reply = reply.replace("\r", " ")
    reply = reply.strip()

    # 모델이 혹시 메타 문장을 뱉으면 최소한의 정리
    reply = reply.replace("AI:", "")
    reply = reply.replace("이태양:", "")
    reply = reply.strip()

    if not reply:
        reply = "어왜ㅋ"


    # =====================================================
    # 16. 대화 저장
    # =====================================================

    try:

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

    except Exception as e:

        print(f"[대화 저장 실패] {e}")


    return {
        "reply": reply
    }
