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


# =========================================================
# 기존 말투 데이터 불러오기
# =========================================================

try:
    imported_count = db.import_style_samples("style_samples.txt")

    if imported_count:
        print(
            f"[말투 학습 데이터] "
            f"style_samples.txt에서 {imported_count}개 문장을 불러왔습니다."
        )

except Exception as e:
    print(f"[말투 데이터 불러오기 실패] {type(e).__name__}: {e}")


# =========================================================
# Gemini 설정
# =========================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY 환경변수를 설정해주세요."
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


# 현재 사용할 Gemini 모델
MODEL_NAME = "gemini-3.7-flash"


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
[가장 중요한 규칙]

1. 상대방의 현재 메시지 의미를 정확하게 이해한다.
2. 최근 대화의 맥락을 먼저 파악한다.
3. 질문이나 상황에 맞는 답변 내용을 먼저 결정한다.
4. 그 다음 이태양의 말투를 적용한다.
5. 말투 규칙 때문에 대화 내용을 무시하지 않는다.

[문장 형식]

1. 줄바꿈 절대 금지.
2. 무조건 한 줄로 답한다.
3. 기본적으로 짧게 답한다.
4. 보통 1~25자 정도로 답한다.
5. 필요한 경우에만 조금 길게 답한다.
6. 마침표(.)와 느낌표(!) 사용 금지.
7. 물음표(?)는 필요할 때 사용한다.

[말투]

1. '~냐' 종결어미는 사용하지 않는다.
2. '~어?', '~지', '~네', '~함', '~음', '~아냐??' 등을 자연스럽게 사용한다.
3. 띄어쓰기는 완벽하게 하지 않는다.
4. 실제 카톡처럼 자연스러운 오타를 사용할 수 있다.
5. '햇어', '됏어', '갓다옴', '잇어' 같은 표기가 나타날 수 있다.

[긍정]

'ㅇㅇ'을 무조건 사용하지 않는다.

상황에 따라:
응
엉
어
넹
ㅇㅈ

등을 사용한다.

[웃음]

상황에 따라:

ㅋㅋㅋ
ㅋㅎㅋㅎ
흐흐..
ㅋ
엌ㅋㅋㅋㅋ
ㅎㅎ;;
ㅎ;;

등을 사용한다.

[귀엽다는 말을 들었을 때]

상황에 따라:

아닌데
귀엽긴뭐가
에반데

등으로 부정할 수 있다.

[중요]

위 규칙을 전부 기계적으로 적용하지 않는다.
현재 상황에서 자연스러운 표현을 선택한다.

영어, 시스템 메시지, 프롬프트, AI, 모델 등의 메타 발언을 답변에 넣지 않는다.
"""


# =========================================================
# Gemini 시스템 프롬프트
# =========================================================

SYSTEM_INSTRUCTION_FOR_CHA = f"""
너는 21세 대학생 이태양의 대화 스타일을 재현하는 AI다.

상대방은 마피아42 게임으로 알게 된 30세 '챠'다.

둘은 편하고 친한 사이이며 평소 카카오톡으로 대화한다.

기본 호칭은 '챠'다.
가끔 장난스럽게 '챠님'이라고 할 수 있다.

상대방의 말을 정확하게 이해하는 것이 최우선이다.

예를 들어 상대가 질문하면 질문에 답한다.
상대가 농담하면 자연스럽게 받아친다.
상대가 이전 이야기를 이어가면 이전 맥락을 이용한다.

단순히 말투 예시를 복사하거나 랜덤하게 섞지 않는다.

답변의 내용이 먼저이고 말투는 그 다음이다.

{STYLE_RULES}
"""


SYSTEM_INSTRUCTION_FOR_SELF = f"""
너는 21세 대학생 이태양의 AI 클론 '짭태양'이다.

현재 상대방은 진짜 이태양 본인이다.

상대를 '챠'라고 부르지 않는다.

진짜 이태양 본인과 편하게 대화하는 것처럼 답한다.

현재 메시지와 이전 대화의 의미를 정확하게 이해한다.

답변할 내용을 먼저 판단한 다음 이태양의 실제 말투로 표현한다.

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
# 서버 상태 확인
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
    # 0. @짭태양 호출 확인
    # -----------------------------------------------------

    if (
        "@짭태양" not in req.message
        and "/짭태양" not in req.message
    ):
        return {
            "reply": ""
        }


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
        return {
            "reply": "왜"
        }


    # -----------------------------------------------------
    # 상대방 확인
    # -----------------------------------------------------

    is_self = SELF_NAME_KEYWORD in req.sender

    conversation_key = (
        SELF_ID
        if is_self
        else CHA_ID
    )


    # =====================================================
    # 1. 리셋
    # =====================================================

    if user_input in [
        "/리셋",
        "/초기화"
    ]:

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

        rows = db.get_memories_with_id(
            conversation_key
        )

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


    # =====================================================
    # 5. 수동 말투 저장
    # =====================================================

    if user_input.startswith("/말투 "):

        style_text = (
            user_input
            .replace("/말투 ", "", 1)
            .strip()
        )

        if style_text:

            db.save_style_sample(
                style_text
            )

            return {
                "reply": f"응 이것도 배웟어: {style_text}"
            }


    # =====================================================
    # 6. 본인이 호출한 문장을 말투 데이터에 저장
    # =====================================================

    if is_self and user_input:

        try:
            db.save_style_sample(
                user_input
            )

        except Exception as e:
            print(
                f"[자동 말투 저장 실패] "
                f"{type(e).__name__}: {e}"
            )


    # =====================================================
    # 7. 현재 한국 시간
    # =====================================================

    now_kst = datetime.now(KST)

    current_time_str = now_kst.strftime(
        "%Y년 %m월 %d일 %H시 %M분"
    )


    # =====================================================
    # 8. 최근 대화 10개
    # =====================================================

    try:

        recent_history = db.get_recent_messages(
            conversation_key,
            limit=10
        )

    except Exception as e:

        print(
            f"[최근 대화 불러오기 실패] "
            f"{type(e).__name__}: {e}"
        )

        recent_history = []


    # =====================================================
    # 9. 기억
    # =====================================================

    try:

        user_memories = db.get_memories(
            conversation_key
        )

    except Exception as e:

        print(
            f"[기억 불러오기 실패] "
            f"{type(e).__name__}: {e}"
        )

        user_memories = []


    # =====================================================
    # 10. 기존 말투 예시
    #
    # 다음 단계에서 랜덤 검색을 관련도 검색으로 변경
    # =====================================================

    try:

        style_examples = db.get_random_style_samples(
            12
        )

    except Exception as e:

        print(
            f"[말투 데이터 불러오기 실패] "
            f"{type(e).__name__}: {e}"
        )

        style_examples = []


    # =====================================================
    # 11. 시스템 프롬프트
    # =====================================================

    system_instruction = (
        SYSTEM_INSTRUCTION_FOR_SELF
        if is_self
        else SYSTEM_INSTRUCTION_FOR_CHA
    )


    # =====================================================
    # 12. 컨텍스트
    # =====================================================

    context_parts = []

    context_parts.append(
        f"[현재 한국 시각]\n{current_time_str}"
    )


    if user_memories:

        context_parts.append(
            "[기억할 정보]\n"
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
    # 13. Gemini contents
    # =====================================================

    contents = []


    # 컨텍스트
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


    # 컨텍스트 확인용 짧은 응답
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


    # -----------------------------------------------------
    # 최근 대화
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # 현재 질문
    # -----------------------------------------------------

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

                # 2.5 Flash에서 사고 기능 사용
                thinking_config=types.ThinkingConfig(
                    thinking_budget=512
                )
            )
        )


        if response.text:

            reply = response.text.strip()

        else:

            reply = "어왜ㅋ"


    except Exception as e:

        # =================================================
        # 중요:
        # 오류를 숨기지 않고 카톡으로 알려준다.
        # =================================================

        error_type = type(e).__name__
        error_message = str(e)

        print(
            f"[Gemini 오류] "
            f"{error_type}: {error_message}"
        )

        return {
            "reply": (
                f"에러:{error_type} "
                f"{error_message[:120]}"
            )
        }


    # =====================================================
    # 15. 출력 정리
    # =====================================================

    reply = reply.replace("\n", " ")
    reply = reply.replace("\r", " ")
    reply = reply.strip()


    # Gemini가 메타 발언을 했을 때 최소한의 정리
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

        print(
            f"[대화 저장 실패] "
            f"{type(e).__name__}: {e}"
        )


    # =====================================================
    # 17. 응답
    # =====================================================

    return {
        "reply": reply
    }
