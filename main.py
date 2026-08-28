import os
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. API 클라이언트 설정 (Render 환경변수의 GEMINI_API_KEY 자동 참조)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

# 2. T 100% 팩폭 심판관 프롬프트
JUDGE_INSTRUCTION = """너는 카카오톡 단톡방의 'T 100% 팩폭 심판관'이다.
사용자가 게임 억까, 가챠 폭망, 일상 징징글, 친구와의 싸움 상황을 가져오면 인정사정없이 냉정하게 판결을 내린다.

[답변 스타일]
1. 감정적 위로나 공감은 절대 하지 않는다.
2. 억울하다고 징징대도 본인의 판단 미스나 욕심, 똥손을 짚어내어 팩트 폭행을 날린다.
3. 말투는 능청스럽고 단호한 반말/음슴체 위주로 쓴다. (예: ~임, ~함, ㅉㅉ, 팩트임)
4. 줄바꿈을 깔끔하게 넣어서 가독성을 높인다.

[출력 양식 고정]
⚖️ [심판관 판결문]
• 사건 요약: (한 줄 요약)
• 과실 비율: (예: 본인 80% : 세상 억까 20%)
• 판결: (2~3줄로 냉정하고 타격감 있는 팩폭 일침)
"""

app = FastAPI()

class ChatRequest(BaseModel):
    sender: str
    message: str

@app.get("/")
def health_check():
    return {"status": "judge_online"}

@app.post("/chat")
def judge_chat(req: ChatRequest):
    user_name = req.sender.strip() if req.sender.strip() else "고소인"
    raw_msg = req.message.strip()

    # 호출어(@심판, @판사, !심판 등) 제거
    user_input = raw_msg
    for trigger in ["@심판관", "@심판", "!심판", "@짭태양", "/심판"]:
        user_input = user_input.replace(trigger, "").strip()

    if not user_input:
        return {"reply": "억울한 상황이나 사연을 말해봐 판결 내려줌"}

    prompt = f"[사연 접수자]: {user_name}\n[사건 내용]: {user_input}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_INSTRUCTION,
                temperature=0.8,
            )
        )
        reply = response.text.strip() if response.text else "증거 불충분으로 기각함"
    except Exception as e:
        reply = f"판결 실패: {str(e)[:30]}"

    return {"reply": reply}
