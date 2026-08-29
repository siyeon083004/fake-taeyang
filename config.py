import os
from datetime import timezone, timedelta
from google import genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수를 설정해주세요.")

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
KST = timezone(timedelta(hours=9))

