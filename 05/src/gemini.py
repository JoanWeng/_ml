from google import genai
from .config import GEMINI_API_KEY, MODEL

_gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def call_gemini(prompt: str, system: str = "") -> str:
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    response = _gemini_client.models.generate_content(
        model=MODEL,
        contents=full_prompt,
    )
    return response.text.strip()
