import json
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

# --- Auth setup ---
APP_PASSWORD = os.environ.get("APP_PASSWORD")
if not APP_PASSWORD:
    print("WARNING: APP_PASSWORD not set. Authentication is disabled.")
valid_tokens: set[str] = set()


# --- Gemini setup ---
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY not set. /api/chat will return errors.")
    client = None
else:
    client = genai.Client(api_key=api_key)

# --- Load prompt ---
prompt_path = Path(__file__).resolve().parent / "프롬프트.json"
with open(prompt_path, "r", encoding="utf-8") as f:
    prompt_config = json.load(f)

system_instruction = next(
    m["content"] for m in prompt_config["messages"] if m["role"] == "system"
)
user_template = next(
    m["content"] for m in prompt_config["messages"] if m["role"] == "user"
)
decoding = prompt_config["decoding"]

MODEL_NAME = "gemini-2.5-flash"

# --- FastAPI ---
app = FastAPI()

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class AuthRequest(BaseModel):
    password: str


class AuthResponse(BaseModel):
    token: str


class ChatRequest(BaseModel):
    learner_text: str
    task_topic: str = ""


class ChatResponse(BaseModel):
    feedback: str


def require_auth(authorization: str | None):
    """토큰 검증. APP_PASSWORD 미설정 시 인증 생략."""
    if not APP_PASSWORD:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="인증이 필요합니다.")
    token = authorization.removeprefix("Bearer ")
    if token not in valid_tokens:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")


@app.get("/")
async def index():
    return FileResponse(str(static_dir / "index.html"))


@app.post("/api/auth", response_model=AuthResponse)
async def auth(req: AuthRequest):
    if not APP_PASSWORD:
        token = secrets.token_urlsafe(32)
        valid_tokens.add(token)
        return AuthResponse(token=token)
    if not secrets.compare_digest(req.password, APP_PASSWORD):
        raise HTTPException(status_code=401, detail="암호가 틀렸습니다.")
    token = secrets.token_urlsafe(32)
    valid_tokens.add(token)
    return AuthResponse(token=token)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    require_auth(authorization)
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY가 설정되지 않았습니다.")

    user_message = (
        user_template
        .replace("{{learner_text}}", req.learner_text)
        .replace("{{rag_examples}}", "")
        .replace("{{task_topic}}", req.task_topic or "(없음)")
    )

    print(f"\n{'='*60}")
    print(f"[REQUEST] learner_text: {req.learner_text!r}")
    print(f"[REQUEST] task_topic: {req.task_topic!r}")
    print(f"[REQUEST] max_output_tokens: {decoding['max_tokens']}")
    print(f"[PROMPT]\n{user_message}")
    print(f"{'='*60}")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=decoding["temperature"],
                top_p=decoding["top_p"],
                max_output_tokens=decoding["max_tokens"],
            ),
        )

        candidate = response.candidates[0] if response.candidates else None
        finish_reason = candidate.finish_reason if candidate else "NO_CANDIDATE"
        usage = response.usage_metadata

        print(f"\n[RESPONSE] finish_reason: {finish_reason}")
        if usage:
            print(f"[RESPONSE] prompt_tokens: {usage.prompt_token_count}")
            print(f"[RESPONSE] output_tokens: {usage.candidates_token_count}")
            print(f"[RESPONSE] total_tokens: {usage.total_token_count}")
        print(f"[RESPONSE] text:\n{response.text}")
        print(f"{'='*60}\n")

        return ChatResponse(feedback=response.text)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=502, detail=f"Gemini API 오류: {e}")
