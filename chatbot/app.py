import json
import os
import secrets
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

from rag import get_rag_examples

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

# --- In-memory conversation store ---
# conversation_id → { title, created_at, turns: [{role, text}] }
conversations: dict[str, dict] = {}

MAX_TURNS_CONTEXT = 20  # 최근 20턴(10 user + 10 model)만 Gemini에 전송

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
    conversation_id: Optional[str] = None  # None이면 새 대화 생성


class ChatResponse(BaseModel):
    feedback: str
    conversation_id: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: float


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: float
    turns: list[dict]


class NewConversationResponse(BaseModel):
    id: str


def require_auth(authorization: str | None):
    """토큰 검증. APP_PASSWORD 미설정 시 인증 생략."""
    if not APP_PASSWORD:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="인증이 필요합니다.")
    token = authorization.removeprefix("Bearer ")
    if token not in valid_tokens:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")


def make_title(learner_text: str) -> str:
    """첫 메시지에서 대화 제목 생성 (최대 20자)."""
    title = learner_text.strip().replace("\n", " ")
    return title[:20] + ("…" if len(title) > 20 else "")


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


# --- Conversation CRUD ---

@app.get("/api/conversations", response_model=list[ConversationSummary])
async def list_conversations(authorization: str | None = Header(default=None)):
    require_auth(authorization)
    result = [
        ConversationSummary(id=cid, title=c["title"], created_at=c["created_at"])
        for cid, c in conversations.items()
    ]
    result.sort(key=lambda x: x.created_at, reverse=True)
    return result


@app.get("/api/conversations/{cid}", response_model=ConversationDetail)
async def get_conversation(cid: str, authorization: str | None = Header(default=None)):
    require_auth(authorization)
    if cid not in conversations:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
    c = conversations[cid]
    return ConversationDetail(
        id=cid,
        title=c["title"],
        created_at=c["created_at"],
        turns=c["turns"],
    )


@app.delete("/api/conversations/{cid}")
async def delete_conversation(cid: str, authorization: str | None = Header(default=None)):
    require_auth(authorization)
    if cid not in conversations:
        raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
    del conversations[cid]
    return {"ok": True}


@app.post("/api/conversations", response_model=NewConversationResponse)
async def new_conversation(authorization: str | None = Header(default=None)):
    require_auth(authorization)
    cid = str(uuid.uuid4())
    conversations[cid] = {"title": "새 대화", "created_at": time.time(), "turns": []}
    return NewConversationResponse(id=cid)


# --- Chat (multi-turn) ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    require_auth(authorization)
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY가 설정되지 않았습니다.")

    # 대화 세션 가져오기 또는 생성
    cid = req.conversation_id
    if not cid or cid not in conversations:
        cid = str(uuid.uuid4())
        conversations[cid] = {"title": "새 대화", "created_at": time.time(), "turns": []}

    conv = conversations[cid]

    rag_examples = get_rag_examples(req.learner_text)

    user_message = (
        user_template
        .replace("{{learner_text}}", req.learner_text)
        .replace("{{rag_examples}}", rag_examples)
        .replace("{{task_topic}}", req.task_topic or "(없음)")
    )

    # 이전 대화 이력 (최근 MAX_TURNS_CONTEXT 턴만)
    history = conv["turns"][-MAX_TURNS_CONTEXT:]

    contents = [
        types.Content(role=t["role"], parts=[types.Part(text=t["text"])])
        for t in history
    ]
    contents.append(
        types.Content(role="user", parts=[types.Part(text=user_message)])
    )

    print(f"\n{'='*60}")
    print(f"[REQUEST] conversation_id: {cid}")
    print(f"[REQUEST] learner_text: {req.learner_text!r}")
    print(f"[REQUEST] task_topic: {req.task_topic!r}")
    print(f"[REQUEST] history_turns: {len(history)}")
    print(f"[REQUEST] max_output_tokens: {decoding['max_tokens']}")
    print(f"[PROMPT]\n{user_message}")
    print(f"{'='*60}")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
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

        # 대화 이력 저장
        conv["turns"].append({"role": "user", "text": user_message})
        conv["turns"].append({"role": "model", "text": response.text})

        # 첫 메시지로 제목 설정
        if conv["title"] == "새 대화":
            conv["title"] = make_title(req.learner_text)

        return ChatResponse(feedback=response.text, conversation_id=cid)
    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=502, detail=f"Gemini API 오류: {e}")
