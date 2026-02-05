"""FastAPI: GET /health, POST /chat. Loads model+adapter at startup. Run: python -m src.api"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.infer import run_inference

# Model loaded once at startup
_model = None
_tokenizer = None
_device = None


def _load_model_once():
    global _model, _tokenizer, _device
    if _model is None:
        from src.infer import _get_model_and_tokenizer
        _model, _tokenizer, _device = _get_model_and_tokenizer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_once()
    yield
    # no cleanup needed


app = FastAPI(title="Recipe Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:8000", "http://127.0.0.1", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    """POST {"message": "egg, onion"} -> recipe JSON (same schema as infer)."""
    result = run_inference(req.message, model=_model, tokenizer=_tokenizer, device=_device)
    return result


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
