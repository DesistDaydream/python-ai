"""
OpenAI-compatible API server for local HuggingFace models.

Usage:
    pip install fastapi uvicorn transformers torch accelerate
    python openai_server.py --model /path/to/model --port 8000
"""

import os
import argparse
import time
import uuid
from typing import AsyncIterator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Example:
# python .\cmd\inference_service\inference_service.py --model D:\appdata\models\desistdaydream

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model path or HuggingFace model ID")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
args, _ = parser.parse_known_args()
model_id = os.path.basename(os.path.normpath(args.model))


# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------

print(f"Loading model from: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded.")

# ---------------------------------------------------------------------------
# OpenAI-compatible schema
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str  # system / user / assistant / tool / function ...
    content: str | None = None  # tool 消息的 content 有时为 None


class ChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="OpenAI-compatible Local Model API")


_SUPPORTED_ROLES = {"system", "user", "assistant"}


def build_input_ids(messages: list[Message]) -> torch.Tensor:
    """Apply chat template and tokenize.

    Filters out roles the model chat template does not understand
    (e.g. tool, function) to avoid template rendering errors.
    """
    raw = [
        {"role": m.role, "content": m.content or ""}
        for m in messages
        if m.role in _SUPPORTED_ROLES
    ]
    text = tokenizer.apply_chat_template(raw, tokenize=False, add_generation_prompt=True)
    return tokenizer([text], return_tensors="pt").to(model.device)


def make_chunk(delta_content: str, finish_reason: str | None, request_id: str, model_name: str) -> dict:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": delta_content},
                "finish_reason": finish_reason,
            }
        ],
    }


async def stream_response(request: ChatCompletionRequest, request_id: str) -> AsyncIterator[str]:
    import json

    inputs = build_input_ids(request.messages)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        **inputs,  # type: ignore
        "streamer": streamer,
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "do_sample": request.temperature > 0,
    }

    # Run generation in a background thread so we can stream from the main coroutine.
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for token in streamer:
        chunk = make_chunk(token, None, request_id, request.model)
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    # Send the final chunk with finish_reason.
    chunk = make_chunk("", "stop", request_id, request.model)
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

    thread.join()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    print(request)
    request_id = f"chatcmpl-{uuid.uuid4().hex}"

    if request.stream:
        return StreamingResponse(
            stream_response(request, request_id),
            media_type="text/event-stream",
        )

    # Non-streaming
    inputs = build_input_ids(request.messages)
    input_len = inputs["input_ids"].shape[1]  # type: ignore

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,  # type: ignore
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
        )

    # Decode only the newly generated tokens.
    new_ids = output_ids[0][input_len:]
    content = tokenizer.decode(new_ids, skip_special_tokens=True)
    prompt_tokens = input_len
    completion_tokens = len(new_ids)

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }

@app.get("/v1/models/{mid:path}")
async def retrieve_model(mid: str):
    if mid != model_id:
        raise HTTPException(status_code=404, detail=f"Model '{mid}' not found")
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "local",
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)