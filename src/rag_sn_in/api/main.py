from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse 
import uuid
import json
import gc
import torch
from rag_sn_in.config import LLM_NAME
from rag_sn_in.llm.generator import load_generation_llm, clear_vram
from transformers import TextIteratorStreamer
import threading
from rag_sn_in.database.db_setup import ensure_ToCreate_collection
from rag_sn_in.eval.evaluate_RAG import pure_retrieval
from rag_sn_in.database.client import get_client as get_qdrant_client


app = FastAPI(
    title="RAG-SN-IN API",
    version="0.1.0",
)



class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False

    
@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": LLM_NAME, "created": 1720000000,"object": "model", "owned_by": "local"}
    ],
}


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten later; fine for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a specialist in RAG. You must answer questions only from the context "
        "given to you alongside the user's question. If the question and the contexts are "
        "irrelevant to each other, do not hallucinate or try to please the user; apologize "
        "and say the question is outside the scope of the documentation you can access, and "
        "ask them to ask a relevant question. "
        "The contexts are always in French, but the user may write in any language. Always "
        "reply in the user's language, not the context's language. "
        "When possible, cite your answers by page, e.g.: 'According to page 30, ...'."
    ),
}


tokenizer = None
model = None


def get_model():
    global tokenizer,model
    if model is None:
        tokenizer, model = load_generation_llm(LLM_NAME)
    return tokenizer, model


def augment_last_user_message(messages: list[dict], retrieved: list) -> list[dict]:
    augmented = [m.copy() for m in messages]

    if retrieved:
        blocks = []
        for r in retrieved:
            p = r.payload
            start, end = p.get("page_start"), p.get("page_end")
            label = f"[page {start}]" if start == end else f"[pages {start}-{end}]"
            blocks.append(f"{label}\n{p['text']}")
        suffix = "\n\nContext:\n" + "\n\n".join(blocks)
    else:
        suffix = "\n\nContext:\n[No relevant passages found in the documentation.]"

    augmented[-1]["content"] += suffix
    return augmented


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    print("Preparing Qdrant Client DB. . . ")
    qdrant_client = get_qdrant_client()
    contexts = []
    last_message = request.messages[-1].content  
    print("Searching & Renranking . . ")
    contexts = []
    collection_names = [c.name for c in qdrant_client.get_collections().collections]
    for c in collection_names:
        contexts.extend(pure_retrieval(qdrant_client, c, last_message,
                        retrieve_k=30, use_reranker=True))

    # Unload retrieval models to guarantee VRAM headroom before LLM generation
    from rag_sn_in.llm.reranker import unload_reranker
    from rag_sn_in.llm.embedding import unload_embedder
    unload_reranker()
    unload_embedder()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Retrieval models unloaded from VRAM.")


    print('CONTEXTS = ', contexts)
    print(f"Flattened {len(contexts)} chunks")
    print(type(contexts[0]), contexts[0].payload.get("page_start"))
    messages = [SYSTEM_PROMPT] + [m.model_dump() for m in request.messages]

    augmented = augment_last_user_message(messages,contexts)
    tokenizer, model = get_model()
    prompt_text = tokenizer.apply_chat_template(
        augmented,
        tokenize=False,
        add_generation_prompt=True
    )
    print(prompt_text)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,          
    skip_special_tokens=True  
    )
    
    generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
            )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    request_uuid = str(uuid.uuid4())
    
    if request.stream:
        print("→ STREAMING path")
        def gen():
            for text_chunk in streamer:
                chunk = {
                    "id": request_uuid,
                    "object": "chat.completion.chunk",
                    "model": "rag-sn-in",
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": text_chunk},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            final = {
                "id": request_uuid,
                "object": "chat.completion.chunk",
                "model": "rag-sn-in",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")
    else:
        full_text = "".join(text_chunk for text_chunk in streamer)
    return {
        "id": request_uuid,
        "object": "chat.completion",
        "model": LLM_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": "stop"
        }]
    }
