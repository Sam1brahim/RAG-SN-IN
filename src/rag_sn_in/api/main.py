from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse 
import uuid
import json
import gc
import torch
from rag_sn_in.config import LLM_NAME
from rag_sn_in.llm.generator import load_generation_llm, OLLAMA_MODELS
import ollama
from transformers import TextIteratorStreamer
import threading
from rag_sn_in.observability.client import is_enabled, get_client as get_langfuse_client
from langfuse import propagate_attributes

#from rag_sn_in.database.db_setup import ensure_ToCreate_collection
from rag_sn_in.llm.retrieval import pure_retrieval
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

OLLAMA_OPTIONS = {"temperature": 0.2, "top_p": 0.9}

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


def _completion_chunk(request_uuid: str, delta: dict, finish_reason=None) -> str:
    chunk = {
        "id": request_uuid,
        "object": "chat.completion.chunk",
        "model": LLM_NAME,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _completion_response(request_uuid: str, content: str) -> dict:
    return {
        "id": request_uuid,
        "object": "chat.completion",
        "model": LLM_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    is_ollama = LLM_NAME.lower() in OLLAMA_MODELS
    # ******************************************** LANGFUSE *****************************
    # ******************************************** LANGFUSE *****************************

    # Enabling Langfuse
    if is_enabled():
        print("Langfuse For Observability is Enabled. \n")
        print("Starting Langfuse . . . \n")
        langfuse = get_langfuse_client()
        
        qdrant_client = get_qdrant_client()
        last_message = request.messages[-1].content  
    
        print("Preparing Qdrant Client DB. . . ")
        print("Searching & Reranking . . ")
        contexts = []
        available_collections = [c.name for c in qdrant_client.get_collections().collections]
        target_collections = ["railway"] if "railway" in available_collections else available_collections
        for c in target_collections:
            contexts.extend(pure_retrieval(qdrant_client, c, last_message,
                            retrieve_k=30, use_reranker=True))

        if not is_ollama:
            # Local HF models need the GPU freed; Ollama inference is server-side,
            # so keep the retrieval models hot and skip the reload penalty per request.
            from rag_sn_in.llm.reranker import unload_reranker
            from rag_sn_in.llm.embedding import unload_embedder
            unload_reranker()
            unload_embedder()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Retrieval models unloaded from VRAM.")

        print(f"Flattened {len(contexts)} chunks")
        if contexts:
            print(type(contexts[0]), contexts[0].payload.get("page_start"))

        messages = [SYSTEM_PROMPT] + [m.model_dump() for m in request.messages]
        augmented = augment_last_user_message(messages, contexts)
        request_uuid = str(uuid.uuid4())

        if is_ollama:
                # Ollama applies the gemma4 chat renderer server-side to the raw
                # messages list — never pre-format a prompt string for /api/chat.
                ollama_model = OLLAMA_MODELS[LLM_NAME.lower()]
                print(f"INFO: Using Ollama model '{ollama_model}'")
                if request.stream:
                    def gen():
                        root_span = langfuse.start_observation(
                            as_type="span",
                            name="user-request",
                            input={"user_query": last_message},
                        )
                        gen1 = langfuse.start_observation(
                            as_type="generation",
                            name="llm-step-1",
                            model=LLM_NAME,
                            input=augmented,
                        )
                        full_text = []
                        output_lang = ""
                        try:
                            stream = ollama.chat(
                                model=ollama_model,
                                messages=augmented,
                                stream=True,
                                options=OLLAMA_OPTIONS,
                            )
                            for chunk in stream:
                                text_chunk = chunk["message"]["content"]
                                if text_chunk:
                                    full_text.append(text_chunk)
                                    yield _completion_chunk(
                                        request_uuid,
                                        {"role": "assistant", "content": text_chunk},
                                    )
                            yield _completion_chunk(request_uuid, {}, finish_reason="stop")
                            yield "data: [DONE]\n\n"

                            output_lang = "".join(full_text)
                            gen1.update(output=output_lang)
                            root_span.update(output=output_lang)
                        except Exception as e:
                            print(f"ERROR connecting to Ollama: {e}")
                            gen1.update(output=f"[Error: {e}]", level="ERROR", status_message=str(e))
                            root_span.update(level="ERROR", status_message=str(e))
                            yield _completion_chunk(
                                request_uuid,
                                {"role": "assistant", "content": f"\n\n[Error: {e}]"},
                                finish_reason="error",
                            )
                            yield "data: [DONE]\n\n"
                        finally:
                            # explicit end — no context detach, so no cross-context error
                            gen1.end()
                            root_span.end()
                            langfuse.flush()

                    return StreamingResponse(gen(), media_type="text/event-stream")

                try:
                        response = ollama.chat(
                            model=ollama_model,
                            messages=augmented,
                            stream=False,
                            options=OLLAMA_OPTIONS,
                        )
                        return _completion_response(request_uuid, response["message"]["content"])
                except Exception as e:
                        raise HTTPException(
                            status_code=503,
                            detail=f"Failed to communicate with Ollama server. Ensure 'ollama serve' is running on localhost:11434. Error: {e}"
                        )
    # ******************************************** LANGFUSE *****************************
    # ******************************************** LANGFUSE *****************************
    else:

            print("Preparing Qdrant Client DB. . . ")
            qdrant_client = get_qdrant_client()
            last_message = request.messages[-1].content  
            print("Searching & Reranking . . ")
            contexts = []
            available_collections = [c.name for c in qdrant_client.get_collections().collections]
            target_collections = ["railway"] if "railway" in available_collections else available_collections
            for c in target_collections:
                contexts.extend(pure_retrieval(qdrant_client, c, last_message,
                                retrieve_k=30, use_reranker=True))
        
            if not is_ollama:
                # Local HF models need the GPU freed; Ollama inference is server-side,
                # so keep the retrieval models hot and skip the reload penalty per request.
                from rag_sn_in.llm.reranker import unload_reranker
                from rag_sn_in.llm.embedding import unload_embedder
                unload_reranker()
                unload_embedder()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Retrieval models unloaded from VRAM.")
        
            print(f"Flattened {len(contexts)} chunks")
            if contexts:
                print(type(contexts[0]), contexts[0].payload.get("page_start"))
        
            messages = [SYSTEM_PROMPT] + [m.model_dump() for m in request.messages]
            augmented = augment_last_user_message(messages, contexts)
            request_uuid = str(uuid.uuid4())
        
            if is_ollama:
                # Ollama applies the gemma4 chat renderer server-side to the raw
                # messages list — never pre-format a prompt string for /api/chat.
                ollama_model = OLLAMA_MODELS[LLM_NAME.lower()]
                print(f"INFO: Using Ollama model '{ollama_model}'")
        
                if request.stream:
                    def gen():
                        try:
                            stream = ollama.chat(
                                model=ollama_model,
                                messages=augmented,
                                stream=True,
                                options=OLLAMA_OPTIONS,
                            )
                            for chunk in stream:
                                text_chunk = chunk["message"]["content"]
                                if text_chunk:
                                    yield _completion_chunk(
                                        request_uuid,
                                        {"role": "assistant", "content": text_chunk},
                                    )
                            yield _completion_chunk(request_uuid, {}, finish_reason="stop")
                            yield "data: [DONE]\n\n"
                        except Exception as e:
                            print(f"ERROR connecting to Ollama: {e}")
                            yield _completion_chunk(
                                request_uuid,
                                {"role": "assistant", "content": f"\n\n[Error: Could not connect to Ollama server. Is 'ollama serve' running on localhost:11434? ({e})]"},
                                finish_reason="error"
                            )
                            yield "data: [DONE]\n\n"
        
                    return StreamingResponse(gen(), media_type="text/event-stream")
        
                try:
                    response = ollama.chat(
                        model=ollama_model,
                        messages=augmented,
                        stream=False,
                        options=OLLAMA_OPTIONS,
                    )
                    return _completion_response(request_uuid, response["message"]["content"])
                except Exception as e:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Failed to communicate with Ollama server. Ensure 'ollama serve' is running on localhost:11434. Error: {e}"
                    )
            else:
                tokenizer, model = get_model()
                prompt_text = tokenizer.apply_chat_template(
                    augmented,
                    tokenize=False,
                    add_generation_prompt=True
                )
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
                
                if request.stream:
                    print("→ STREAMING path")
                    def gen():
                        for text_chunk in streamer:
                            yield _completion_chunk(
                                request_uuid,
                                {"role": "assistant", "content": text_chunk},
                            )
                        yield _completion_chunk(request_uuid, {}, finish_reason="stop")
                        yield "data: [DONE]\n\n"

                    return StreamingResponse(gen(), media_type="text/event-stream")

                full_text = "".join(text_chunk for text_chunk in streamer)
                return _completion_response(request_uuid, full_text)
