import os
from dotenv import load_dotenv 
from rag_sn_in.database.client import get_client as get_qdrant_client
from rag_sn_in.database.db_setup import ensure_ToCreate_collection
from langchain_deepseek import ChatDeepSeek
from rag_sn_in.llm.generator import load_generation_llm, clear_vram, verify_llm_name
from rag_sn_in.config import LLM_NAME, VECTOR_SIZE, EMBEDDING_MODEL_NAME, RERANKER_NAME
from rag_sn_in.eval.evaluate_RAG import evaluate_retrieval,load_eval_set
import gc
import torch
import time
from datasets import Dataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
import random
import json
from langchain_core.callbacks import BaseCallbackHandler
from rag_sn_in.observability.client import _enabled, get_client as get_langfuse_client
from rag_sn_in.processing import animate

# The main launchable code needs a: Auto Session_id, User_id (both should be asked when UI is launched automatically) and other meta data like:
#user_id = "demo_user_001"
#session_id = "session_2026_08_11_001"
#request_id = "req_uuid_..."
#trace_name = "drr_question_answering"
#environment = "development"  # development / staging / production
#release = "v0.1.0"

def main():

    # Enabling Langfuse
        if _enabled:
            print("Langfuse For Observability is Enabled. \n")
            print("Starting Langfuse . . . \n")
            langfuse = get_langfuse_client()
            with langfuse.start_as_current_observation(
                as_type="retriever ",
                name="Call-Qdrant"
            ) as retriever:
                collection_name = ensure_ToCreate_collection(client, collection_name, vector_size=VECTOR_SIZE)
                retriever.update(output=collection_name)
            print("Launching RAGAS Pipeline: \n")
            print("Performing Reranking ...\n")
            
            eval_set = load_eval_set(r"E:\Project RAG-SN-IN\data\eval\eval chunks 512 max\realistic")
            _, _, ragas_data, retrieval_stats = evaluate_retrieval(client, collection_name, eval_set, retrieve_k=30, use_reranker=True)
        else:
            print("Langfuse Disabled \n")  