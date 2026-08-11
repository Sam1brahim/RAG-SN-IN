from rag_sn_in.database.client import get_client
from rag_sn_in.llm.reranker import rerank
from rag_sn_in.database.db_setup import ensure_ToCreate_collection
import json
from rag_sn_in.llm.embedding import embed_queries_batch
import time
import os
import glob
from rag_sn_in.config import VECTOR_SIZE, RERANKER_NAME,EMBEDDING_MODEL_NAME
from rag_sn_in.llm.generator import clear_vram



def load_eval_set(path):
    """
    Loads eval data from either:
    - a single .jsonl file, or
    - a directory containing multiple .jsonl files (all will be merged).
    """
    all_items = []

    if os.path.isdir(path):
        jsonl_files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
        if not jsonl_files:
            raise ValueError(f"No .jsonl files found in directory: {path}")

        for file_path in jsonl_files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_items.append(json.loads(line))

        print(f"Loaded {len(all_items)} items from {len(jsonl_files)} files in {path}")

    elif os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_items.append(json.loads(line))
        print(f"Loaded {len(all_items)} items from {path}")

    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    return all_items



def evaluate_retrieval(client, collection_name, eval_set, k_values=(1, 3, 5, 10),
                        retrieve_k=30, use_reranker=False):
    results_per_k = {k: {"hits": 0, "mrr_total": 0} for k in k_values}
    max_k = max(k_values)
    n = len(eval_set)
    
    # NEW: List to store the extracted contexts for RAGAS
    ragas_dataset = [] 

    print(f"\n{'='*60}")
    print(f"Starting evaluation | {n} questions | reranker={'ON' if use_reranker else 'OFF'}")
    if use_reranker:
        print(f"Stage 1: dense_search(top_k={retrieve_k}) -> Stage 2: rerank(top_k={max_k})")
    print(f"{'='*60}\n")

    # ---- PRECOMPUTE EMBEDDINGS BATCH ----
    print(f"Precomputing embeddings for {n} questions in batch...")
    t_embed_start = time.time()
    questions = [item["question"] for item in eval_set]

    # Generate all vectors at once using your new imported function
    query_vectors = embed_queries_batch(questions)

    embed_time = time.time() - t_embed_start
    print(f"Batch embedding complete in {embed_time:.1f}s")
    print(f"{'-'*60}\n")

    start = time.time()
    dense_time_total = 0.0
    rerank_time_total = 0.0

    # zip pairs the item with its precomputed vector
    for idx, (item, vector) in enumerate(zip(eval_set, query_vectors), start=1):
        # ---- Stage 1: dense retrieval ----
        t0 = time.time()

        retrieved = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=retrieve_k,
            with_payload=True # Payload contains the text!
        )
        retrieved = retrieved.points
        t1 = time.time()
        dense_time_total += (t1 - t0)

        # ---- Stage 2: reranking ----
        if use_reranker:    
            retrieved = rerank(item["question"], retrieved, top_k=max_k)
        else:
            retrieved = retrieved[:max_k]
        t2 = time.time()
        rerank_time_total += (t2 - t1)

        # ---- METRICS CALCULATION ----
        retrieved_ids = [r.payload["id"] for r in retrieved]
        gold_ids = set(item["gold_chunk_ids"])

        for k in k_values:
            top_k_ids = retrieved_ids[:k]
            hit_rank = next((i + 1 for i, rid in enumerate(top_k_ids) if rid in gold_ids), None)
            if hit_rank:
                results_per_k[k]["hits"] += 1
                results_per_k[k]["mrr_total"] += 1 / hit_rank

        # ---- NEW: EXTRACT TOP 5 CONTEXTS FOR RAGAS ----
        top_5_contexts = [r.payload["text"] for r in retrieved[:5]]
        
        # Append to our RAGAS list
        ragas_dataset.append({
            "question": item["question"],
            "contexts": top_5_contexts,
            "ground_truth": item["gold_answer"]
        })

        # ---- Per-item verbose print ----
        got_hit = any(rid in gold_ids for rid in retrieved_ids[:max_k])
        status = "HIT " if got_hit else "MISS"

        if idx % 25 == 0 or idx == n:
            elapsed = time.time() - start
            rate = idx / elapsed
            eta = (n - idx) / rate
            avg_dense = dense_time_total / idx
            avg_rerank = rerank_time_total / idx if use_reranker else 0.0

            print(f"[{idx}/{n}] {status} | item id: {item.get('id', '?')} | "
                  f"elapsed={elapsed:.1f}s | ETA={eta:.1f}s | "
                  f"avg_dense={avg_dense*1000:.0f}ms | avg_rerank={avg_rerank*1000:.0f}ms")

    # [Metrics printing omitted for brevity, same as yours]

    # Package the final dictionary of metrics
    final_metrics = {
        k: {
            "hit_rate": v["hits"] / n,
            "mrr": v["mrr_total"] / n
        }
        for k, v in results_per_k.items()
    }

    # Latency/throughput stats for runtime monitoring & load simulation
    wall_time = time.time() - start
    retrieval_stats = {
        "num_queries": n,
        "retrieve_k": retrieve_k,
        "reranker_enabled": use_reranker,
        "batch_embed_time_s": round(embed_time, 2),
        "wall_time_s": round(wall_time, 2),
        "avg_dense_latency_ms": round(dense_time_total / n * 1000, 1),
        "avg_rerank_latency_ms": round(rerank_time_total / n * 1000, 1),
        "avg_total_latency_ms": round((dense_time_total + rerank_time_total) / n * 1000, 1),
        "throughput_queries_per_s": round(n / wall_time, 2) if wall_time else None,
    }

    return final_metrics, use_reranker, ragas_dataset, retrieval_stats

def generate_answer(question, chunks):
    pass


def evaluate_answer(question, answer, reference_answer, chunks):
    pass


if __name__ == "__main__":
    
    collection_name = "DRR_SNCF"
    client = get_client()
    ensure_ToCreate_collection(client, collection_name, vector_size=VECTOR_SIZE)

    eval_set = load_eval_set(r"E:\Project RAG-SN-IN\data\eval\eval chunks 512 max\realistic")

    metrics, use_reranker, _, retrieval_stats = evaluate_retrieval(client, collection_name, eval_set, retrieve_k=30, use_reranker=True)

    if use_reranker:
        print(f"Final Metrics With Reranker \n {RERANKER_NAME}. \n Embedding model {EMBEDDING_MODEL_NAME}:")
    else:
        print(f"Final Metrics Without Reranker\n Embedding model {EMBEDDING_MODEL_NAME}:")
    for k, m in metrics.items():
        print(f"k={k}: hit_rate={m['hit_rate']:.3f}, mrr={m['mrr']:.3f}")
    print("Retrieval stats:", retrieval_stats)