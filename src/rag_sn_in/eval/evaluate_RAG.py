from rag_sn_in.database.client import get_client
from rag_sn_in.llm.reranker import rerank
from rag_sn_in.database.db_setup import ensure_ToCreate_collection
import json
from rag_sn_in.llm.embedding import embed_queries_batch, embed_query
import time
import os
import glob
from rag_sn_in.config import (
    VECTOR_SIZE, RERANKER_NAME, EMBEDDING_MODEL_NAME,
    DATA_EVAL_DIR, DATA_EVAL_METRICS_DIR,
)
import random
from qdrant_client.models import FieldCondition, Filter, MatchValue


def load_eval_set(path):
    """
    Loads eval data from either:
    - a single .jsonl file, or
    - a directory containing multiple .jsonl files (all will be merged).
    """
    def normalize_item(item):
        """Ensures consistent keys and list-based gold IDs."""
        # 1. Normalize Question
        if "Question" in item:
            item["question"] = item.pop("Question")
        
        # 2. Normalize Ground Truth
        if "golden_answer" in item:
            item["ground_truth"] = item.get("golden_answer") # keep both or replace? evaluate_RAG uses golden_answer later
        elif "ground_truth" in item:
            item["golden_answer"] = item["ground_truth"]

        # 3. Normalize Gold IDs
        # We want a list of strings in 'gold_chunk_ids'
        gids = item.get("gold_chunk_ids") or item.get("Gold_chunk_id") or item.get("gold_ids")
        if isinstance(gids, str):
            item["gold_chunk_ids"] = [gids]
        elif isinstance(gids, list):
            item["gold_chunk_ids"] = gids
        else:
            item["gold_chunk_ids"] = []
            
        # Also keep Gold_chunk_id for backward compatibility if needed, 
        # but the code should ideally use gold_chunk_ids
        item["Gold_chunk_id"] = item["gold_chunk_ids"] 
        
        return item

    all_items = []

    if os.path.isdir(path):
        jsonl_files = sorted(glob.glob(os.path.join(path, "*.json")))
        if not jsonl_files:
            raise ValueError(f"No .jsonl files found in directory: {path}")

        for file_path in jsonl_files:
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    # Try loading as a single JSON object (e.g. a list)
                    data = json.load(f)
                    if isinstance(data, list):
                        all_items.extend([normalize_item(i) for i in data])
                    else:
                        all_items.append(normalize_item(data))
                except json.JSONDecodeError:
                    # Fallback to JSONL
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                all_items.append(normalize_item(json.loads(line)))
                            except json.JSONDecodeError as e:
                                print(f"Skipping invalid JSON line in {file_path}: {e}")

        print(f"Loaded {len(all_items)} items from {len(jsonl_files)} files in {path}")

    elif os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    all_items.extend([normalize_item(i) for i in data])
                else:
                    all_items.append(normalize_item(data))
            except json.JSONDecodeError:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        all_items.append(normalize_item(json.loads(line)))
        print(f"Loaded {len(all_items)} items from {path}")

    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    return all_items


def pure_retrieval(client, collection_name, query, k_values=(1, 3, 5, 10),
                        retrieve_k=30, use_reranker=False):
    #results_per_k = {k: {"hits": 0, "mrr_total": 0} for k in k_values}
    #max_k = max(k_values)
    
    print(f"\n{'='*60}")
    print(f"Starting retrieval | reranker={'ON' if use_reranker else 'OFF'}")
    if use_reranker:
        print(f"Stage 1: dense_search(top_k={retrieve_k}) -> Stage 2: rerank(top_k=10)")
    print(f"{'='*60}\n")

    # ---- PRECOMPUTE EMBEDDINGS BATCH ----
    print(f"Precomputing embeddings for the query...")
    # Generate all vectors at once using your new imported function
    query_vector = embed_query(query)

    # ---- Stage 1: dense retrieval ----
    retrieved = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=retrieve_k,
                with_payload=True # Payload contains the text!
            )
    
    retrieved = retrieved.points
    if not retrieved:
        return []

    # ---- Stage 2: reranking ----
    if use_reranker:    
        retrieved = rerank(query, retrieved, top_k=10)
    else:
        retrieved = retrieved[:10]

    # ---- EXTRACT TOP 5 CONTEXTS FOR RAGAS ----
    top_5_contexts = retrieved[:5] #to keep meta data needed for citation for llm generation and their responses
    
    return top_5_contexts
def preflight_id_check(client, collection_name, eval_set, sample_size=5):
    """
    Verify gold_chunk_ids exist in payload space before burning a full eval run.
    A format mismatch (int vs str, different ID scheme) yields 0% hit rate that
    looks like catastrophic retrieval failure.
    """
    sample_size = min(sample_size, len(eval_set))
    sample = random.sample(eval_set, sample_size)

    for item in sample:
        for gid in item["gold_chunk_ids"]:
            points, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="id", match=MatchValue(value=gid))]
                ),
                limit=1,
                with_payload=True,
            )
            assert points, (
                f"PREFLIGHT FAILED: gold ID {gid!r} (type {type(gid).__name__}) "
                f"not found in collection payload 'id' field — format mismatch?"
            )
            assert "text" in points[0].payload, (
                f"PREFLIGHT FAILED: payload for {gid!r} has no 'text' field"
            )

    print(f"Pre-flight check passed: {sample_size} questions, "
          f"gold IDs resolvable in payload, 'text' field present\n")


def evaluate_retrieval(client, collection_name, eval_set, k_values=(1, 3, 5, 10),
                       retrieve_k=30, use_reranker=False,
                       preflight=True, misses_path=None):
    results_per_k = {k: {"hits": 0, "mrr_total": 0} for k in k_values}
    max_k = max(k_values)
    n = len(eval_set)

    # RAGAS / generation context
    ragas_dataset = []

    # NEW: miss log + gold recall (multi-chunk leniency check)
    misses = []
    gold_recalls = []          # per-question fraction of gold chunks retrieved @ max_k
    multi_chunk_recalls = []   # same, restricted to questions with >1 gold chunk

    print(f"\n{'='*60}")
    print(f"Starting evaluation | {n} questions | reranker={'ON' if use_reranker else 'OFF'}")
    if use_reranker:
        print(f"Stage 1: dense_search(top_k={retrieve_k}) -> Stage 2: rerank(top_k={max_k})")
    print(f"{'='*60}\n")

    # ---- PRE-FLIGHT: gold ID format alignment ----
    if preflight:
        preflight_id_check(client, collection_name, eval_set)

    # ---- PRECOMPUTE EMBEDDINGS BATCH ----
    print(f"Precomputing embeddings for {n} questions in batch...")
    t_embed_start = time.time()
    questions = [item["question"] for item in eval_set]
    query_vectors = embed_queries_batch(questions)
    embed_time = time.time() - t_embed_start
    print(f"Batch embedding complete in {embed_time:.1f}s")
    print(f"{'-'*60}\n")

    query_limit = retrieve_k if use_reranker else max_k

    start = time.time()
    dense_time_total = 0.0
    rerank_time_total = 0.0
    stage1_hits = 0

    for idx, (item, vector) in enumerate(zip(eval_set, query_vectors), start=1):
        # ---- Stage 1: dense retrieval ----
        t0 = time.time()
        retrieved_stage1 = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=query_limit,
            with_payload=True,
        ).points
        t1 = time.time()
        dense_time_total += (t1 - t0)

        # Track Stage 1 hit
        stage1_ids = [r.payload["id"] for r in retrieved_stage1]
        gold_ids = set(item["gold_chunk_ids"])
        if any(rid in gold_ids for rid in stage1_ids):
            stage1_hits += 1

        # ---- Stage 2: reranking ----
        if use_reranker:
            retrieved = rerank(item["question"], retrieved_stage1, top_k=max_k)
        else:
            retrieved = retrieved_stage1[:max_k]
        t2 = time.time()
        rerank_time_total += (t2 - t1)

        # ---- METRICS ----
        retrieved_ids = [r.payload["id"] for r in retrieved]
        gold_ids = set(item["gold_chunk_ids"])

        for k in k_values:
            top_k_ids = retrieved_ids[:k]
            hit_rank = next((i + 1 for i, rid in enumerate(top_k_ids) if rid in gold_ids), None)
            if hit_rank:
                results_per_k[k]["hits"] += 1
                results_per_k[k]["mrr_total"] += 1 / hit_rank

        # ---- NEW: gold recall @ max_k (catches multi-chunk leniency) ----
        gold_recall = len(gold_ids & set(retrieved_ids[:max_k])) / len(gold_ids) if gold_ids else 0
        gold_recalls.append(gold_recall)
        if len(gold_ids) > 1:
            multi_chunk_recalls.append(gold_recall)

        # ---- RAGAS contexts (explicit top-5) ----
        top_5_contexts = [r.payload["text"] for r in retrieved[:5]]
        ragas_dataset.append({
            "question": item["question"],
            "contexts": top_5_contexts,
            "ground_truth": item["ground_truth"],
        })

        # ---- Per-item status: print every MISS live, periodic otherwise ----
        got_hit = any(rid in gold_ids for rid in retrieved_ids[:max_k])

        if not got_hit:
            # NEW: log the miss for post-hoc effective-hit-rate review
            misses.append({
                "id": item.get("id"),
                "question": item["question"],
                "gold_ids": list(item["gold_chunk_ids"]),
                "retrieved_ids": retrieved_ids[:max_k],
                "top_retrieved_preview": [
                    r.payload["text"][:300] for r in retrieved[:3]
                ],
            })
            print(f"[{idx}/{n}] MISS | gold={list(item['gold_chunk_ids'])}")
            print(f"    Top 3 retrieved IDs: {retrieved_ids[:3]}")
            if use_reranker:
                 # Also show stage 1 top IDs to see if reranker dropped the hit
                 stage1_top3 = [r.payload["id"] for r in retrieved_stage1[:3]]
                 print(f"    Stage 1 Top 3 IDs: {stage1_top3}")
            
            # Check if the gold ID was even in Stage 1 results
            if any(rid in gold_ids for rid in stage1_ids):
                rank_in_stage1 = next(i for i, rid in enumerate(stage1_ids) if rid in gold_ids) + 1
                print(f"    !!! Gold ID found in Stage 1 at rank {rank_in_stage1}, but dropped by Reranker or outside top {max_k}")
            else:
                print(f"    Gold ID NOT found in Stage 1 (top {query_limit})")

        if idx % 25 == 0 or idx == n:
            elapsed = time.time() - start
            rate = idx / elapsed
            eta = (n - idx) / rate
            avg_dense = dense_time_total / idx
            avg_rerank = rerank_time_total / idx if use_reranker else 0.0

            print(f"[{idx}/{n}] {'HIT ' if got_hit else 'MISS'} | "
                  f"item id: {item.get('id', '?')} | "
                  f"elapsed={elapsed:.1f}s | ETA={eta:.1f}s | "
                  f"avg_dense={avg_dense*1000:.0f}ms | avg_rerank={avg_rerank*1000:.0f}ms")

    # ---- Final metrics ----
    final_metrics = {
        k: {
            "hit_rate": v["hits"] / n,
            "mrr": v["mrr_total"] / n,
        }
        for k, v in results_per_k.items()
    }

    # NEW: aggregate gold-recall stats
    final_metrics["gold_recall"] = {
        "mean_at_max_k": sum(gold_recalls) / n,
        "multi_chunk_count": len(multi_chunk_recalls),
        "multi_chunk_mean": (
            sum(multi_chunk_recalls) / len(multi_chunk_recalls)
            if multi_chunk_recalls else None
        ),
    }

    wall_time = time.time() - start
    retrieval_stats = {
        "num_queries": n,
        "retrieve_k": query_limit,
        "reranker_enabled": use_reranker,
        "stage1_hit_rate": round(stage1_hits / n, 3),
        "batch_embed_time_s": round(embed_time, 2),
        "wall_time_s": round(wall_time, 2),
        "avg_dense_latency_ms": round(dense_time_total / n * 1000, 1),
        "avg_rerank_latency_ms": round(rerank_time_total / n * 1000, 1),
        "avg_total_latency_ms": round((dense_time_total + rerank_time_total) / n * 1000, 1),
        "throughput_queries_per_s": round(n / wall_time, 2) if wall_time else None,
        "miss_count": len(misses),
    }

    # NEW: persist misses for manual effective-hit-rate review
    if misses_path:
        with open(misses_path, "w", encoding="utf-8") as f:
            json.dump({
                "reranker_enabled": use_reranker,
                "strict_hit_rate_at_max_k": final_metrics[max_k]["hit_rate"],
                "miss_count": len(misses),
                "misses": misses,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nMisses saved to {misses_path} ({len(misses)} misses to review)")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    for k in sorted(k_values):
        m = final_metrics[k]
        print(f"  k={k:2d} | hit_rate={m['hit_rate']:.3f} | MRR={m['mrr']:.3f}")
    
    print(f"  Stage 1 (dense) hit_rate@{query_limit}: {stage1_hits/n:.3f}")
    
    gr = final_metrics["gold_recall"]
    print(f"  gold_recall@{max_k}: mean={gr['mean_at_max_k']:.3f} | "
          f"multi-chunk ({gr['multi_chunk_count']} q): "
          f"{gr['multi_chunk_mean']:.3f}" if gr["multi_chunk_mean"] is not None
          else f"  gold_recall@{max_k}: mean={gr['mean_at_max_k']:.3f}")
    print(f"  misses: {len(misses)}/{n}")
    print(f"{'='*60}\n")

    return final_metrics, use_reranker, ragas_dataset, retrieval_stats


if __name__ == "__main__":
    collection_name = "railway"
    client = get_client()
    ensure_ToCreate_collection(client, collection_name, vector_size=VECTOR_SIZE)

    eval_set = load_eval_set(str(DATA_EVAL_DIR))

    metrics, use_reranker, _, retrieval_stats = evaluate_retrieval(
        client, collection_name, eval_set, retrieve_k=30, use_reranker=False
    )

    if use_reranker:
        print(f"Final Metrics With Reranker \n {RERANKER_NAME}. \n Embedding model {EMBEDDING_MODEL_NAME}:")
    else:
        print(f"Final Metrics Without Reranker\n Embedding model {EMBEDDING_MODEL_NAME}:")
    for k, m in metrics.items():
        if isinstance(k, int):
            print(f"k={k}: hit_rate={m['hit_rate']:.3f}, mrr={m['mrr']:.3f}")
    print("Retrieval stats:", retrieval_stats)

    # ---------------- Save Summary ----------------
    DATA_EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = str(DATA_EVAL_METRICS_DIR)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Clean model names for filename
    emb_tag = EMBEDDING_MODEL_NAME.split("/")[-1]
    rerank_tag = RERANKER_NAME.split("/")[-1] if use_reranker else "no_reranker"
    
    summary = {
        "timestamp": timestamp,
        "config": {
            "embedding_model": EMBEDDING_MODEL_NAME,
            "reranker": RERANKER_NAME if use_reranker else None,
            "collection": collection_name,
            "vector_size": VECTOR_SIZE,
        },
        "retrieval_metrics": metrics,
        "retrieval_performance": retrieval_stats,
    }
    
    summary_path = os.path.join(out_dir, f"retrieval_eval_{emb_tag}_{rerank_tag}_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nEvaluation summary saved to: {summary_path}")