from rag_sn_in.llm.reranker import rerank
import json
from rag_sn_in.llm.embedding import embed_query
import os
import glob


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