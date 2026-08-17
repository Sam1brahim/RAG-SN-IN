"""
RAG-SN-IN — Interactive CLI Entry Point
========================================
Railway Regulatory Intelligence Platform

Guides the user through model selection, database initialisation,
and pipeline execution (retrieval eval, RAGAS eval, or API server).

Usage:
    .venv\Scripts\python start.py        # Windows
    .venv/bin/python start.py            # Linux/macOS
    python start.py                      # if venv is activated
"""

import os
import sys
import importlib
from pathlib import Path

# ── Resolve project root regardless of where the script is invoked from ──
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Ensure we're running inside the project's virtual environment ────────
_VENV_PYTHON_WIN = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
_VENV_PYTHON_UNIX = PROJECT_ROOT / ".venv" / "bin" / "python"

def _in_venv() -> bool:
    """Detect whether the current interpreter is the project venv."""
    prefix = Path(sys.prefix).resolve()
    return prefix == (PROJECT_ROOT / ".venv").resolve()

if not _in_venv():
    venv_python = _VENV_PYTHON_WIN if _VENV_PYTHON_WIN.exists() else _VENV_PYTHON_UNIX
    if venv_python.exists():
        print(f"  ⚠  Not running inside the project virtual environment.")
        print(f"     Relaunching with: {venv_python}\n")
        os.execv(str(venv_python), [str(venv_python), str(PROJECT_ROOT / "start.py")] + sys.argv[1:])
    else:
        print("  ✖  No .venv found. Run:  python -m venv .venv  &&  uv sync")
        sys.exit(1)

# ── Model catalogues ──────────────────────────────────────────────────────

EMBEDDING_MODELS = {
    1: {"name": "google/embeddinggemma-300m",        "dims": 768,  "label": "EmbeddingGemma-300M  (768-d) — Recommended"},
    2: {"name": "intfloat/multilingual-e5-large",     "dims": 1024, "label": "Multilingual-E5-Large (1024-d)"},
    3: {"name": "BAAI/bge-m3",                        "dims": 1024, "label": "BGE-M3                (1024-d)"},
    4: {"name": "nomic-ai/nomic-embed-text-v1.5",     "dims": 768,  "label": "Nomic-Embed-v1.5      (768-d)"},
}

RERANKERS = {
    1: {"name": "BAAI/bge-reranker-v2-m3",           "label": "BGE-Reranker-v2-m3   — Recommended"},
    2: {"name": "Qwen/Qwen3-Reranker-0.6B",           "label": "Qwen3-Reranker-0.6B"},
    3: {"name": None,                                  "label": "No reranker (dense-only baseline)"},
}

GENERATION_LLMS = {
    1: {"name": "gemma4-e2b",                         "label": "Gemma4-E2B via Ollama  — Recommended (0 VRAM)"},
    2: {"name": "qwen",                               "label": "Qwen2.5-7B-Instruct 4-bit (local HF)"},
    3: {"name": "gemma",                              "label": "Gemma-2-9B-it 4-bit      (local HF)"},
}

# ── Helpers ───────────────────────────────────────────────────────────────

def banner():
    print()
    print("=" * 64)
    print("   RAG-SN-IN  ·  Railway Regulatory Intelligence Platform")
    print("   Part 1 — Retrieval Engineering & Evaluation")
    print("=" * 64)
    print()


def pick(title: str, options: dict, allow_none=False):
    """Display a numbered menu and return the selected dict."""
    print(f"\n  {title}")
    print(f"  {'─' * (len(title))}")
    for key, opt in options.items():
        print(f"    [{key}]  {opt['label']}")
    if allow_none:
        print(f"    [0]  Skip / None")
    print()

    while True:
        raw = input(f"  Enter choice (1-{len(options)}): ").strip()
        if allow_none and raw == "0":
            return None
        try:
            choice = int(raw)
            if choice in options:
                selected = options[choice]
                print(f"  ✔  Selected: {selected['label']}\n")
                return selected
        except ValueError:
            pass
        print("  ✖  Invalid choice — try again.")


def ask_yes_no(question: str) -> bool:
    while True:
        raw = input(f"  {question} (y/n): ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n.")


def check_env_for_ragas() -> bool:
    """Check whether .env exists with a DEEPSEEK_API_KEY."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return False
    content = env_path.read_text(encoding="utf-8")
    return "DEEPSEEK_API_KEY" in content and "your_" not in content


def write_config(embedding: dict, reranker: dict | None, llm: dict):
    """Update config.py with the user's selections."""
    config_path = SRC_DIR / "rag_sn_in" / "config.py"
    reranker_name = reranker["name"] if reranker else "BAAI/bge-reranker-v2-m3"

    content = f'''"""
Shared project configuration.

Single source of truth for the embedding model name and data paths, so
chunking (token counting), embedding, indexing and eval tooling never
drift apart.

All paths are derived from the project root — no hardcoded absolute paths.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL_NAME = "{embedding["name"]}"

VECTOR_SIZE = {embedding["dims"]}

RERANKER_NAME = "{reranker_name}"

LLM_NAME = "{llm["name"]}"
EVALUATOR = "deepseek-chat"

DATA_PROCESSED_TEXT_DIR = ROOT / "data" / "processed" / "text"
DATA_CHUNKS_DIR = ROOT / "data" / "processed" / "chunks"
DATA_CHUNKS_512_DIR = DATA_CHUNKS_DIR / "max token 512"
DATA_EVAL_DIR = ROOT / "data" / "eval" / "eval chunks 512 max"
DATA_RAGAS_DIR = ROOT / "data" / "ragas"
DATA_EVAL_METRICS_DIR = ROOT / "data" / "Eval RAG metrics" / "ragas"
DATA_RAW_DIR = ROOT / "data" / "raw"
'''
    config_path.write_text(content, encoding="utf-8")
    print(f"  ✔  config.py updated (embedding={embedding['name']}, dims={embedding['dims']})")


def ensure_indexed(embedding: dict, reranker_on: bool):
    """Check if Qdrant has data; if not, run indexing."""
    from rag_sn_in.database.client import get_client
    from rag_sn_in.database.db_setup import ensure_ToCreate_collection
    from rag_sn_in.ingestion.indexing import index_chunks
    from rag_sn_in.config import VECTOR_SIZE, DATA_CHUNKS_512_DIR

    client = get_client()
    collection = "railway"

    ensure_ToCreate_collection(client, collection, vector_size=VECTOR_SIZE)

    # Check if collection has points
    info = client.get_collection(collection)
    count = info.points_count if hasattr(info, "points_count") else 0

    if count > 0:
        print(f"  ✔  Collection '{collection}' already has {count} vectors. Skipping indexing.\n")
        return

    print(f"\n  ⚠  Collection '{collection}' is empty — running indexing pipeline...")
    print(f"     Embedding model: {embedding['name']}  ({embedding['dims']}-dim)")
    print(f"     Chunks directory: {DATA_CHUNKS_512_DIR}\n")

    if not DATA_CHUNKS_512_DIR.exists():
        print(f"  ✖  Chunks directory not found: {DATA_CHUNKS_512_DIR}")
        print(f"     Run the ingestion + chunking pipeline first.")
        sys.exit(1)

    # Re-import embedding module to pick up the updated config
    import rag_sn_in.llm.embedding as emb_mod
    importlib.reload(emb_mod)

    total = index_chunks(str(DATA_CHUNKS_512_DIR), client, emb_mod.embed_document, collection)
    print(f"\n  ✔  Indexed {total} chunks into '{collection}'.\n")


# ── Pipeline runners ─────────────────────────────────────────────────────

def run_retrieval_eval(embedding: dict, reranker: dict | None):
    """Run the retrieval-only evaluation (Hit@K, MRR, Gold Recall)."""
    write_config(embedding, reranker, GENERATION_LLMS[1])  # LLM not used but config needs one

    ensure_indexed(embedding, reranker is not None)

    from rag_sn_in.database.client import get_client
    from rag_sn_in.database.db_setup import ensure_ToCreate_collection
    from rag_sn_in.eval.evaluate_RAG import load_eval_set, evaluate_retrieval
    from rag_sn_in.config import VECTOR_SIZE, EMBEDDING_MODEL_NAME, RERANKER_NAME, DATA_EVAL_DIR, DATA_EVAL_METRICS_DIR

    collection_name = "railway"
    client = get_client()
    ensure_ToCreate_collection(client, collection_name, vector_size=VECTOR_SIZE)

    eval_set = load_eval_set(str(DATA_EVAL_DIR))

    use_reranker = reranker is not None and reranker["name"] is not None
    metrics, _, _, retrieval_stats = evaluate_retrieval(
        client, collection_name, eval_set,
        retrieve_k=30, use_reranker=use_reranker
    )

    # ── Print results ──
    print("\n" + "=" * 60)
    print("  RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    for k, m in metrics.items():
        if isinstance(k, int):
            print(f"    k={k:2d}  |  hit_rate={m['hit_rate']:.3f}  |  MRR={m['mrr']:.3f}")

    gr = metrics.get("gold_recall", {})
    print(f"\n    Gold Recall:  {gr.get('mean_at_max_k', 'N/A'):.3f}")
    print(f"    Misses:       {retrieval_stats['miss_count']}/{retrieval_stats['num_queries']}")
    print(f"    Avg Latency:  {retrieval_stats['avg_total_latency_ms']:.0f} ms")
    print("=" * 60)

    # ── Persist summary ──
    import json, time
    DATA_EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
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

    summary_path = DATA_EVAL_METRICS_DIR / f"retrieval_eval_{emb_tag}_{rerank_tag}_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  ✔  Results saved to: {summary_path}")


def run_ragas_eval(embedding: dict, reranker: dict | None, llm: dict):
    """Run the full RAGAS pipeline (retrieval + generation + judge)."""
    write_config(embedding, reranker, llm)

    # ── .env check ──
    if not check_env_for_ragas():
        print("\n  ⚠  RAGAS evaluation requires a DeepSeek API key.")
        print("     Create a .env file in the project root with:")
        print("       DEEPSEEK_API_KEY=your_key_here\n")

        if not ask_yes_no("Have you created the .env file with your DEEPSEEK_API_KEY?"):
            print("\n  ✖  Please set up your .env file and re-run.")
            sys.exit(0)

        if not check_env_for_ragas():
            print("\n  ✖  .env file still not detected. Exiting.")
            sys.exit(1)

    print("\n  ✔  DeepSeek API key found in .env\n")

    ensure_indexed(embedding, reranker is not None)

    # Now import and run PRE_RAGAS
    from rag_sn_in.llm import PRE_RAGAS
    importlib.reload(PRE_RAGAS)
    PRE_RAGAS.main()


def run_api_server(embedding: dict, reranker: dict | None, llm: dict):
    """Launch the FastAPI server."""
    write_config(embedding, reranker, llm)

    ensure_indexed(embedding, reranker is not None)

    print("\n  Starting FastAPI server on http://0.0.0.0:8000")
    print("  Connect Open WebUI to http://localhost:8000/v1\n")

    import uvicorn
    uvicorn.run(
        "rag_sn_in.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


# ── Main flow ────────────────────────────────────────────────────────────

def main():
    banner()

    # ── Step 1: Choose pipeline ──
    print("  What would you like to do?\n")
    print("    [1]  Evaluate Retrieval  (Hit@K, MRR, Gold Recall)")
    print("    [2]  Run RAGAS Evaluation  (Retrieval + Generation + LLM Judge)")
    print("    [3]  Launch API Server  (OpenAI-compatible, for Open WebUI)")
    print()

    while True:
        raw = input("  Enter choice (1-3): ").strip()
        if raw in ("1", "2", "3"):
            mode = int(raw)
            break
        print("  ✖  Invalid choice — try again.")

    # ── Step 2: Choose models ──
    embedding = pick("Select Embedding Model", EMBEDDING_MODELS)

    reranker = pick("Select Reranker", RERANKERS, allow_none=True)
    if reranker is not None and reranker.get("name") is None:
        reranker = None  # "No reranker" selected

    llm = None
    if mode in (2, 3):
        llm = pick("Select Generation LLM", GENERATION_LLMS)
    else:
        llm = GENERATION_LLMS[1]  # default for config consistency

    # ── Step 3: Execute ──
    print("\n" + "─" * 64)
    print("  CONFIGURATION SUMMARY")
    print("─" * 64)
    print(f"    Embedding:  {embedding['name']}  ({embedding['dims']}-dim)")
    print(f"    Reranker:   {reranker['name'] if reranker else 'None'}")
    print(f"    LLM:        {llm['name']}")
    print(f"    Mode:       {['', 'Retrieval Eval', 'RAGAS Eval', 'API Server'][mode]}")
    print("─" * 64 + "\n")

    if not ask_yes_no("Proceed with this configuration?"):
        print("\n  Aborted. Run again to reconfigure.\n")
        sys.exit(0)

    print()

    if mode == 1:
        run_retrieval_eval(embedding, reranker)
    elif mode == 2:
        run_ragas_eval(embedding, reranker, llm)
    elif mode == 3:
        run_api_server(embedding, reranker, llm)


if __name__ == "__main__":
    main()
