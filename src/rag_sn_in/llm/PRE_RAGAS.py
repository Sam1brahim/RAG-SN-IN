import os
import contextlib
from dotenv import load_dotenv
from rag_sn_in.database.client import get_client as get_qdrant_client
from rag_sn_in.database.db_setup import ensure_ToCreate_collection
from langchain_deepseek import ChatDeepSeek
from rag_sn_in.llm.generator import load_generation_llm, clear_vram, verify_llm_name, OLLAMA_MODELS
from rag_sn_in.config import LLM_NAME, VECTOR_SIZE, EMBEDDING_MODEL_NAME, RERANKER_NAME, EVALUATOR
from rag_sn_in.eval.evaluate_RAG import evaluate_retrieval, load_eval_set
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
from rag_sn_in.observability.client import is_enabled, get_client as get_langfuse_client

class DeepSeekReasoningTracer(BaseCallbackHandler):
    """Captures raw LLM-as-a-judge reasoning for a few samples."""
    def __init__(self, limit=10):
        self.limit = limit
        self.captured = []
        self.count = 0

    def on_llm_end(self, response, **kwargs):
        if self.count < self.limit:
            for generation in response.generations:
                for chunk in generation:
                    self.captured.append({
                        "prompt": "Captured via RAGAS step", # Prompts are harder to match to items
                        "response": chunk.text
                    })
            self.count += 1

from ragas import evaluate
from ragas.cost import get_token_usage_for_openai
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

#find  .env file and load DEEPSEEK_API_KEY
load_dotenv()

#***************************************************************************************************************************************************************************************************
#***************************************************************************************************************************************************************************************************
#***************************************************************************************************************************************************************************************************
def main():
    pipeline_start = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 0. PRE-CHECK: Verify LLM configuration before starting heavy work
    try:
        model_id = verify_llm_name(LLM_NAME)
        print(f"CHECK: LLM_NAME '{LLM_NAME}' is valid. Will use: {model_id}")
        print("Pre-check PASSED. Proceeding to pipeline...\n")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        print("Pipeline aborted to save time/resources.")
        return

    # Check if using Ollama model
    is_ollama = LLM_NAME.lower() in ["gemma4-e2b", "gemma4-e4b"]
    if is_ollama:
        print(f"INFO: Using Ollama for {LLM_NAME} - bypassing HuggingFace transformers")
        print("INFO: Make sure 'ollama serve' is running on localhost:11434\n")

    # Dynamic tagging for filenames to prevent overwrites when switching models
    gen_model_tag = LLM_NAME.split("/")[-1].replace("-it", "").replace("-Instruct", "").lower()
    judge_model_tag = EVALUATOR.split("/")[-1].replace("-chat", "").lower()
    run_tag = f"{gen_model_tag}_vs_{judge_model_tag}"
    out_dir = r"E:\Project RAG-SN-IN\data\ragas"
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device USED:", device)
    collection_name = "railway"
    client = get_qdrant_client()
    pre_ragas_data = []

    # Single pipeline path: when Langfuse is disabled, observe() degrades to a
    # no-op context manager and every .update() is skipped (obs is None).
    langfuse = get_langfuse_client() if is_enabled() else None
    if langfuse is not None:
        print("Langfuse For Observability is Enabled.\n")
    else:
        print("Langfuse Tracing Disabled\n")

    def observe(as_type, name, **kwargs):
        if langfuse is None:
            return contextlib.nullcontext(None)
        return langfuse.start_as_current_observation(as_type=as_type, name=name, **kwargs)

    with observe("span", "Start-Pipeline") as root_span:

        # ---------- Retrieval & reranking ----------
        print("Launching RAGAS Pipeline:\n")
        print("Performing Reranking ...\n")
        with observe(
            "retriever",
            "Call-Qdrant",
            input={"collection": collection_name, "retrieve_k": 30, "use_reranker": True},
            metadata={"reranker": RERANKER_NAME, "embedding_model": EMBEDDING_MODEL_NAME},
        ) as retriever:
            ensure_ToCreate_collection(client, collection_name, vector_size=VECTOR_SIZE)
            eval_set = load_eval_set(r"E:\Project RAG-SN-IN\data\eval\eval chunks 512 max")
            retrieval_metrics, _, ragas_data, retrieval_stats = evaluate_retrieval(
                client, collection_name, eval_set, retrieve_k=30, use_reranker=True
            )
            retrieval_stage_time = retrieval_stats["batch_embed_time_s"] + retrieval_stats["wall_time_s"]
            if retriever is not None:
                retriever.update(output={
                    "num_items": len(ragas_data),
                    "questions": [it["question"] for it in ragas_data],
                })
        print("Finished Reranking ...\n")

        # ---------- Free GPU before loading the generator ----------
        # gc.collect() alone can't free them: embedding.py and reranker.py keep
        # their models in module-level globals that are still referenced.
        print("Clearing GPU memory...")
        from rag_sn_in.llm.reranker import unload_reranker
        from rag_sn_in.llm.embedding import unload_embedder
        unload_reranker()
        unload_embedder()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---------- Local generation ----------
        print("loading Local LLM ...\n")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()  # baseline for generation VRAM peak

        # Load LLM (either via transformers or Ollama)
        tokenizer, model = load_generation_llm(LLM_NAME)
        print(f"\n {LLM_NAME} Loaded Successfully\n")

        MAX_EVAL_ITEMS = 20
        if MAX_EVAL_ITEMS is not None and len(ragas_data) > MAX_EVAL_ITEMS:
            ragas_data = random.Random(42).sample(ragas_data, MAX_EVAL_ITEMS)
            print(f"Subsampled to {len(ragas_data)} items (seed=42)\n")

        print("Starting LLM generation ...\n")
        total_items = len(ragas_data)
        gen_start = time.time()
        gen_times = []    # per-item wall time (s)
        gen_tokens = []   # per-item generated token count

        if is_ollama:
            import ollama

        with observe(
            "chain",
            "Local-Generation",
            metadata={"generator_llm": LLM_NAME, "num_items": total_items},
        ) as gen_stage:
            for i, item in enumerate(ragas_data):
                iteration_start_time = time.time()

                print(f"--- Processing Item {i + 1}/{total_items} ---")
                print(f"Question: {item['question']}")

                # 1. Combine the contexts into a single string for the LLM
                context_str = "\n\n".join(item["contexts"])

                # 2. Build the prompt with the model's CHAT TEMPLATE.
                # Gemma 4 and Qwen 2.5 both support the 'system' role natively.
                # This provides better instruction adherence and grounding.
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Tu es un assistant expert en réglementation ferroviaire française. "
                            "Réponds à la question en français en utilisant UNIQUEMENT le contexte fourni. "
                            "Sois précis, factuel et concis. Si la réponse n'est pas dans le contexte, "
                            "indique que tu ne sais pas."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Contexte :\n{context_str}\n\n"
                            f"Question : {item['question']}"
                        ),
                    }
                ]

                # One generation observation per item: update() overwrites, so a
                # single shared observation would only keep the last item's answer.
                with observe(
                    "generation",
                    f"answer-{i + 1}",
                    model=LLM_NAME,
                    input=messages,
                ) as gen_obs:
                    if is_ollama:
                        # Ollama reports real token counts in the response metadata
                        response = ollama.chat(
                            model=OLLAMA_MODELS[LLM_NAME.lower()],
                            messages=messages,
                            stream=False
                        )
                        generated_answer = response['message']['content']
                        new_token_count = getattr(response, 'eval_count', None)
                        prompt_tokens = getattr(response, 'prompt_eval_count', None)
                        if new_token_count is None:
                            new_token_count = len(generated_answer.split())  # Rough fallback
                        if prompt_tokens is None:
                            prompt_tokens = sum(len(m["content"].split()) for m in messages)  # Rough fallback
                    else:
                        # Use HuggingFace transformers
                        prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = tokenizer(prompt, return_tensors="pt").to(device)

                        with torch.no_grad():
                            output_tokens = model.generate(
                                **inputs,
                                max_new_tokens=200,
                                do_sample=False,  # greedy: deterministic + reproducible eval
                            )

                        # Decode only the generated part (slice off the input prompt length)
                        input_length = inputs.input_ids.shape[1]
                        new_token_count = output_tokens.shape[1] - input_length
                        generated_answer = tokenizer.decode(output_tokens[0][input_length:], skip_special_tokens=True)
                        prompt_tokens = input_length

                    if gen_obs is not None:
                        gen_obs.update(
                            output=generated_answer,
                            usage_details={
                                "input": prompt_tokens,
                                "output": new_token_count,
                                "total": prompt_tokens + new_token_count,
                            },
                        )

                gen_tokens.append(new_token_count)

                # 4. Build the final dictionary
                pre_ragas_data.append({
                    "user_input": item["question"],
                    "retrieved_contexts": item["contexts"],
                    "reference": item["ground_truth"],
                    "response": generated_answer
                })

                # Calculate and print timing/status
                iteration_time = time.time() - iteration_start_time
                gen_times.append(iteration_time)

                # Print the first 75 characters of the answer so you can verify it's not generating gibberish
                preview = generated_answer.replace('\n', ' ')[:75]
                print(f"Answer Preview: {preview}...")
                print(f"Time taken: {iteration_time:.2f} seconds ({new_token_count} tokens)\n")

            # ---- Generation stage stats (runtime monitoring + resource usage) ----
            gen_wall_time = time.time() - gen_start
            peak_vram_gb = (
                round(torch.cuda.max_memory_allocated() / 1024**3, 2)
                if torch.cuda.is_available() else None
            )
            total_new_tokens = sum(gen_tokens)
            generation_stats = {
                "num_items": total_items,
                "wall_time_s": round(gen_wall_time, 1),
                "mean_latency_s": round(sum(gen_times) / len(gen_times), 2),
                "min_latency_s": round(min(gen_times), 2),
                "max_latency_s": round(max(gen_times), 2),
                "total_generated_tokens": total_new_tokens,
                "avg_tokens_per_answer": round(total_new_tokens / total_items, 1),
                "tokens_per_sec": round(total_new_tokens / gen_wall_time, 1),
                "items_per_hour": round(total_items / gen_wall_time * 3600, 1),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "peak_vram_gb": peak_vram_gb,
            }
            if gen_stage is not None:
                gen_stage.update(output=generation_stats)

        print("\n----- Generation stage stats -----")
        for k, v in generation_stats.items():
            print(f"  {k}: {v}")

        pre_ragas_path = os.path.join(out_dir, f"{run_tag}_pre_ragas_{timestamp}.json")
        with open(pre_ragas_path, "w", encoding="utf-8") as f:
            json.dump(pre_ragas_data, f, ensure_ascii=False, indent=2)
        print(f"Intermediate generation results saved to: {pre_ragas_path}")
        final_ragas_dataset = Dataset.from_list(pre_ragas_data)

        print("Clearing LOCAL LLM from GPU")
        clear_vram(model, tokenizer)
        print("Cleaned ...")
        ragas_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name="google/embeddinggemma-300m",
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        )

        # ---------- LLM-as-a-judge ----------
        print("Preparing LLM-As-A-Judge. . .")

        # NEW: Tracer to see what DeepSeek is thinking
        reasoning_tracer = DeepSeekReasoningTracer(limit=40) # 40 calls ~ approx 10 full items

        judge_llm = LangchainLLMWrapper(ChatDeepSeek(
            model=EVALUATOR,
            api_key=os.environ["DEEPSEEK_API_KEY"],
            max_tokens=4096,
            callbacks=[reasoning_tracer] # Wire it here
        ))

        metric_cols = ["faithfulness", "answer_relevancy",
                       "context_precision", "context_recall"]

        print("API Key loaded and model is ready!\n")
        print("Running LLM-as-a-judge evaluation...")
        judge_start = time.time()
        with observe(
            "evaluator",
            "LLM-As-A-Judge",
            input={"num_items": len(pre_ragas_data), "metrics": metric_cols},
            metadata={"judge_model": EVALUATOR, "embedding_model": "google/embeddinggemma-300m"},
        ) as evaluator:
            results = evaluate(
                dataset=final_ragas_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=judge_llm,
                embeddings=ragas_embeddings,
                # DeepSeek's API returns OpenAI-format usage, so the OpenAI parser
                # can read token counts straight out of the judge's responses.
                token_usage_parser=get_token_usage_for_openai,
            )
            df = results.to_pandas()
            metric_cols = [c for c in metric_cols if c in df.columns]
            scores_df = df[metric_cols]

            if evaluator is not None:
                evaluator.update(
                    output={
                        "aggregate_scores": {m: float(df[m].mean()) for m in metric_cols},
                        "nan_counts": scores_df.isna().sum().to_dict(),  # failed judge parses!
                        "per_sample_scores": scores_df.where(scores_df.notna(), None)
                                                    .to_dict(orient="records"),
                    },
                )
        judge_wall_time = time.time() - judge_start

        # ---------------- Aggregate scores ----------------
        print("Evaluation Results (aggregate):")
        print(results)

        # ---------------- Judge API token usage ----------------
        token_usage = results.total_tokens()
        n_items = len(pre_ragas_data)
        n_judge_calls = n_items * 4  # 4 metrics per item (approx; faithfulness may add steps)
        judge_token_stats = {
            "wall_time_s": round(judge_wall_time, 1),
            "judge_calls_approx": n_judge_calls,
            "calls_per_sec": round(n_judge_calls / judge_wall_time, 2),
            "input_tokens": token_usage.input_tokens,
            "output_tokens": token_usage.output_tokens,
            "total_tokens": token_usage.input_tokens + token_usage.output_tokens,
        }
        print("\n----- Judge (DeepSeek) token usage -----")
        print(f"Wall time:      {judge_token_stats['wall_time_s']:.0f}s "
              f"({judge_token_stats['judge_calls_approx']} calls, "
              f"{judge_token_stats['calls_per_sec']}/s)")
        print(f"Input tokens:   {judge_token_stats['input_tokens']:,}")
        print(f"Output tokens:  {judge_token_stats['output_tokens']:,}")
        print(f"Total tokens:   {judge_token_stats['total_tokens']:,}")

        # ---------------- Per-item details ----------------
        # to_pandas() gives one row per question: user_input, response,
        # reference, retrieved_contexts + one column per metric score.

        # 1. Full per-item scores -> CSV (easy to open/filter in Excel)
        csv_path = os.path.join(out_dir, f"ragas_results_{run_tag}_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")  # utf-8-sig: French accents open correctly in Excel
        print(f"\nPer-item scores saved to: {csv_path}")

        # 2. Full per-item scores -> JSON (keeps lists/nesting intact)
        json_path = os.path.join(out_dir, f"ragas_results_{run_tag}_{timestamp}.json")
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        print(f"Per-item JSON saved to:  {json_path}")

        # 3. Summary JSON: aggregates + per-metric stats + run metadata
        summary = {
            "timestamp": timestamp,
            "run_id": run_tag,
            "num_items": len(df),
            "config": {
                "generator_llm": LLM_NAME,
                "judge_llm": EVALUATOR,
                "judge_embeddings": "google/embeddinggemma-300m",
                "retrieval_embedding_model": EMBEDDING_MODEL_NAME,
                "reranker": RERANKER_NAME,
                "collection": collection_name,
            },
            "aggregate_scores": {c: float(df[c].mean()) for c in metric_cols},
            "runtime": {
                "total_pipeline_s": round(time.time() - pipeline_start, 1),
                "stages": {
                    "retrieval_rerank_s": round(retrieval_stage_time, 1),
                    "local_generation_s": round(gen_wall_time, 1),
                    "llm_judge_s": round(judge_wall_time, 1),
                },
            },
            "resource_usage": {
                "peak_vram_gb": generation_stats.get("peak_vram_gb"),
                "gpu_name": generation_stats.get("gpu_name"),
            },
            "retrieval_metrics": retrieval_metrics,
            "retrieval_performance": retrieval_stats,
            "generation_performance": generation_stats,
            "judge_token_usage": judge_token_stats,
            "per_metric_stats": {
                c: {
                    "mean": float(df[c].mean()),
                    "std": float(df[c].std()),
                    "min": float(df[c].min()),
                    "max": float(df[c].max()),
                    "pct_below_0.5": float((df[c] < 0.5).mean()),
                }
                for c in metric_cols
            },
        }
        summary_path = os.path.join(out_dir, f"ragas_summary_{run_tag}_{timestamp}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Summary saved to:        {summary_path}")

        # 4. Save raw DeepSeek reasoning samples
        reasoning_path = os.path.join(out_dir, f"judge_reasoning_samples_{run_tag}_{timestamp}.txt")
        with open(reasoning_path, "w", encoding="utf-8") as f:
            f.write(f"SAMPLES OF DEEPSEEK REASONING (LIMIT {reasoning_tracer.limit} CALLS)\n")
            f.write("="*60 + "\n\n")
            for i, entry in enumerate(reasoning_tracer.captured):
                f.write(f"--- CALL {i+1} ---\n")
                f.write(f"{entry['response']}\n\n")
        print(f"Reasoning samples saved: {reasoning_path}")

        # 5. Console: per-metric stats
        print("\n----- Per-metric statistics -----")
        for c in metric_cols:
            s = summary["per_metric_stats"][c]
            print(f"{c:20s} mean={s['mean']:.3f}  std={s['std']:.3f}  "
                  f"min={s['min']:.3f}  max={s['max']:.3f}  "
                  f"below_0.5={s['pct_below_0.5']*100:.1f}%")

        # 6. Console: worst 5 items by faithfulness (hallucination hunting)
        if "faithfulness" in df.columns:
            print("\n----- 5 lowest-faithfulness answers (inspect these) -----")
            worst = df.nsmallest(5, "faithfulness")
            for rank, (_, row) in enumerate(worst.iterrows(), start=1):
                q = str(row["user_input"]).replace("\n", " ")[:90]
                a = str(row["response"]).replace("\n", " ")[:90]
                print(f"[{rank}] faithfulness={row['faithfulness']:.3f}")
                print(f"    Q: {q}")
                print(f"    A: {a}...")

        if root_span is not None:
            root_span.update(output={
                "run_id": run_tag,
                "aggregate_scores": summary["aggregate_scores"],
                "runtime_s": summary["runtime"],
            })

    # Push remaining batched spans before process exit
    if langfuse is not None:
        langfuse.flush()

if __name__== '__main__':
    main()
