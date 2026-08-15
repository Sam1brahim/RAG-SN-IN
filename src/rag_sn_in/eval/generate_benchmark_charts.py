import os
import glob
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_charts():
    metrics_dir = r"E:\Project RAG-SN-IN\data\Eval RAG metrics"
    files = sorted(glob.glob(os.path.join(metrics_dir, "*.json")))
    
    # Filter only retrieval_eval_*.json files
    eval_files = [f for f in files if os.path.basename(f).startswith("retrieval_eval_")]
    
    data = []
    for f in eval_files:
        with open(f, "r", encoding="utf-8") as fp:
            d = json.load(fp)
            emb = d["config"]["embedding_model"].split("/")[-1]
            rerank = d["config"]["reranker"].split("/")[-1] if d["config"]["reranker"] else "Dense Only"
            m = d["retrieval_metrics"]
            p = d["retrieval_performance"]
            data.append({
                "emb": emb,
                "rerank": rerank,
                "hit1": m["1"]["hit_rate"] * 100,
                "hit3": m["3"]["hit_rate"] * 100,
                "hit5": m["5"]["hit_rate"] * 100,
                "hit10": m["10"]["hit_rate"] * 100,
                "mrr": m["10"]["mrr"],
                "misses": p["miss_count"],
                "dense_ms": p["avg_dense_latency_ms"],
                "rerank_ms": p["avg_rerank_latency_ms"],
                "total_ms": p["avg_total_latency_ms"],
                "qps": p["throughput_queries_per_s"]
            })
            
    # Set plot styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("RAG Retrieval Ablation Benchmark: French Railway Regulatory Corpus (187 Queries)", fontsize=16, fontweight='bold', y=0.98)

    # -------------------------------------------------------------
    # 1. Hit@1 and MRR Comparison (Grouped Bar Chart)
    # -------------------------------------------------------------
    models = ["embeddinggemma-300m", "multilingual-e5-large", "bge-m3", "nomic-embed-text-v1.5"]
    rerankers = ["None (Dense Only)", "bge-reranker-v2-m3", "Qwen3-Reranker-0.6B"]
    
    x = np.arange(len(models))
    width = 0.26
    
    colors = {
        "None (Dense Only)": "#F59E0B",
        "bge-reranker-v2-m3": "#10B981",
        "Qwen3-Reranker-0.6B": "#3B82F6"
    }

    for i, rk in enumerate(["None (Dense Only)", "bge-reranker-v2-m3", "Qwen3-Reranker-0.6B"]):
        vals = []
        for m in models:
            match = next((d for d in data if d["emb"] == m and (d["rerank"] == rk or (rk.startswith("None") and d["rerank"] == "Dense Only"))), None)
            vals.append(match["hit1"] if match else 0)
        
        label_name = "Dense Only" if rk.startswith("None") else rk
        rects = ax1.bar(x + (i - 1) * width, vals, width, label=label_name, color=colors[label_name if label_name in colors else "None (Dense Only)"], alpha=0.9)
        ax1.bar_label(rects, padding=3, fmt='%.1f%%', fontsize=9, fontweight='bold')

    ax1.set_title("Hit@1 Accuracy by Embedding & Reranker", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Gemma 300M", "Multilingual E5", "BGE-M3", "Nomic v1.5"], fontsize=11, fontweight='semibold')
    ax1.set_ylabel("Hit@1 Rate (%)", fontsize=11)
    ax1.set_ylim(40, 95)
    ax1.legend(loc='upper right', frameon=True)

    # -------------------------------------------------------------
    # 2. MRR@10 Comparison
    # -------------------------------------------------------------
    for i, rk in enumerate(["None (Dense Only)", "bge-reranker-v2-m3", "Qwen3-Reranker-0.6B"]):
        vals = []
        for m in models:
            match = next((d for d in data if d["emb"] == m and (d["rerank"] == rk or (rk.startswith("None") and d["rerank"] == "Dense Only"))), None)
            vals.append(match["mrr"] if match else 0)
        
        label_name = "Dense Only" if rk.startswith("None") else rk
        rects = ax2.bar(x + (i - 1) * width, vals, width, label=label_name, color=colors[label_name if label_name in colors else "None (Dense Only)"], alpha=0.9)
        ax2.bar_label(rects, padding=3, fmt='%.3f', fontsize=9, fontweight='bold')

    ax2.set_title("Mean Reciprocal Rank (MRR@10)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Gemma 300M", "Multilingual E5", "BGE-M3", "Nomic v1.5"], fontsize=11, fontweight='semibold')
    ax2.set_ylabel("MRR Score (0 - 1.0)", fontsize=11)
    ax2.set_ylim(0.5, 0.98)
    ax2.legend(loc='upper right', frameon=True)

    # -------------------------------------------------------------
    # 3. Hit@k Curves for Top Configurations
    # -------------------------------------------------------------
    k_vals = [1, 3, 5, 10]
    
    top_runs = [
        ("embeddinggemma-300m", "bge-reranker-v2-m3", "#10B981", "o-", "Gemma 300M + BGE v2-m3 (Best)"),
        ("multilingual-e5-large", "bge-reranker-v2-m3", "#059669", "s--", "E5 Large + BGE v2-m3"),
        ("embeddinggemma-300m", "Dense Only", "#F59E0B", "^-.", "Gemma 300M (Dense Only)"),
        ("bge-m3", "bge-reranker-v2-m3", "#3B82F6", "d-", "BGE-M3 + BGE v2-m3"),
        ("nomic-embed-text-v1.5", "Dense Only", "#EF4444", "x:", "Nomic v1.5 (Dense Only)"),
    ]

    for emb, rk, color, style, label in top_runs:
        match = next((d for d in data if d["emb"] == emb and d["rerank"] == rk), None)
        if match:
            y = [match["hit1"], match["hit3"], match["hit5"], match["hit10"]]
            ax3.plot(k_vals, y, style, label=label, color=color, linewidth=2.2, markersize=7)

    ax3.set_title("Recall Curve Across k (Hit@k)", fontsize=13, fontweight='bold', pad=12)
    ax3.set_xlabel("Top-k Cutoff", fontsize=11)
    ax3.set_ylabel("Hit Rate (%)", fontsize=11)
    ax3.set_xticks(k_vals)
    ax3.set_ylim(50, 100)
    ax3.legend(loc='lower right', frameon=True)

    # -------------------------------------------------------------
    # 4. Latency vs Accuracy Frontier (Scatter Plot)
    # -------------------------------------------------------------
    for d in data:
        rk = d["rerank"]
        col = colors.get(rk, "#6B7280")
        size = 130 if "bge-reranker" in rk else (90 if "Dense" in rk else 110)
        ax4.scatter(d["total_ms"], d["hit1"], s=size, color=col, alpha=0.85, edgecolors='black', linewidth=1)
        
        # Add annotation for key points
        short_name = d["emb"].replace("embedding", "").replace("-embed-text", "").replace("multilingual-", "")
        if "bge-reranker" in rk or "Dense" in rk:
            ax4.annotate(f"{short_name}\n({rk[:8]})", (d["total_ms"], d["hit1"]), textcoords="offset points", xytext=(8, -4), fontsize=8, fontweight='medium')

    ax4.set_xscale('log')
    ax4.set_title("Latency vs Accuracy Frontier (Log Scale)", fontsize=13, fontweight='bold', pad=12)
    ax4.set_xlabel("Average Query Latency (ms, Log Scale)", fontsize=11)
    ax4.set_ylabel("Hit@1 Rate (%)", fontsize=11)
    ax4.set_ylim(50, 90)
    
    # Custom Legend for Scatter
    for rk_name, rk_col in colors.items():
        ax4.scatter([], [], color=rk_col, label=rk_name, s=80, edgecolors='black')
    ax4.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    out_path = os.path.join(metrics_dir, "retrieval_benchmark_ablation_matrix.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Benchmark chart saved successfully to: {out_path}")

if __name__ == "__main__":
    generate_charts()
