import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def get_reranker():
    global _tokenizer, _model
    if _model is None:
        print("Loading reranker model: BAAI/bge-reranker-v2-m3 ...")
        t0 = time.time()
        _tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3', use_fast=True)
        _model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3',dtype=torch.float16)
        _model.to(_device)
        _model.eval()
        print(f"Reranker loaded in {time.time() - t0:.1f}s on {_device}\n")
    return _tokenizer, _model

def rerank(query, candidates, top_k=10):
    tokenizer, model = get_reranker()
    pairs = [[query, c.payload["text"]] for c in candidates]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(_device)

        scores = model(**inputs, return_dict=True).logits.view(-1).float()
        scores = torch.sigmoid(scores).cpu().tolist()  # normalize to [0,1]

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [c for c, s in scored[:top_k]]