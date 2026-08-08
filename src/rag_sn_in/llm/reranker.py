import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import time
from rag_sn_in.config import RERANKER_NAME

_tokenizer = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

# Globals for Qwen to be initialized later
max_length = 1024
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n\n\n"
task = 'Given a question query, retrieve relevant passages that answer the query'

# We'll set these inside get_reranker() once the tokenizer is actually loaded
token_false_id = None
token_true_id = None
prefix_tokens = None
suffix_tokens = None

def process_inputs(pairs):
    inputs = _tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=True, # <--- CHANGE THIS TO TRUE
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs['attention_mask'][i] = [1] * len(prefix_tokens) + inputs['attention_mask'][i] + [1] * len(suffix_tokens)
    
    inputs = _tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(_model.device)
    return inputs

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction, query=query, doc=doc)

def compute_logits(inputs, **kwargs):
    outputs = _model(**inputs)
    logits = outputs.logits

    # Left padding: every sequence ends at the last column, so the yes/no
    # logits are always at position -1.
    batch_scores = logits[:, -1, :]
    
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

def get_reranker():
    global _tokenizer, _model, token_false_id, token_true_id, prefix_tokens, suffix_tokens
    
    if _model is None:
        print(f"Loading reranker model: {RERANKER_NAME} ...")
        t0 = time.time()
        _tokenizer = AutoTokenizer.from_pretrained(RERANKER_NAME, use_fast=True,local_files_only=True)

        if 'qwen' in RERANKER_NAME.lower():
            _tokenizer.padding_side = "left"
            # Initialize Qwen globals now that tokenizer exists!
            token_false_id = _tokenizer.convert_tokens_to_ids("no")
            token_true_id = _tokenizer.convert_tokens_to_ids("yes")
            prefix_tokens = _tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = _tokenizer.encode(suffix, add_special_tokens=False)
            
            # Using RERANKER_NAME instead of hardcoded string to be flexible
            _model = AutoModelForCausalLM.from_pretrained(
                RERANKER_NAME,
                dtype=torch.float16 if _device == "cuda" else torch.float32,
                attn_implementation="sdpa",local_files_only=True
            ).to(_device).eval()
        else:
            _model = AutoModelForSequenceClassification.from_pretrained(RERANKER_NAME, dtype=torch.float16)
            _model.to(_device)
            _model.eval()

        print(f"Reranker loaded in {time.time() - t0:.1f}s on {_device}\n")
        
    return _tokenizer, _model, RERANKER_NAME.lower()

def rerank(query, candidates, top_k=10):
    _tokenizer, _model, reranker_name = get_reranker()
    
    if 'qwen' in reranker_name:
        # FIX: Removed the zip() to iterate candidates correctly
        pairs = [format_instruction(task, query, doc.payload["text"]) for doc in candidates]
        all_scores = []
        batch_size = 4
       
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                
                # Tokenize and compute just this small batch
                inputs = process_inputs(batch_pairs)
                batch_scores = compute_logits(inputs)
                
                # Extract the score for the "yes" token (assuming you want the positive class)
                # You might already have a compute_logits function for this
                all_scores.extend(batch_scores)

        # 3. Pair and sort
        scored = list(zip(candidates, all_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, s in scored[:top_k]]
            
    else:
        pairs = [[query, c.payload["text"]] for c in candidates]
        with torch.no_grad():
            inputs = _tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(_device)

            scores = _model(**inputs, return_dict=True).logits.view(-1).float()
            scores = torch.sigmoid(scores).cpu().tolist()  # normalize to [0,1]

            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            return [c for c, s in scored[:top_k]]