import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import snapshot_download

# Map friendly names to actual Hugging Face repo IDs
SUPPORTED_MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "gemma4-e4b": "google/gemma-4-E4B-it",
    "gemma4-e2b": "google/gemma-4-E2B-it"
}

def verify_llm_name(model_name: str) -> str:
    """Verifies support and ensures the model weights/tokenizer are cached locally."""
    model_id = SUPPORTED_MODELS.get(model_name.lower())
    if not model_id:
        raise ValueError(f"Unsupported LLM_NAME: '{model_name}'. Supported models: {list(SUPPORTED_MODELS.keys())}")
    
    try:
        print(f"Verifying/Downloading {model_id} (checking local cache first)...")
        # Ensure we download the actual weights (safetensors) specifically
        snapshot_download(
            repo_id=model_id, 
            local_files_only=False,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.py"] # Force weights
        )
        # Verify tokenizer specifically as it's needed for prompt formatting
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        raise ValueError(f"Failed to ensure model '{model_id}' is available: {e}")
        
    return model_id

def load_generation_llm(model_name: str):
    """
    Loads selected model in 4-bit to fit in 8GB VRAM.
    """
    import traceback
    import sys
    model_id = verify_llm_name(model_name)
    
    # Pre-load cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        info = torch.cuda.mem_get_info()
        print(f"DEBUG: GPU memory info: Free={info[0]/1024**3:.2f}GB, Total={info[1]/1024**3:.2f}GB", flush=True)

    print(f"DEBUG: Starting tokenizer load for {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("DEBUG: Tokenizer loaded successfully.", flush=True)

    # Configure 4-bit quantization
    print("DEBUG: Configuring BitsAndBytes...", flush=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    try:
        print(f"DEBUG: Calling AutoModelForCausalLM.from_pretrained for {model_id}...", flush=True)
        
        # Calculate a safe memory limit (leave some room for KV Cache and system)
        # We'll use 90% of your current free memory
        max_mem = f"{int(info[0] * 0.9 / 1024**2)}MB"
        print(f"DEBUG: Setting max_memory for GPU 0 to {max_mem}", flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map={"": 0}, # FORCE entire model to GPU 0 (skips accelerate's cautious offloading)
            max_memory={0: max_mem},
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("DEBUG: Model instantiation complete!", flush=True)
        return tokenizer, model
    except BaseException as e:
        print("\n" + "!"*60, flush=True)
        print(f"FATAL ERROR during {model_id} instantiation:", flush=True)
        print(f"Error Type: {type(e).__name__}", flush=True)
        print(f"Error Message: {str(e)}", flush=True)
        traceback.print_exc(file=sys.stdout)
        print("!"*60 + "\n", flush=True)
        sys.stdout.flush()
        raise e

def clear_vram(model, tokenizer):
    """
    Crucial for 8GB GPUs: Call this when switching between embedding/reranker/LLM
    """
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM cleared.")