import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import snapshot_download
from rag_sn_in.config import LLM_NAME

# Map friendly names to actual Hugging Face repo IDs
SUPPORTED_MODELS = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-2-9b-it",
    "gemma2-2b": "google/gemma-2-2b-it",
}

# Ollama models (for local inference via Ollama server)
# Key: friendly name used in config.py, Value: actual Ollama model name
OLLAMA_MODELS = {
    "gemma4-e2b": "gemma4:e2b",  # Maps to Ollama's gemma4:e2b model
}

def verify_llm_name(model_name: str) -> str:
    """
    Verifies support and ensures the model weights/tokenizer are cached locally.
    For Ollama models, this is a no-op (returns the model name as-is).
    """
    # Check if this is an Ollama model - skip HuggingFace verification entirely
    if model_name.lower() in OLLAMA_MODELS:
        ollama_model = OLLAMA_MODELS[model_name.lower()]
        print(f"Ollama model detected: {ollama_model}")
        print(f"Skipping HuggingFace verification. Make sure 'ollama serve' is running.")
        return model_name.lower()
    
    # HuggingFace model verification
    model_id = SUPPORTED_MODELS.get(model_name.lower()) or model_name
    
    try:
        print(f"Verifying/Downloading {model_id} (checking local cache first)...")
        snapshot_download(
            repo_id=model_id, 
            local_files_only=False,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.py", "tokenizer*"]
        )
        print(f"Loading tokenizer for {model_id}...")
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to ensure model '{model_id}' is available: {e}")
        
    return model_id

def load_generation_llm(model_name: str):
    """
    Loads selected model in 4-bit to fit in 8GB VRAM.
    If using Ollama, this will connect to the Ollama server instead.
    """
    import traceback
    import sys
    
    # Check if we should use Ollama
    if model_name.lower() in OLLAMA_MODELS:
        return load_ollama_llm(model_name)
    
    model_id = verify_llm_name(model_name)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        info = torch.cuda.mem_get_info()
        print(f"DEBUG: GPU memory info: Free={info[0]/1024**3:.2f}GB, Total={info[1]/1024**3:.2f}GB", flush=True)

    print(f"DEBUG: Starting tokenizer load for {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("DEBUG: Tokenizer loaded successfully.", flush=True)

    print("DEBUG: Configuring BitsAndBytes...", flush=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    try:
        print(f"DEBUG: Calling AutoModelForCausalLM.from_pretrained for {model_id}...", flush=True)
        max_mem = f"{int(info[0] * 0.9 / 1024**2)}MB"
        print(f"DEBUG: Setting max_memory for GPU 0 to {max_mem}", flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map={"": 0},
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

def load_ollama_llm(model_name: str):
    """
    Loads Gemma 4 via Ollama (bypasses transformers metadata issues entirely).
    Requires: ollama serve (running on localhost:11434)
    """
    try:
        import ollama
    except ImportError:
        raise ImportError("ollama package not installed. Run: pip install ollama")
    
    ollama_model = OLLAMA_MODELS.get(model_name.lower())
    print(f"DEBUG: Loading Ollama model: {ollama_model}")
    print("DEBUG: Make sure 'ollama serve' is running on localhost:11434")
    
    # Return a dummy tokenizer and model that use Ollama's API
    return OllamaTokenizerWrapper(ollama_model), OllamaModelWrapper(ollama_model)


class OllamaTokenizerWrapper:
    """Minimal tokenizer wrapper for Ollama (doesn't need actual tokenization)."""
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Ollama handles chat templating internally
        # We just need to format the messages for the API call
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        if add_generation_prompt:
            formatted += "Assistant: "
        return formatted
    
    def __call__(self, text, **kwargs):
        # Dummy call for compatibility
        return type('obj', (object,), {'input_ids': torch.tensor([[0]])})


class OllamaModelWrapper:
    """Minimal model wrapper that calls Ollama API."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "ollama"  # Dummy device indicator
    
    def generate(self, **kwargs):
        """
        Dummy generate method - actual inference happens via Ollama API in PRE_RAGAS.
        This is just for compatibility with the existing code structure.
        """
        raise NotImplementedError(
            "Ollama models should be called via the Ollama API directly. "
            "Use ollama.generate() or ollama.chat() instead."
        )
    
    def to(self, device):
        """Dummy device placement for compatibility."""
        return self


def clear_vram(model, tokenizer):
    """
    Crucial for 8GB GPUs: Call this when switching between embedding/reranker/LLM
    For Ollama models, this is a no-op since inference happens on the Ollama server.
    """
    if isinstance(model, OllamaModelWrapper):
        print("VRAM cleared. (Ollama model - no local VRAM used)")
        return
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM cleared.")