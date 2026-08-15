from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path(r"E:\Project RAG-SN-IN\data\processed\text\drr-2027")
MODEL_NAME = "google/embeddinggemma-300m"

# EmbeddingGemma documented maximum input length
MODEL_MAX_TOKENS = 2048

# Conservative operational limit
SAFE_LIMIT = 1800

# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

# --------------------------------------------------
# Count tokens in one file
# --------------------------------------------------

def count_tokens(text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )

    return len(encoded["input_ids"])


# --------------------------------------------------
# Scan files
# --------------------------------------------------

records = []

files = sorted(
    path for path in DATA_DIR.rglob("*.md")
    if path.name.lower().startswith("page")
)

if not files:
    raise FileNotFoundError(
        f"No Markdown files beginning with 'page' found in:\n{DATA_DIR}"
    )

print(f"Found {len(files)} files.")

for file_path in tqdm(
    files,
    desc="Counting tokens",
    unit="file"
):
    try:
        text = file_path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        token_count = count_tokens(text)

        records.append({
            "file": file_path.name,
            "path": str(file_path),
            "characters": len(text),
            "words_approx": len(text.split()),
            "tokens": token_count,
            "exceeds_2048": token_count > MODEL_MAX_TOKENS,
            "exceeds_safe_limit": token_count > SAFE_LIMIT,
        })

    except Exception as error:
        print(f"\nCould not process {file_path}: {error}")


# --------------------------------------------------
# Create dataframe
# --------------------------------------------------

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("No files were successfully processed.")

tokens = df["tokens"]

# --------------------------------------------------
# Statistics
# --------------------------------------------------

percentiles = {
    "minimum": tokens.min(),
    "1%": np.percentile(tokens, 1),
    "5%": np.percentile(tokens, 5),
    "10%": np.percentile(tokens, 10),
    "25%": np.percentile(tokens, 25),
    "median": np.median(tokens),
    "75%": np.percentile(tokens, 75),
    "90%": np.percentile(tokens, 90),
    "95%": np.percentile(tokens, 95),
    "99%": np.percentile(tokens, 99),
    "maximum": tokens.max(),
    "mean": tokens.mean(),
    "standard_deviation": tokens.std(),
}

print("\n" + "=" * 60)
print("TOKEN STATISTICS")
print("=" * 60)

for name, value in percentiles.items():
    print(f"{name:20}: {value:,.2f}")

# --------------------------------------------------
# Threshold analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("THRESHOLD ANALYSIS")
print("=" * 60)

thresholds = [256, 384, 512, 768, 1024, 1280, 1536, 1800, 2048]

for threshold in thresholds:
    count = (tokens <= threshold).sum()
    percentage = count / len(tokens) * 100

    print(
        f"<= {threshold:4} tokens: "
        f"{count:5} files "
        f"({percentage:6.2f}%)"
    )

# --------------------------------------------------
# Suggested logical limits
# --------------------------------------------------

p90 = np.percentile(tokens, 90)
p95 = np.percentile(tokens, 95)

print("\n" + "=" * 60)
print("POSSIBLE CHUNKING INTERPRETATION")
print("=" * 60)

print(f"90th percentile: {p90:.0f} tokens")
print(f"95th percentile: {p95:.0f} tokens")

if p95 <= 512:
    recommendation = 512
elif p95 <= 768:
    recommendation = 768
elif p95 <= 1024:
    recommendation = 1024
elif p95 <= 1280:
    recommendation = 1280
else:
    recommendation = 1536

print(f"\nSuggested initial maximum: {recommendation} tokens")

if recommendation >= MODEL_MAX_TOKENS:
    print("Warning: recommendation reaches the model maximum.")
else:
    print(
        f"This leaves approximately "
        f"{MODEL_MAX_TOKENS - recommendation} tokens of safety margin."
    )

# --------------------------------------------------
# Largest files
# --------------------------------------------------

print("\n" + "=" * 60)
print("10 LARGEST FILES")
print("=" * 60)

print(
    df.sort_values("tokens", ascending=False)
      [["file", "tokens", "characters", "exceeds_2048"]]
      .head(10)
      .to_string(index=False)
)

# --------------------------------------------------
# Save results
# --------------------------------------------------

output_file = DATA_DIR / "page_token_statistics.csv"
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"\nDetailed results saved to:\n{output_file}")