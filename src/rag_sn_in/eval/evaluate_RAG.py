import json 
import os

def load_evaluation_set(evaluation_path) -> list:
    evaluation_samples = []
    print("\n Loading RAG Evaluation Dataset. . .\n")
    with open (evaluation_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            example = json.loads(line)
            evaluation_samples.append(example)

        print("Number of examples:", len(evaluation_samples))
        
    if evaluation_samples:
        required_fields = [
                "id",
                "question",
                "gold_chunk_ids",
                "gold_answer",
            ]

        for example in evaluation_samples:
                for field in required_fields:
                    if field not in example:
                        print(
                            "Missing field:",
                            field,
                            "in example:",
                            example.get("id")
                            )
    return evaluation_samples


def retrieve_chunks(question, top_k):
    pass


def evaluate_retrieval(relevant_ids, retrieved_ids):
    pass


def generate_answer(question, chunks):
    pass


def evaluate_answer(question, answer, reference_answer, chunks):
    pass


def main():
 pass

if __name__ == "__main__":
    load_evaluation_set(r"E:\Project RAG-SN-IN\data\eval\rag_eval_set.jsonl")