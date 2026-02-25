import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.json"

def check_accuracy():
    print("Loading index and chunks...")
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    # Define questions and expected keywords to check if the retrieved context is relevant
    evaluation_set = [
        ("What is the total revenue reported?", ["revenue", "$", "billion", "million"]),
        ("Who are the key executives mentioned?", ["ceo", "cfo", "director", "president", "executive"]),
        ("What are the major risk factors?", ["risk", "factor", "uncertainty", "forward-looking"]),
        ("When is the next earnings call?", ["call", "earnings", "quarter", "q1", "q2", "q3", "q4"]),
        ("What is the company's core business?", ["business", "platform", "app", "oil", "gas", "energy", "aviation"]),
    ]

    q_texts = [q[0] for q in evaluation_set]
    print(f"Embedding {len(q_texts)} queries to check retrieval accuracy...")
    q_embs = model.encode(q_texts, convert_to_numpy=True)
    
    # Retrieve top 5 chunks
    distances, indices = index.search(q_embs, 5)

    print("\\n" + "="*50)
    print("--- RETRIEVAL ACCURACY REPORT ---")
    print("="*50)
    
    total_score = 0
    for i, (q, keywords) in enumerate(evaluation_set):
        context_chunks = [all_chunks[idx] for idx in indices[i] if idx < len(all_chunks)]
        context_lower = " ".join(context_chunks).lower()
        
        # Check how many expected domain keywords appear in the retrieved context
        hits = [kw for kw in keywords if kw in context_lower]
        score = len(hits) / len(keywords) * 100
        total_score += score
        
        print(f"\\nQuestion: {q}")
        print(f"Expected Keywords: {keywords}")
        print(f"Keywords Found in Context: {hits} ({score:.0f}%)")
        print(f"Top Hit Snippet: '{context_chunks[0][:250]}...'")
        
    print("\\n" + "="*50)
    print(f"Average Keyword Retrieval Score: {total_score / len(evaluation_set):.0f}%")
    print("="*50)

if __name__ == "__main__":
    check_accuracy()
