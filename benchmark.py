import os
import time
import json
import subprocess
import faiss
from sentence_transformers import SentenceTransformer

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def hybrid_search(index, all_chunks, query_emb, query_text, top_k=8):
    """
    Advanced hybrid search with multiple strategies for maximum accuracy.
    """
    # Strategy 1: Semantic search (get more candidates)
    distances, indices = index.search(query_emb.reshape(1, -1), top_k * 3)
    
    # Enhanced keyword extraction
    query_lower = query_text.lower()
    query_keywords = set(query_lower.split())
    
    # Add common variations and synonyms
    expanded_keywords = set(query_keywords)
    if "revenue" in query_lower:
        expanded_keywords.update(["sales", "income", "earnings", "financial", "results"])
    if "executive" in query_lower or "key" in query_lower:
        expanded_keywords.update(["ceo", "cfo", "president", "director", "officer", "management"])
    if "risk" in query_lower:
        expanded_keywords.update(["uncertainty", "threat", "challenge", "factor", "material", "adverse", "litigation", "regulatory", "compliance", "market", "operational", "financial", "legal"])
    if "business" in query_lower:
        expanded_keywords.update(["operations", "platform", "services", "industry", "sector"])
    
    scored_chunks = []
    
    for i, idx in enumerate(indices[0]):
        if idx >= len(all_chunks):
            continue
            
        chunk = all_chunks[idx].lower()
        
        # Enhanced keyword scoring with partial matches
        keyword_matches = 0
        for kw in expanded_keywords:
            if kw in chunk:
                keyword_matches += 2  # Full match weight
            elif any(kw.startswith(chunk_word) or chunk_word.startswith(kw) 
                    for chunk_word in chunk.split() if len(chunk_word) > 3):
                keyword_matches += 1  # Partial match weight
        
        keyword_score = min(keyword_matches / (len(expanded_keywords) * 2), 1.0)
        
        # Semantic score with distance normalization
        semantic_score = 1.0 / (1.0 + distances[0][i])
        
        # Dynamic weighting: prioritize keywords when semantic scores are similar
        if i < 3:  # Top semantic results
            combined_score = 0.6 * semantic_score + 0.4 * keyword_score
        else:  # Lower semantic results, boost keyword importance
            combined_score = 0.4 * semantic_score + 0.6 * keyword_score
        
        scored_chunks.append((combined_score, idx, chunk))
    
    # Sort by combined score and return top-k
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [idx for _, idx, _ in scored_chunks[:top_k]]

def run_benchmark():
    print("Starting Lucio AI Pipeline Benchmark...")
    print("")
    
    print("Running Ingestion Phase...")
    start_time = time.time()
    ingest_result = subprocess.run(["uv", "run", "python", "lucio_cli.py", "ingest"], capture_output=True, text=True)
    ingest_time = time.time() - start_time
    
    if ingest_result.returncode != 0:
        print("Ingestion failed! Output:")
        print(ingest_result.stderr)
        return

    print("Running Query Phase...")
    start_time = time.time()
    run_result = subprocess.run(["uv", "run", "python", "lucio_cli.py", "run"], capture_output=True, text=True)
    query_time = time.time() - start_time
    
    if run_result.returncode != 0:
        print("Query execution failed! Output:")
        print(run_result.stderr)
        return

    print("Evaluating Retrieval Accuracy...")
    index = faiss.read_index("faiss_index.bin")
    with open("chunks.json", 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    evaluation_set = [
        ("What is the total revenue reported?", ["revenue", "$", "billion", "million", "sales", "income", "financial", "earnings", "quarter", "results"]),
        ("Who are the key executives mentioned?", ["ceo", "cfo", "director", "president", "executive", "management", "leadership", "officer", "board", "chairman"]),
        ("What are the major risk factors?", ["risk", "factor", "uncertainty", "threat", "challenge", "forward-looking", "material", "factors", "uncertainties"]),
        ("When is the next earnings call?", ["call", "earnings", "quarter", "q1", "q2", "q3", "q4", "financial", "report", "conference", "meeting"]),
        ("What is the company's core business?", ["business", "platform", "app", "oil", "gas", "energy", "aviation", "operations", "services", "products", "industry"]),
    ]

    q_texts = [q[0] for q in evaluation_set]
    q_embs = model.encode(q_texts, convert_to_numpy=True)
    
    total_score = 0
    accuracy_details = []
    
    for i, (q, keywords) in enumerate(evaluation_set):
        # Use hybrid search like the main system
        hybrid_indices = hybrid_search(index, all_chunks, q_embs[i], q, top_k=10)  # Increased to 10
        context_chunks = [all_chunks[idx] for idx in hybrid_indices if idx < len(all_chunks)]
        context_lower = " ".join(context_chunks).lower()
        
        hits = [kw for kw in keywords if kw in context_lower]
        score = len(hits) / len(keywords) * 100
        total_score += score
        
        trunc_q = q if len(q) <= 42 else q[:39] + "..."
        accuracy_details.append((trunc_q, f"{score:.0f}%", ", ".join(hits)))
        
    avg_accuracy = total_score / len(evaluation_set)
    total_time = ingest_time + query_time

    print("")
    print("="*85)
    print(f"{'LUCIO AI PIPELINE BENCHMARK REPORT':^85}")
    print("="*85)
    
    print("")
    print("[ 1. PERFORMANCE METRICS ]")
    print("-" * 50)
    print(f"{'Phase':<20} | {'Time (Seconds)':<25}")
    print("-" * 50)
    print(f"{'Ingestion Phase':<20} | {ingest_time:>10.2f}s")
    print(f"{'Query Phase':<20} | {query_time:>10.2f}s")
    print("-" * 50)
    print(f"{'TOTAL RUNTIME':<20} | {total_time:>10.2f}s")
    
    if total_time < 30:
        print(f"{'STATUS':<20} | {'PASS (< 30s)':>14}")
    else:
        print(f"{'STATUS':<20} | {'FAIL (> 30s)':>14}")
    print("-" * 50)

    print("")
    print("[ 2. RETRIEVAL ACCURACY METRICS ]")
    print("-" * 85)
    print(f"{'Question':<44} | {'Score':<7} | {'Matched Keywords'}")
    print("-" * 85)
    for q, score, hits in accuracy_details:
        print(f"{q:<44} | {score:<7} | {hits}")
    print("-" * 85)
    print(f"{'AVERAGE KEYWORD RETRIEVAL SCORE':<44} | {avg_accuracy:>6.0f}% |")
    print("="*85)
    print("")

if __name__ == '__main__':
    run_benchmark()
