import os
import time
import json
import subprocess

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    import json
    with open("chunks.json", 'r', encoding='utf-8') as f:
        all_chunks = json.load(f)

    evaluation_set = [
        ("What is the total revenue reported?", ["revenue", "$", "billion", "million", "sales", "income", "financial", "earnings", "quarter", "results"]),
        ("Who are the key executives mentioned?", ["ceo", "cfo", "director", "president", "executive", "management", "leadership", "officer", "board", "chairman"]),
        ("What are the major risk factors?", ["risk", "factor", "uncertainty", "threat", "challenge", "forward-looking", "material", "factors", "uncertainties"]),
        ("When is the next earnings call?", ["call", "earnings", "quarter", "q1", "q2", "q3", "q4", "financial", "report", "conference", "meeting"]),
        ("What is the company's core business?", ["business", "platform", "app", "oil", "gas", "energy", "aviation", "operations", "services", "products", "industry"]),
    ]
    
    total_score = 0
    accuracy_details = []
    
    for i, (q, keywords) in enumerate(evaluation_set):
        # Use the same hybrid retrieve as the main system
        from lucio_cli import hybrid_retrieve, BM25Index
        import json
        
        # Load BM25 index
        with open('tfidf_index.json', 'r') as f:
            bm25 = BM25Index.from_dict(json.load(f))
        
        retrieved_indices = hybrid_retrieve(bm25, all_chunks, q, model=None, top_k=8, bm25_candidates=30)
        context_chunks = [all_chunks[idx] for idx in retrieved_indices if idx < len(all_chunks)]
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
