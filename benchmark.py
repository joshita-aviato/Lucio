import os
import time
import json
import subprocess
import re
import sys
from typing import Set

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def extract_numbers(text: str) -> Set[str]:
    nums = set()
    for m in re.finditer(r'[\$₹]?\s*[\d,]+\.?\d*\s*(?:billion|million|thousand|cr|crore|lakh|%|b|m|k)?', text.lower()):
        n = re.sub(r'[,\s]', '', m.group()).strip()
        if n: nums.add(n)
    for m in re.finditer(r'\b\d[\d,.]*\d\b|\b\d+\b', text):
        nums.add(m.group().replace(',', ''))
    return nums


def extract_key_terms(text: str) -> Set[str]:
    terms = set()
    for m in re.finditer(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text):
        terms.add(m.group().lower())
    for m in re.finditer(r'\b[A-Z]{2,}\b', text):
        terms.add(m.group().lower())
    return terms


def normalize(text: str) -> str:
    if not text: return ""
    t = str(text).strip().lower()
    for p in ["according to the document,", "based on the context,", "the answer is",
              "answer:", "based on the provided context,", "based on the provided documents,"]:
        if t.startswith(p): t = t[len(p):].strip()
    t = t.rstrip('.')
    return re.sub(r'\s+', ' ', t).strip()


def score_answer(predicted: str, ground_truth: str) -> dict:
    pred_raw, truth_raw = str(predicted).strip(), str(ground_truth).strip()
    pred, truth = normalize(pred_raw), normalize(truth_raw)

    if not pred or not truth:
        return {"exact_match": False, "contains_truth": False, "token_f1": 0.0,
                "number_match": 0.0, "key_term_match": 0.0, "composite": 0.0}

    exact = pred == truth
    pred_has_truth = truth in pred
    truth_has_pred = pred in truth

    pt, tt = set(pred.split()), set(truth.split())
    common = pt & tt
    f1 = (2 * (len(common)/len(pt)) * (len(common)/len(tt)) /
          ((len(common)/len(pt)) + (len(common)/len(tt)))) if common else 0.0

    pn, tn = extract_numbers(pred_raw), extract_numbers(truth_raw)
    num_score = len(pn & tn) / len(tn) if tn else 0.0

    pk, tk = extract_key_terms(pred_raw), extract_key_terms(truth_raw)
    term_score = len(pk & tk) / len(tk) if tk else 0.0

    if exact: composite = 1.0
    elif pred_has_truth: composite = 0.9 + 0.1 * f1
    elif truth_has_pred and len(pred) > 5: composite = 0.75 + 0.1 * f1
    else: composite = max(f1, 0.35*f1 + 0.35*num_score + 0.30*term_score)

    return {"exact_match": exact, "contains_truth": pred_has_truth, "token_f1": round(f1, 3),
            "number_match": round(num_score, 3), "key_term_match": round(term_score, 3),
            "composite": round(composite, 3)}


def run_benchmark():
    print("=" * 90)
    print(f"{'LUCIO AI PIPELINE BENCHMARK':^90}")
    print("=" * 90)
    print()

    for f in ["chunks.json", "metadata.json", "tfidf_index.json", "tfidf_vectors.npz",
              "tfidf_vocab.json", "faiss_index.bin", "results.json"]:
        if os.path.exists(f): os.remove(f)

    # Phase 1: Ingest
    print("[Phase 1] Running Ingestion...")
    t0 = time.time()
    r = subprocess.run(["uv", "run", "python", "lucio_cli.py", "ingest"], capture_output=True, text=True)
    ingest_time = time.time() - t0
    if r.returncode != 0:
        print("  INGESTION FAILED!"); print(r.stderr[-2000:]); return
    meta_info = ""
    if os.path.exists("metadata.json"):
        with open("metadata.json") as f:
            m = json.load(f)
            meta_info = f"  Documents: {m.get('doc_count','?')}, Chunks: {m.get('chunk_count','?')}"
    print(f"  Ingestion completed in {ingest_time:.2f}s")
    if meta_info: print(meta_info)

    # Phase 2: Run
    print()
    print("[Phase 2] Running Queries + Gemini Answering...")
    t0 = time.time()
    r = subprocess.run(["uv", "run", "python", "lucio_cli.py", "run"], capture_output=True, text=True)
    query_time = time.time() - t0
    if r.returncode != 0:
        print("  QUERY PHASE FAILED!"); print(r.stderr[-2000:]); return
    print(f"  Query phase completed in {query_time:.2f}s")

    # Phase 3: Evaluate
    print()
    print("[Phase 3] Evaluating Against Ground Truth...")
    if not os.path.exists("results.json"):
        print("  ERROR: results.json not found"); return

    with open("results.json") as f: results = json.load(f)
    total_time = ingest_time + query_time

    print()
    print("=" * 90)
    print(f"{'BENCHMARK RESULTS':^90}")
    print("=" * 90)

    # Timing
    print()
    print("[ PERFORMANCE ]")
    print("-" * 55)
    print(f"  {'Phase':<25} {'Time':>10}")
    print("-" * 55)
    print(f"  {'Ingestion':<25} {ingest_time:>9.2f}s")
    print(f"  {'Query + LLM':<25} {query_time:>9.2f}s")
    print(f"  {'TOTAL':<25} {total_time:>9.2f}s")
    status = "PASS" if total_time < 30 else "FAIL"
    c = "\033[92m" if status == "PASS" else "\033[91m"
    print(f"  {'Status':<25} {c}{status} ({'< 30s' if status == 'PASS' else '> 30s'})\033[0m")
    print("-" * 55)

    # Per-question
    print()
    print("[ ANSWER ACCURACY ]")
    print("-" * 95)
    print(f"  {'#':<3} {'Question':<35} {'Score':>6} {'Nums':>6} {'Terms':>6}  {'Match':<10} {'Src':<8}")
    print("-" * 95)

    eval_scores = []
    gemini_ct = fallback_ct = 0

    for i, r in enumerate(results):
        src = r.get("source", "?")
        if src == "gemini": gemini_ct += 1
        else: fallback_ct += 1

        gt = r.get("ground_truth")
        if gt and str(gt).strip().lower() not in ('nan', 'none', ''):
            sc = score_answer(r.get("answer", ""), gt)
            eval_scores.append(sc)
            q_short = r["question"][:33] + ".." if len(r["question"]) > 35 else r["question"]
            comp = sc['composite']
            if sc['exact_match']: mt = "EXACT"
            elif sc['contains_truth']: mt = "CONTAINS"
            elif comp > 0: mt = f"F1={sc['token_f1']:.0%}"
            else: mt = "MISS"
            clr = "\033[92m" if comp >= 0.7 else ("\033[93m" if comp >= 0.3 else "\033[91m")
            print(f"  {i+1:<3} {q_short:<35} {clr}{comp:>5.0%}\033[0m {sc['number_match']:>5.0%} "
                  f"{sc['key_term_match']:>5.0%}  {mt:<10} {src:<8}")
        else:
            print(f"  {i+1:<3} {r['question'][:35]:<35} {'N/A':>6} {'':>6} {'':>6}  {'NO GT':<10} {src:<8}")

    print("-" * 95)

    # Detailed
    print()
    print("[ DETAILED COMPARISON ]")
    print("-" * 90)
    for i, r in enumerate(results):
        gt = r.get("ground_truth")
        if gt and str(gt).strip().lower() not in ('nan', 'none', ''):
            sc = score_answer(r.get("answer", ""), gt)
            print(f"  Q{i+1}: {r['question']}")
            print(f"    Expected:  {gt[:150]}")
            print(f"    Got:       {r.get('answer','')[:150]}")
            print(f"    Score:     {sc['composite']:.0%} (nums={sc['number_match']:.0%}, "
                  f"terms={sc['key_term_match']:.0%}, f1={sc['token_f1']:.0%})")
            print()

    # Summary
    if eval_scores:
        n = len(eval_scores)
        exact = sum(1 for s in eval_scores if s['exact_match'])
        contains = sum(1 for s in eval_scores if s['contains_truth'])
        avg_f1 = sum(s['token_f1'] for s in eval_scores) / n
        avg_num = sum(s['number_match'] for s in eval_scores) / n
        avg_term = sum(s['key_term_match'] for s in eval_scores) / n
        avg_comp = sum(s['composite'] for s in eval_scores) / n
        high = sum(1 for s in eval_scores if s['composite'] >= 0.7)

        print("=" * 90)
        print(f"{'SUMMARY':^90}")
        print("=" * 90)
        print(f"  Questions Evaluated:    {n}")
        print(f"  Exact Matches:          {exact}/{n} ({exact/n:.0%})")
        print(f"  Contains Ground Truth:  {contains}/{n} ({contains/n:.0%})")
        print(f"  High Quality (>=70%):   {high}/{n} ({high/n:.0%})")
        print(f"  Avg Token F1:           {avg_f1:.1%}")
        print(f"  Avg Number Match:       {avg_num:.1%}")
        print(f"  Avg Key Term Match:     {avg_term:.1%}")
        print(f"  Avg Composite:          {avg_comp:.1%}")
        print()
        print(f"  Gemini Answers:         {gemini_ct}")
        print(f"  No Answer / Fallback:   {fallback_ct}")
        print(f"  Total Time:             {total_time:.2f}s {'(PASS)' if total_time < 30 else '(FAIL)'}")
        print("=" * 90)


if __name__ == '__main__':
    run_benchmark()