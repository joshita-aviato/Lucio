import os
import time
import json
import subprocess
import re
import sys
from typing import Set, List, Dict, Tuple


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def _normalize_number_suffix(n: str) -> str:
    n = n.lower().strip()
    for long, short in [("billion", "b"), ("million", "m"), ("thousand", "k"),
                         ("crore", "cr"), ("lakh", "l")]:
        n = n.replace(long, short)
    return n


def extract_numbers(text: str) -> Set[str]:
    nums = set()
    for m in re.finditer(r'[\$₹]?\s*[\d,]+\.?\d*\s*(?:billion|million|thousand|cr|crore|lakh|%|b|m|k)?', text.lower()):
        n = re.sub(r'[,\s]', '', m.group()).strip()
        if n: nums.add(_normalize_number_suffix(n))
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


# ==============================================================================
# DOCUMENT / PAGE REFERENCE SCORING
# ==============================================================================

def parse_ground_truth_doc_ref(ref_str: str) -> List[Dict]:
    """Parse ground truth 'Documents Referred' into list of {doc_name, page}.

    Format examples:
      'META-Q1-2025-Earnings-Call-Transcript-1.pdf, p4'
      'CCI Combination Guide.pdf, p4\\n1652423343024.pdf, p52\\n1652423343024.pdf, p54'
    """
    if not ref_str or str(ref_str).strip().lower() in ('nan', 'none', ''):
        return []

    entries = []
    # Split by newline for multiple references
    lines = str(ref_str).strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Try to parse "filename.ext, pN"
        m = re.match(r'^(.+?),\s*p(\d+)$', line)
        if m:
            doc_name = m.group(1).strip()
            page = int(m.group(2))
            entries.append({"doc_name": doc_name, "page": page})
        else:
            # Just a filename without page
            entries.append({"doc_name": line.strip(), "page": None})

    return entries


def score_doc_reference(predicted_docs: List[str], predicted_pages: List[int],
                        ground_truth_ref: str) -> Dict:
    """Score whether the system returned the correct document names and page numbers."""
    gt_entries = parse_ground_truth_doc_ref(ground_truth_ref)
    if not gt_entries:
        return {"doc_name_match": None, "page_match": None, "doc_page_match": None}

    # Expected doc names (normalized for comparison)
    gt_doc_names = set()
    gt_pages = set()
    gt_doc_page_pairs = set()
    for entry in gt_entries:
        dn = entry["doc_name"].strip().lower()
        gt_doc_names.add(dn)
        if entry["page"] is not None:
            gt_pages.add(entry["page"])
            gt_doc_page_pairs.add((dn, entry["page"]))

    # Predicted doc names (normalized)
    pred_doc_names = set(d.strip().lower() for d in predicted_docs if d)
    pred_pages = set(predicted_pages) if predicted_pages else set()

    # Doc name matching — check if any predicted doc name matches (partial match OK for filenames)
    doc_hits = 0
    for gt_dn in gt_doc_names:
        for pred_dn in pred_doc_names:
            # Exact match or one contains the other
            if gt_dn == pred_dn or gt_dn in pred_dn or pred_dn in gt_dn:
                doc_hits += 1
                break
    doc_score = doc_hits / len(gt_doc_names) if gt_doc_names else 0.0

    # Page matching
    if gt_pages:
        page_hits = len(pred_pages & gt_pages)
        page_score = page_hits / len(gt_pages)
    else:
        page_score = None  # No page info in ground truth

    # Combined doc+page matching
    if gt_doc_page_pairs:
        pair_hits = 0
        for gt_dn, gt_p in gt_doc_page_pairs:
            for pred_dn in pred_doc_names:
                if (gt_dn == pred_dn or gt_dn in pred_dn or pred_dn in gt_dn) and gt_p in pred_pages:
                    pair_hits += 1
                    break
        doc_page_score = pair_hits / len(gt_doc_page_pairs)
    else:
        doc_page_score = None

    return {
        "doc_name_match": round(doc_score, 3),
        "page_match": round(page_score, 3) if page_score is not None else None,
        "doc_page_match": round(doc_page_score, 3) if doc_page_score is not None else None,
    }


def run_benchmark():
    print("=" * 90)
    print(f"{'LUCIO AI PIPELINE BENCHMARK':^90}")
    print("=" * 90)
    print()

    for f in ["chunks.json", "metadata.json", "tfidf_index.json", "tfidf_vectors.npz",
              "tfidf_vocab.json", "results.json"]:
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

    # Per-question answer accuracy
    print()
    print("[ ANSWER ACCURACY ]")
    print("-" * 95)
    print(f"  {'#':<3} {'Question':<35} {'Score':>6} {'Nums':>6} {'Terms':>6}  {'Match':<10} {'Src':<8}")
    print("-" * 95)

    eval_scores = []
    doc_ref_scores = []
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

        # Score document/page references
        gt_doc_ref = r.get("ground_truth_doc_ref")
        if gt_doc_ref and str(gt_doc_ref).strip().lower() not in ('nan', 'none', ''):
            pred_docs = r.get("document_name", [])
            pred_pages = r.get("page_numbers", [])
            dsc = score_doc_reference(pred_docs, pred_pages, gt_doc_ref)
            doc_ref_scores.append(dsc)
        else:
            doc_ref_scores.append(None)

    print("-" * 95)

    # Document/Page reference accuracy
    print()
    print("[ DOCUMENT & PAGE REFERENCE ACCURACY ]")
    print("-" * 100)
    print(f"  {'#':<3} {'Question':<30} {'Doc':>6} {'Page':>6} {'Both':>6}  "
          f"{'Predicted Doc':<25} {'Expected Doc':<25}")
    print("-" * 100)

    valid_doc_scores = []
    valid_page_scores = []
    valid_pair_scores = []

    for i, r in enumerate(results):
        dsc = doc_ref_scores[i]
        q_short = r["question"][:28] + ".." if len(r["question"]) > 30 else r["question"]
        pred_docs = r.get("document_name", [])
        gt_ref = r.get("ground_truth_doc_ref", "")

        if dsc and dsc.get("doc_name_match") is not None:
            doc_m = dsc["doc_name_match"]
            page_m = dsc["page_match"]
            pair_m = dsc["doc_page_match"]

            valid_doc_scores.append(doc_m)
            if page_m is not None:
                valid_page_scores.append(page_m)
            if pair_m is not None:
                valid_pair_scores.append(pair_m)

            doc_clr = "\033[92m" if doc_m >= 0.7 else ("\033[93m" if doc_m > 0 else "\033[91m")
            page_str = f"{page_m:>5.0%}" if page_m is not None else "  N/A"
            pair_str = f"{pair_m:>5.0%}" if pair_m is not None else "  N/A"
            page_clr = "\033[92m" if (page_m or 0) >= 0.7 else ("\033[93m" if (page_m or 0) > 0 else "\033[91m")

            pred_doc_str = ", ".join(pred_docs[:2]) if pred_docs else "none"
            if len(pred_doc_str) > 23: pred_doc_str = pred_doc_str[:21] + ".."

            gt_entries = parse_ground_truth_doc_ref(gt_ref)
            gt_doc_str = ", ".join(set(e["doc_name"][:20] for e in gt_entries[:2])) if gt_entries else "none"
            if len(gt_doc_str) > 23: gt_doc_str = gt_doc_str[:21] + ".."

            print(f"  {i+1:<3} {q_short:<30} {doc_clr}{doc_m:>5.0%}\033[0m "
                  f"{page_clr}{page_str}\033[0m {pair_str}  {pred_doc_str:<25} {gt_doc_str:<25}")
        else:
            print(f"  {i+1:<3} {q_short:<30} {'N/A':>6} {'N/A':>6} {'N/A':>6}  {'':>25} {'NO REF':<25}")

    print("-" * 100)

    # Detailed comparison
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
            pred_docs = r.get("document_name", [])
            pred_pages = r.get("page_numbers", [])
            gt_ref = r.get("ground_truth_doc_ref", "")
            if pred_docs or gt_ref:
                print(f"    Pred Docs: {pred_docs}, Pages: {pred_pages}")
                print(f"    GT Ref:    {gt_ref}")
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
        print()
        print("  --- Answer Accuracy ---")
        print(f"  Questions Evaluated:    {n}")
        print(f"  Exact Matches:          {exact}/{n} ({exact/n:.0%})")
        print(f"  Contains Ground Truth:  {contains}/{n} ({contains/n:.0%})")
        print(f"  High Quality (>=70%):   {high}/{n} ({high/n:.0%})")
        print(f"  Avg Token F1:           {avg_f1:.1%}")
        print(f"  Avg Number Match:       {avg_num:.1%}")
        print(f"  Avg Key Term Match:     {avg_term:.1%}")
        print(f"  Avg Composite:          {avg_comp:.1%}")
        print()

        # Document/page reference summary
        print("  --- Document & Page Attribution ---")
        if valid_doc_scores:
            avg_doc = sum(valid_doc_scores) / len(valid_doc_scores)
            doc_correct = sum(1 for s in valid_doc_scores if s >= 1.0)
            print(f"  Doc Name Evaluated:     {len(valid_doc_scores)}")
            print(f"  Doc Name Exact Match:   {doc_correct}/{len(valid_doc_scores)} "
                  f"({doc_correct/len(valid_doc_scores):.0%})")
            print(f"  Avg Doc Name Match:     {avg_doc:.1%}")
        else:
            print(f"  Doc Name Evaluated:     0 (no ground truth references)")

        if valid_page_scores:
            avg_page = sum(valid_page_scores) / len(valid_page_scores)
            page_correct = sum(1 for s in valid_page_scores if s >= 1.0)
            print(f"  Page Number Evaluated:  {len(valid_page_scores)}")
            print(f"  Page Number Exact:      {page_correct}/{len(valid_page_scores)} "
                  f"({page_correct/len(valid_page_scores):.0%})")
            print(f"  Avg Page Match:         {avg_page:.1%}")

        if valid_pair_scores:
            avg_pair = sum(valid_pair_scores) / len(valid_pair_scores)
            pair_correct = sum(1 for s in valid_pair_scores if s >= 1.0)
            print(f"  Doc+Page Pair Evaluated:{len(valid_pair_scores)}")
            print(f"  Doc+Page Pair Exact:    {pair_correct}/{len(valid_pair_scores)} "
                  f"({pair_correct/len(valid_pair_scores):.0%})")
            print(f"  Avg Doc+Page Match:     {avg_pair:.1%}")

        print()
        print(f"  Gemini Answers:         {gemini_ct}")
        print(f"  No Answer / Fallback:   {fallback_ct}")
        print(f"  Total Time:             {total_time:.2f}s {'(PASS)' if total_time < 30 else '(FAIL)'}")
        print("=" * 90)


if __name__ == '__main__':
    run_benchmark()
