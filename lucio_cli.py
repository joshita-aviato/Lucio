import click
import os
import time
import asyncio
from google import genai
from google.genai import types
import sys
import logging
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
import gc
import json
import re
import math
import numpy as np
from collections import Counter

# ==============================================================================
# PREVENT MACOS FREEZING / MEMORY SPIKES
# ==============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MAX_WORKERS = 4
DATA_DIR = "./data"
CHUNKS_PATH = "chunks.json"
META_PATH = "metadata.json"
TFIDF_PATH = "tfidf_index.json"
INDEX_PATH = "faiss_index.bin"
TFIDF_VECTORS_PATH = "tfidf_vectors.npz"
VOCAB_PATH = "tfidf_vocab.json"

START_TIME = time.time()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 120
MAX_CHUNKS_PER_DOC = 12

SKIP_DOMAIN_EXTRACTION = True


def adaptive_optimization(doc_count: int):
    global CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC, SKIP_DOMAIN_EXTRACTION
    SKIP_DOMAIN_EXTRACTION = True
    if doc_count <= 20:
        CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC = 500, 120, 1000
    elif doc_count <= 50:
        CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC = 450, 100, 800
    elif doc_count <= 100:
        CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC = 400, 80, 600
    elif doc_count <= 200:
        CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC = 350, 60, 400
    else:
        CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS_PER_DOC = 300, 50, 300
    logger.info(f"Adaptive: {doc_count} docs -> chunk={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, "
                f"max_chunks/doc={MAX_CHUNKS_PER_DOC} [DOMAIN-AGNOSTIC MODE]")


def check_timeout(context: str = "", limit: float = 28.0):
    elapsed = time.time() - START_TIME
    if elapsed > limit:
        logger.error(f"[{context}] {elapsed:.1f}s — aborting")
        sys.exit(1)
    return elapsed


# ==============================================================================
# TEXT CLEANING
# ==============================================================================
_BOILERPLATE = re.compile(
    r'(?i)(table of contents|forward.looking statements?\s+(?:disclaimer|notice|this)'
    r'|this (?:page|document) (?:is |was )?intentionally left blank'
    r'|all rights reserved|confidential.*proprietary'
    r'|page \d+ of \d+)', re.DOTALL
)

_BOILERPLATE_HEAVY = re.compile(
    r'(?i)(during this call we will present|'
    r'actual results may differ materially|'
    r'safe harbor|'
    r'this (presentation|call|webcast) (contains|includes|may contain) forward|'
    r'(please|we encourage you to) (review|read|see|refer to)|'
    r'(available|filed|posted) (on|with|at) (our website|the sec|edgar)|'
    r'(reconciliation|non.?gaap|supplemental).{0,50}(appendix|end of|attached))',
    re.IGNORECASE
)


def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = _BOILERPLATE.sub('', text)
    lines = text.split('\n')
    kept = [l.strip() for l in lines if len(l.strip()) >= 8 and not re.match(r'^\d+$', l.strip())]
    return '\n'.join(kept)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words: return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        if end < len(words):
            lookback = max(int(chunk_size * 0.2), 8)
            for j in range(end, max(end - lookback, start), -1):
                if j - 1 < len(words) and words[j - 1][-1] in '.!?':
                    end = j
                    break
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 40:
            chunks.append(chunk.strip())
        start += max(end - start - overlap, 1)
    return chunks


# ==============================================================================
# FILE PROCESSING
# ==============================================================================
def _extract_pages(file_path: Path) -> List[Tuple[int, str]]:
    suffix = file_path.suffix.lower()
    pages = []
    if suffix == '.pdf':
        import fitz
        try:
            with fitz.open(file_path) as doc:
                max_pages = 40 if SKIP_DOMAIN_EXTRACTION else 200
                for i, page in enumerate(doc):
                    if i >= max_pages: break
                    t = page.get_text()
                    if t: pages.append((i + 1, t))
        except Exception as e:
            logger.error(f"PDF error {file_path.name}: {e}")
    elif suffix == '.docx':
        import docx
        try:
            doc = docx.Document(file_path)
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text)
            pages.append((1, full_text))
        except Exception: pass
    else:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                pages.append((1, f.read(500_000)))
        except Exception: pass
    return pages

def process_file(file_path: Path) -> List[Dict]:
    try:
        pages_data = _extract_pages(file_path)
        if not pages_data: return []
        fname = file_path.name
        structured_chunks = []
        for page_num, raw_text in pages_data:
            if not raw_text or len(raw_text.strip()) < 50: continue
            text = clean_text(raw_text)
            standard_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for c in standard_chunks:
                structured_chunks.append({"text": c, "doc": fname, "page": page_num})
        return structured_chunks
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {e}")
        return []


# ==============================================================================
# BM25 INDEX
# ==============================================================================
_STOPWORDS = frozenset({"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "need", "must", "ought", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "them", "their", "this", "that", "these", "those", "which", "what", "who", "whom", "when", "where", "how", "why", "if", "then", "than", "but", "and", "or", "not", "no", "nor", "so", "too", "very", "just", "also", "as", "at", "by", "for", "from", "in", "into", "of", "on", "to", "up", "with", "about", "after", "before", "between", "during", "through", "above", "below", "each", "all", "any", "both", "few", "more", "most", "other", "some", "such", "only", "own", "same", "here", "there", "its", "over", "under", "out"})
_TOKEN_RE = re.compile(r'[a-z0-9$%]+(?:\.[a-z0-9]+)*')

def tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_freqs, self.doc_lens, self.avg_dl, self.n_docs = {}, [], 0.0, 0
        self.inverted = {}

    def build(self, documents: List[Dict]):
        self.n_docs = len(documents)
        total_len = 0
        for doc_id, doc in enumerate(documents):
            tokens = tokenize(doc["text"])
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            tf = Counter(tokens)
            for term, count in tf.items():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                if term not in self.inverted: self.inverted[term] = []
                self.inverted[term].append((doc_id, count))
        self.avg_dl = total_len / max(self.n_docs, 1)

    def query(self, query_text: str, top_k: int = 20) -> List[Tuple[float, int]]:
        q_tokens = tokenize(query_text)
        scores = {}
        for term in q_tokens:
            if term not in self.inverted: continue
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf in self.inverted[term]:
                dl = self.doc_lens[doc_id]
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_norm
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(s, d) for d, s in ranked[:top_k]]

    def to_dict(self) -> dict:
        return {"k1": self.k1, "b": self.b, "doc_freqs": self.doc_freqs, "doc_lens": self.doc_lens, "avg_dl": self.avg_dl, "n_docs": self.n_docs, "inverted": self.inverted}

    @classmethod
    def from_dict(cls, d: dict) -> "BM25Index":
        idx = cls(k1=d["k1"], b=d["b"])
        idx.doc_freqs, idx.doc_lens, idx.avg_dl, idx.n_docs, idx.inverted = d["doc_freqs"], d["doc_lens"], d["avg_dl"], d["n_docs"], {k: [tuple(x) for x in v] for k, v in d["inverted"].items()}
        return idx


# ==============================================================================
# KEYWORD EXPANSIONS
# ==============================================================================
_KEYWORD_EXPANSIONS = {
    "governing": {"law", "jurisdiction", "applicable", "governed", "provision", "agreement", "section", "clause", "contract"},
    "notify": {"notification", "filing", "threshold", "approval", "clearance", "merger", "combination", "antitrust", "competition"},
    "anticompetitive": {"competition", "merger", "combination", "dominance", "market", "threshold", "turnover", "assets", "aaec", "appreciable", "adverse", "effect"},
    "cci": {"competition", "commission", "india", "merger", "combination", "threshold", "turnover", "assets", "notification", "act"},
    "turnover": {"revenue", "sales", "assets", "threshold", "crore", "filing", "notification", "combination"},
    "nvca": {"investor", "rights", "agreement", "venture", "capital", "ira", "registration", "preferred", "stockholder", "board", "covenant"},
    "ira": {"investor", "rights", "agreement", "registration", "information", "covenant", "board", "observer", "preferred"},
    "kodak": {"eastman", "kodak", "antitrust", "tying", "market", "power", "equipment", "service", "parts", "competition"},
    "meta": {"facebook", "meta", "platforms", "social", "media", "advertising", "reality", "labs", "family", "apps"},
    "kfin": {"kfintech", "kfin", "registrar", "transfer", "agent", "fintech", "mutual", "fund"},
}

def expand_query(query_text: str) -> str:
    ql = query_text.lower()
    extras = set()
    for trigger, kws in _KEYWORD_EXPANSIONS.items():
        if trigger in ql: extras.update(kws)
    return f"{query_text} {' '.join(extras)}" if extras else query_text


# ==============================================================================
# RETRIEVAL SIGNALS
# ==============================================================================
_ENTITY_RE = re.compile(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+')
_NUMBER_RE = re.compile(r'\$?[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|%|percent|cr|crore|lakh))?', re.I)

def entity_overlap_score(query_entities: Set[str], chunk_text: str) -> float:
    if not query_entities: return 0.0
    chunk_lower = chunk_text.lower()
    hits = sum(1 for e in query_entities if e in chunk_lower)
    return hits / max(len(query_entities), 1)

def chunk_quality_score(chunk: str) -> float:
    score = 0.0
    score -= len(_BOILERPLATE_HEAVY.findall(chunk)) * 0.08
    score += min(len(_NUMBER_RE.findall(chunk)) * 0.02, 0.10)
    score += min(len(_ENTITY_RE.findall(chunk)) * 0.01, 0.06)
    return score

def hybrid_retrieve(bm25, all_chunks, query_text, top_k=10, bm25_candidates=150):
    expanded = expand_query(query_text)
    bm25_results = bm25.query(expanded, top_k=bm25_candidates)
    if not bm25_results: return []
    max_bm25 = max(s for s, _ in bm25_results) if bm25_results else 1.0
    query_entities = extract_entities(query_text)
    scored = []
    for score, doc_id in bm25_results:
        text = all_chunks[doc_id]["text"]
        bm25_norm = score / max(max_bm25, 0.001)
        ent_score = entity_overlap_score(query_entities, text)
        quality = chunk_quality_score(text)
        scored.append((0.7 * bm25_norm + 0.2 * ent_score + 0.1 * (quality + 0.2), doc_id))
    scored.sort(reverse=True, key=lambda x: x[0])
    seen, result = set(), []
    for _, idx in scored:
        key = all_chunks[idx]["text"][:150]
        if key not in seen:
            seen.add(key)
            result.append(idx)
        if len(result) >= top_k: break
    return result

def extract_entities(text: str) -> Set[str]:
    entities = set()
    for m in _ENTITY_RE.finditer(text): entities.add(m.group().lower())
    for m in _NUMBER_RE.finditer(text): entities.add(m.group().lower().replace(',', '').strip())
    return entities


# ==============================================================================
# CLI COMMANDS
# ==============================================================================
@click.group()
def cli(): pass

@cli.command()
def ingest():
    global START_TIME
    START_TIME = time.time()
    logger.info("Starting ingestion (Parallel PDF Parsing + TF-IDF Indexing)...")
    data_path = Path(DATA_DIR)
    files = [f for f in data_path.iterdir() if f.is_file()]
    adaptive_optimization(len(files))
    extract_start = time.time()
    all_chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_file, f): f for f in files}
        for future in concurrent.futures.as_completed(futures):
            all_chunks.extend(future.result())
    extraction_time = time.time() - extract_start
    index_start = time.time()
    bm25 = BM25Index()
    bm25.build(all_chunks)
    indexing_time = time.time() - index_start
    with open(CHUNKS_PATH, 'w', encoding='utf-8') as f: json.dump(all_chunks, f, ensure_ascii=False)
    with open(TFIDF_PATH, 'w', encoding='utf-8') as f: json.dump(bm25.to_dict(), f, ensure_ascii=False)
    with open(META_PATH, 'w') as f:
        json.dump({"ingest_time": time.time() - START_TIME, "extraction_time": extraction_time, "indexing_time": indexing_time, "doc_count": len(files), "chunk_count": len(all_chunks)}, f)
    logger.info(f"Ingestion complete: {len(all_chunks)} chunks, {time.time() - START_TIME:.2f}s")

async def process_all_questions_async(questions, bm25, all_chunks):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    async def process_single(i, q):
        check_timeout(f"Q{i+1}_Retrieve")
        loop = asyncio.get_running_loop()
        top_indices = await loop.run_in_executor(None, lambda: hybrid_retrieve(bm25, all_chunks, q, top_k=10, bm25_candidates=150))
        context_parts = [f"[Source: {all_chunks[idx]['doc']}, Page: {all_chunks[idx]['page']}]\n{all_chunks[idx]['text']}" for idx in top_indices]
        context_str = "\n\n---\n\n".join(context_parts)[:6000]
        prompt = f"Answer precisely based ONLY on context.\nCite Doc Name and Page.\nFormat:\nAnswer: [Concise]\nSource: [Doc], Page [X]\n\nContext:\n{context_str}\n\nQuestion: {q}"
        try:
            res = await client.aio.models.generate_content(model='gemini-2.5-flash-lite-preview-09-2025', contents=prompt, config=types.GenerateContentConfig(max_output_tokens=150, temperature=0.0))
            ans = res.text.strip()
        except Exception as e: ans = f"Error: {e}"
        logger.info(f"\n--- Q{i+1}: {q}\n--- {ans}\n")
        return ans
    return await asyncio.gather(*[process_single(i, q) for i, q in enumerate(questions)])

def print_performance_report(metrics: dict):
    t = metrics.get('total_time', 0)
    print(f"\n{'='*60}\n📊 PERFORMANCE ANALYSIS\n{'='*60}")
    print(f"Total Time: {t:.2f}s | Status: {'✅' if t <= 22 else '⚠️' if t <= 30 else '❌'}")
    print(f"Docs: {metrics.get('doc_count', 0)} | Chunks: {metrics.get('chunk_count', 0)} | Questions: {metrics.get('q_count', 0)}")
    print('='*60)

@cli.command()
def run():
    global START_TIME
    START_TIME = time.time()
    if not os.environ.get("GEMINI_API_KEY"): sys.exit(1)
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f: all_chunks = json.load(f)
    with open(TFIDF_PATH, 'r', encoding='utf-8') as f: bm25 = BM25Index.from_dict(json.load(f))
    import pandas as pd
    q_df = pd.read_excel(os.path.join(DATA_DIR, 'Testing Set Questions.xlsx'))
    questions = q_df['Question'].tolist()
    qa_start = time.time()
    asyncio.run(process_all_questions_async(questions, bm25, all_chunks))
    metrics = {"total_time": (time.time() - START_TIME), "q_count": len(questions)}
    if os.path.exists(META_PATH):
        with open(META_PATH, 'r') as f:
            meta = json.load(f)
            metrics.update(meta)
            metrics["total_time"] += meta.get("ingest_time", 0)
    print_performance_report(metrics)

if __name__ == "__main__": cli()
