# Lucio AI: Document Retrieval & QA Pipeline

A high-performance document retrieval and question-answering system that combines TF-IDF vectorization with BM25 lexical search for accurate, domain-agnostic document processing.

## 🚀 Features

### Core Architecture
- **Hybrid Retrieval System**: Combines BM25 keyword matching with TF-IDF semantic similarity
- **Domain-Agnostic Processing**: Works across financial, medical, legal, and technical documents
- **Adaptive Optimization**: Dynamically adjusts processing parameters based on corpus size
- **Parallel Processing**: Multi-threaded ingestion for large document sets
- **LLM Integration**: Optional Gemini API for generative answer generation

### Text Processing
- **Multi-format Support**: PDF, DOCX, XLSX, PPTX, and plain text files
- **Intelligent Chunking**: Sentence boundary-aware text segmentation
- **Structure Preservation**: Maintains document hierarchy and tables
- **Universal Query Expansion**: Domain-independent synonym and term expansion

### Performance
- **Fast Ingestion**: Processes 200+ documents in under 20 seconds
- **Efficient Retrieval**: Sub-second query response times
- **Memory Optimized**: Sparse TF-IDF vectors with L2 normalization
- **Timeout Management**: Graceful handling of large document sets

## 📋 System Overview

### Retrieval Pipeline
1. **Document Ingestion**: Extract, clean, and chunk documents
2. **Index Building**: Create BM25 and TF-IDF indexes
3. **Query Processing**: Expand queries and retrieve relevant chunks
4. **Answer Generation**: Extractive or LLM-based answer generation
5. **Scoring**: Comprehensive answer quality evaluation

### Key Components

#### BM25 Index
- Lexical search with term frequency normalization
- Inverted index structure for fast lookups
- Document frequency statistics for IDF weighting

#### TF-IDF Vectorizer
- Reuses BM25 IDF statistics (no redundant calculations)
- Filters vocabulary (only terms appearing in ≥2 documents)
- Builds term vectors with TF-IDF weighting
- Normalizes vectors for cosine similarity via dot product

#### Hybrid Retrieval
Combines multiple signals for optimal relevance:
- **BM25 Score** (40%): Lexical term matching
- **TF-IDF Cosine** (25%): Semantic similarity
- **Phrase Bonus**: Exact multi-word phrase matches
- **Number Bonus**: Numeric value overlap
- **Name Bonus**: Proper noun matching
- **Length Adjustment**: Quality control for chunk size

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- `uv` package manager (recommended) or pip

### Setup with uv
```bash
# Clone the repository
git clone https://github.com/your-username/lucio-ai.git
cd lucio-ai

# Install dependencies
uv sync

# Set up environment variables
export GEMINI_API_KEY="your-gemini-api-key"  # Optional
```

### Setup with pip
```bash
# Clone the repository
git clone https://github.com/your-username/lucio-ai.git
cd lucio-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GEMINI_API_KEY="your-gemini-api-key"  # Optional
```

### Dependencies
- `numpy`: Numerical computations
- `pandas`: Excel file processing
- `PyMuPDF`: PDF text extraction
- `python-docx`: DOCX file processing
- `openpyxl`: Excel file processing
- `python-pptx`: PowerPoint file processing
- `nltk`: WordNet for query expansion (optional)
- `faiss-cpu`: Vector similarity (placeholder)

## 📖 Usage

### Quick Start
```bash
# 1. Place documents in ./data directory
mkdir -p data
cp your-documents.pdf data/

# 2. Ingest documents
uv run python lucio_cli.py ingest

# 3. Run queries
uv run python lucio_cli.py run
```

### Benchmarking
```bash
# Run full benchmark with evaluation
uv run python benchmark.py
```

### API Usage

#### Document Ingestion
```python
from lucio_cli import ingest

# Ingest all documents in data directory
ingest()
```

#### Query Processing
```python
from lucio_cli import hybrid_retrieve, BM25Index, load_indexes

# Load indexes
bm25, tfidf_matrix, tfidf_vectorizer, chunks = load_indexes()

# Process query
query = "What is the total revenue?"
top_indices = hybrid_retrieve(
    bm25, chunks, query,
    top_k=10, bm25_candidates=60,
    tfidf_matrix=tfidf_matrix,
    tfidf_vectorizer=tfidf_vectorizer
)

# Get context chunks
context_chunks = [chunks[i] for i in top_indices]
```

#### Answer Generation
```python
from lucio_cli import build_qa_prompt, call_gemini

# Build prompt for LLM
prompt = build_qa_prompt(query, context_chunks)

# Generate answer with Gemini
answer = call_gemini(prompt)
```

## ⚙️ Configuration

### Adaptive Parameters
The system automatically adjusts these parameters based on document count:

| Document Count | Chunk Size | Overlap | Max Chunks/Doc | Max PDF Pages |
|---------------|------------|---------|----------------|---------------|
| ≤ 20          | 500        | 100     | 30             | 80            |
| ≤ 50          | 450        | 90      | 20             | 60            |
| ≤ 100         | 400        | 70      | 12             | 40            |
| ≤ 200         | 350        | 60      | 8              | 30            |
| > 200         | 300        | 50      | 6              | 25            |

### Environment Variables
```bash
# Gemini API (optional - for LLM answer generation)
GEMINI_API_KEY="your-api-key"

# Performance tuning
OMP_NUM_THREADS="1"
TOKENIZERS_PARALLELISM="false"
KMP_DUPLICATE_LIB_OK="TRUE"
```

### File Paths
```python
DATA_DIR = "./data"
CHUNKS_PATH = "chunks.json"
META_PATH = "metadata.json"
TFIDF_PATH = "tfidf_index.json"
TFIDF_VECTORS_PATH = "tfidf_vectors.npz"
VOCAB_PATH = "tfidf_vocab.json"
RESULTS_PATH = "results.json"
```

## 🎯 Query Expansion

### Universal Patterns
The system includes domain-independent query expansions:

```python
_EXPANSIONS = {
    "revenue": {"sales", "income", "earnings", "turnover", "profit"},
    "who": {"name", "person", "officer", "director"},
    "when": {"date", "year", "month"},
    "where": {"location", "headquarters", "city", "country"},
    "how much": {"amount", "total", "value", "cost"},
    # ... more patterns
}
```

### Example Expansions
- Query: `"CEO revenue"` → Expanded: `"CEO revenue sales income earnings profit"`
- Query: `"When was the company founded"` → Expanded: `"When was the company founded date year"`

## 📊 Answer Scoring

The system provides comprehensive answer evaluation:

### Scoring Metrics
- **Exact Match**: Perfect answer match
- **Containment**: Answer contains ground truth
- **Token F1**: Token-level similarity
- **Number Match**: Numeric value accuracy
- **Key Term Match**: Entity and proper noun overlap
- **Composite Score**: Weighted combination of all metrics

### Scoring Formula
```python
if exact_match:
    composite = 1.0
elif contains_truth:
    composite = 0.9 + 0.1 * f1
elif truth_has_pred and len(pred) > 5:
    composite = 0.75 + 0.1 * f1
else:
    composite = max(
        f1,
        0.35 * f1 + 0.35 * number_match + 0.30 * term_match
    )
```

## 🏗️ Architecture

### Data Flow
```
Documents → Text Extraction → Cleaning → Chunking → Indexing
    ↓
Questions → Query Expansion → Retrieval → Answer Generation → Scoring
```

### File Structure
```
lucio-ai/
├── lucio_cli.py          # Main pipeline logic
├── benchmark.py          # Evaluation and benchmarking
├── data/                 # Document collection
├── chunks.json           # Processed text chunks
├── metadata.json         # Index metadata
├── tfidf_index.json      # BM25 index
├── tfidf_vectors.npz     # TF-IDF matrix
├── tfidf_vocab.json      # Vocabulary mapping
├── results.json          # Query results and scores
└── README.md             # This file
```

## 🔧 Advanced Usage

### Custom Query Expansion
```python
# Add custom expansions
_EXPANSIONS["medical"] = {"health", "clinical", "patient", "treatment"}
_EXPANSIONS["legal"] = {"court", "case", "law", "ruling"}
```

### Timeout Configuration
```python
# Adjust ingestion timeout (seconds)
ingest_limit = min(28.0, max(20.0, doc_count / 5.0))

# Individual file timeout
future.result(timeout=5)
```

### Parallel Processing
```python
# Configure worker count
MAX_WORKERS = 4

# ThreadPoolExecutor for parallel ingestion
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(process_file, f): f for f in files}
```

## 📈 Performance Characteristics

### Memory Usage
- Sparse TF-IDF vectors minimize memory footprint
- Vocabulary filtering reduces dimensionality
- Efficient data structures (numpy arrays, compressed storage)

### Processing Speed
- Parallel document processing
- Optimized text extraction with page limits
- Fast vector operations with numpy

### Scalability
- Handles 200+ documents efficiently
- Adaptive parameter tuning
- Graceful degradation for large corpora

## 🎨 Design Principles

### Domain Agnosticism
- Universal query patterns work across domains
- No hardcoded domain-specific logic
- Flexible text extraction for various formats

### Performance First
- Sub-30 second total runtime target
- Efficient algorithms and data structures
- Parallel processing and timeout management

### Reliability
- Graceful fallbacks for missing dependencies
- Robust error handling and logging
- Comprehensive testing and evaluation

### Extensibility
- Modular architecture for easy extension
- Configurable parameters and expansion patterns
- Plugin-ready for additional features

## 🔍 Troubleshooting

### Common Issues

#### Ingestion Timeout
```bash
# Increase timeout or reduce document count
export MAX_WORKERS=2
# Or process smaller batches
```

#### Memory Issues
```bash
# Reduce chunk size for large documents
# System automatically adapts based on document count
```

#### API Key Issues
```bash
# Set Gemini API key
export GEMINI_API_KEY="your-key"
# System works without LLM as fallback
```

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/your-username/lucio-ai.git
cd lucio-ai

# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Run benchmark
uv run python benchmark.py
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function signatures
- Add comprehensive docstrings
- Include error handling and logging

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **BM25 Algorithm**: Based on Okapi BM25 probabilistic relevance model
- **TF-IDF**: Classic term frequency-inverse document frequency weighting
- **Gemini API**: Google's generative AI for answer generation
- **Open Source Libraries**: NumPy, Pandas, PyMuPDF, and more

## 📞 Contact

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact: your-email@example.com
- Project: https://github.com/your-username/lucio-ai

---

**Lucio AI** - Fast, accurate, domain-agnostic document retrieval and question answering.

