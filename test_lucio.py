import os
import time
import json
from pathlib import Path
from click.testing import CliRunner
import pytest

import lucio_cli
from lucio_cli import cli, chunk_text, INDEX_PATH, CHUNKS_PATH, META_PATH

def test_chunk_text():
    """Verify that chunking logic handles boundaries and overlap correctly."""
    text = "word " * 100
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    assert len(chunks) > 1
    assert len(chunks[0].split()) == 30
    assert len(chunks[1].split()) == 30

def test_ingest_and_run(tmp_path, monkeypatch):
    """
    Test the full ingestion and querying pipeline securely.
    Ensures that processing completes on 17 files well under 30 seconds
    and handles output correctly.
    """
    runner = CliRunner()
    
    # Store the absolute path to the actual data directory from the workspace
    workspace_data_dir = os.path.abspath("./data")
    
    # Monkeypatch DATA_DIR inside the lucio_cli module to point to the real data folder
    monkeypatch.setattr(lucio_cli, "DATA_DIR", workspace_data_dir)
    
    # Change the current working directory to a clean temporary directory for this test
    # This prevents the test from polluting your workspace and avoids state conflicts
    monkeypatch.chdir(tmp_path)
    
    # --- PHASE 1: INGESTION ---
    start_time = time.time()
    result_ingest = runner.invoke(cli, ['ingest'])
    ingest_time = time.time() - start_time
    
    # Validations
    assert result_ingest.exit_code == 0, f"Ingest command failed: {result_ingest.output}"
    assert os.path.exists(INDEX_PATH), "FAISS index file was not generated"
    assert os.path.exists(CHUNKS_PATH), "Chunks JSON file was not generated"
    assert os.path.exists(META_PATH), "Metadata JSON file was not generated"
    
    # Hackathon Performance Constraint
    assert ingest_time < 30.0, f"Ingestion took {ingest_time:.2f}s, which exceeds the 30s limit"
    
    with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    assert len(chunks) > 0, "No chunks were generated from the dataset"
    # 17 documents * MAX_CHUNKS_PER_DOC (10) = max 170 chunks
    assert len(chunks) <= 170, f"Failed to cap chunks per document: found {len(chunks)} chunks"
    
    # --- PHASE 2: RUN / QUERY ---
    start_time = time.time()
    result_run = runner.invoke(cli, ['run'])
    run_time = time.time() - start_time
    
    # Validations
    assert result_run.exit_code == 0, f"Run command failed: {result_run.output}"
    
    # Hackathon Performance Constraint
    assert run_time < 20.0, f"Run/Query phase took {run_time:.2f}s, which is too slow"
    
    # Overall Performance Check
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    total_time = meta.get("ingest_time", 0) + run_time
    assert total_time < 30.0, f"Total execution time {total_time:.2f}s exceeded the strict 30s hackathon rule"
