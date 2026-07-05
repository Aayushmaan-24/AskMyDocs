"""
eval_ragas.py — RAGAS evaluation pipeline
Metrics: faithfulness, answer_relevancy, context_precision, context_recall
Run:  python -m tests.eval_ragas
CI:   pytest tests/eval_ragas.py -v
"""

import json
import os
import sys
import pytest
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    _faithfulness,
    _answer_relevance,
    _context_precision,
    _context_recall,
)
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.pipeline import ask

load_dotenv()

# ── Thresholds — CI fails if any metric drops below these ─────────

THRESHOLDS = {
    "_faithfulness" : 0.70,
    "_answer_relevance" : 0.70,
    "_context_precision" : 0.60,
    "_context_recall" : 0.60
}

GOLDEN_QA_PATH = "tests/golden_qa.json"