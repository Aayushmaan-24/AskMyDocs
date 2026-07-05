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

# ── 1. Build RAGAS dataset from golden Q&A ────────────────────────

def build_eval_dataset(limit: int=None) -> tuple[Dataset, list[dict]]:
    
    with open(GOLDEN_QA_PATH) as f:
        golden = json.load(f)
        
    if limit:
        golden = golden[:limit]
        
    questions, answers, contexts, ground_truths, raw_results = [] , [] , [] , [] , []
    
    print(f"\nRunning pipeline on {len(golden)} questions...")
    for i, item in enumerate(golden):
        print(f"  [{i+1}/{len(golden)}] {item['question'][:60]}...")
        result = ask(item["question"], top_k=10, top_n=5)
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append([c['text'] for c in result['chunks']])
        ground_truths.append(item['ground_truth'])
        raw_results.append(result)
        
    dataset = Dataset.from_dict({
        "question" : questions,
        "answer" : answers,
        "contexts" : contexts,
        "ground_truth" : ground_truths,
    })
    
    return dataset, raw_results
    
# ── 2. Run RAGAS evaluation ───────────────────────────────────────

def run_evaluation(limit: int=None) -> dict:
    
    dataset , raw_result = build_eval_dataset(limit=limit)
    
    # Groq as LLM
    llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    ))
    
    # use local BGE embeddings
    embeddings = LangchainEmbeddingsWrapper(
        FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )
    
    print(f"\nRunning RAGAS Evaluation...")
    scores = evaluate(
        dataset=Dataset,
        metrics=[_faithfulness, _answer_relevance, _context_precision, _context_recall]
        llm = llm,
        embeddings=embeddings,
    )
    
    results = {
        "_faithfulness" : round(float(scores['_faithfulness']), 4),
        "_answer_relevance" :round(float(scores["_answer_relevancy"]), 4),
        "_context_precision" : round(float(scores["_context_precision"]), 4),
        "_context_recall" : round(float(scores["_context_recall"]), 4),
    }
    
    return results

# ── 3. Print report ───────────────────────────────────────────────

def print_report(scores: dict) -> None:
    
    print("\n" + "="*50)
    print("RAGAS EVALUATION REPORT")
    print("="*50)
    
    for metric, score in scores.items():
        threshold = THRESHOLDS[metric]
        status = "✓ PASS" if score >= threshold else "✗ FAIL"
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"{metric:22s} {bar} {score:.4f}  [{status}]  (threshold: {threshold})")
    print("="*50)
    
    failed = [m for m, s in scores.items() if s < THRESHOLDS[m]]
    if failed:
        print(f"\n✗ FAILED metrics: {', '.join(failed)}")
        print("CI gate would BLOCK this merge.")
    else:
        print("\n✓ All metrics passed. CI gate would ALLOW this merge.")
        
# ── 4. Save scores for CI comparison ─────────────────────────────

def save_scores(scores: dict, path: str = "tests/last_eval_scores.json") -> None:
    with open(path, 'w') as f:
        json.dumps(scores, f, indent=2)
    print(f"\nScores saved → {path}")
    
# ── 5. Pytest CI gate tests ───────────────────────────────────────

@pytest.fixture(scope="module")
def eval_scores():
    """Run eval once, share results across all threshold tests."""
    limit = 5 if os.getenv("CI") else None
    return run_evaluation(limit=limit)

def test_faithfulness(eval_scores):
    score = eval_scores['_faithfulness']
    print(f"\nFaithfulness: {score:.4f} (threshold: {THRESHOLDS['_faithfulness']})")
    assert score >= THRESHOLDS["_faithfulness"], \
        f"Faithfulness {score:.4f} below threshold {THRESHOLDS['_faithfulness']}"

def test_answer_relevancy(eval_scores):
    score = eval_scores["_answer_relevancy"]
    print(f"\nAnswer relevancy: {score:.4f} (threshold: {THRESHOLDS['_answer_relevancy']})")
    assert score >= THRESHOLDS["_answer_relevancy"], \
        f"Answer relevancy {score:.4f} below threshold {THRESHOLDS['_answer_relevancy']}"

def test_context_precision(eval_scores):
    score = eval_scores["_context_precision"]
    print(f"\nContext precision: {score:.4f} (threshold: {THRESHOLDS['_context_precision']})")
    assert score >= THRESHOLDS["_context_precision"], \
        f"Context precision {score:.4f} below threshold {THRESHOLDS['_context_precision']}"


def test_context_recall(eval_scores):
    score = eval_scores["_context_recall"]
    print(f"\nContext recall: {score:.4f} (threshold: {THRESHOLDS['_context_recall']})")
    assert score >= THRESHOLDS["_context_recall"], \
        f"Context recall {score:.4f} below threshold {THRESHOLDS['_context_recall']}"