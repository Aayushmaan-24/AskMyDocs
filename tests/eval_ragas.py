"""
eval_ragas.py — LLM-as-judge evaluation pipeline (Groq-powered)
Metrics: faithfulness, answer_relevancy, context_precision, context_recall
No RAGAS dependency — pure Groq + our own scoring logic.
Run:  python -m tests.eval_ragas
CI:   pytest tests/eval_ragas.py -v
"""

import json
import os
import re
import pytest
from dotenv import load_dotenv
from groq import Groq
from src.pipeline import ask

load_dotenv()

GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
JUDGE_MODEL  = "llama-3.3-70b-versatile"
GOLDEN_QA_PATH = "tests/golden_qa.json"

THRESHOLDS = {
    "faithfulness":      0.70,
    "answer_relevancy":  0.70,
    "context_precision": 0.25,  # low with small corpus (<20 chunks) — rises with more PDFs
    "context_recall":    0.60,
}


# ── 1. LLM judge ──────────────────────────────────────────────────

def judge_score(prompt: str) -> float:
    """Ask Groq to score something 0.0-1.0. Returns float."""
    response = GROQ_CLIENT.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    raw = response.choices[0].message.content.strip()
    match = re.search(r"(\d+\.?\d*)", raw)
    if match:
        val = float(match.group(1))
        return min(val, 1.0) if val <= 1.0 else val / 10.0
    return 0.0


# ── 2. Individual metrics ─────────────────────────────────────────

def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """Are all claims in the answer supported by the context?"""
    context_text = "\n\n".join(contexts[:3])
    prompt = f"""Score how faithful this answer is to the given context.
A faithful answer only contains claims that are directly supported by the context.
Score: 0.0 (completely unfaithful) to 1.0 (completely faithful).
Respond with ONLY a single decimal number like 0.8

CONTEXT:
{context_text}

ANSWER:
{answer}

SCORE:"""
    return judge_score(prompt)


def score_answer_relevancy(question: str, answer: str) -> float:
    """Does the answer actually address the question asked?"""
    prompt = f"""Score how relevant this answer is to the question.
A relevant answer directly addresses what was asked.
Score: 0.0 (completely irrelevant) to 1.0 (perfectly relevant).
Respond with ONLY a single decimal number like 0.8

QUESTION: {question}
ANSWER: {answer}

SCORE:"""
    return judge_score(prompt)


def score_context_precision(question: str, contexts: list[str]) -> float:
    """Are the retrieved contexts actually useful for answering the question?"""
    context_text = "\n\n---\n\n".join(
        f"[CHUNK {i+1}]: {c[:300]}" for i, c in enumerate(contexts[:5])
    )
    prompt = f"""Score how precise the retrieved context chunks are for answering this question.
Precision = what fraction of retrieved chunks are actually relevant to the question.
Score: 0.0 (no chunks are relevant) to 1.0 (all chunks are relevant).
Respond with ONLY a single decimal number like 0.8

QUESTION: {question}

RETRIEVED CHUNKS:
{context_text}

SCORE:"""
    return judge_score(prompt)


def score_context_recall(answer: str, ground_truth: str, contexts: list[str]) -> float:
    """Does the context contain enough info to produce the ground truth answer?"""
    context_text = "\n\n".join(contexts[:3])
    prompt = f"""Score whether the context contains enough information to produce the ground truth answer.
Recall = how much of the ground truth can be derived from the context.
Score: 0.0 (context has nothing needed) to 1.0 (context has everything needed).
Respond with ONLY a single decimal number like 0.8

GROUND TRUTH ANSWER: {ground_truth}

CONTEXT:
{context_text}

SCORE:"""
    return judge_score(prompt)


# ── 3. Full evaluation run ────────────────────────────────────────

def run_evaluation(limit: int = None) -> dict:
    with open(GOLDEN_QA_PATH) as f:
        golden = json.load(f)

    if limit:
        golden = golden[:limit]

    all_scores = {
        "faithfulness":      [],
        "answer_relevancy":  [],
        "context_precision": [],
        "context_recall":    [],
    }

    print(f"\nEvaluating {len(golden)} questions...\n")
    for i, item in enumerate(golden):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i+1}/{len(golden)}] {q[:55]}...")

        result   = ask(q, top_k=10, top_n=5)
        answer   = result["answer"]
        contexts = [c["text"] for c in result["chunks"]]

        f_score  = score_faithfulness(answer, contexts)
        ar_score = score_answer_relevancy(q, answer)
        cp_score = score_context_precision(q, contexts)
        cr_score = score_context_recall(answer, gt, contexts)

        all_scores["faithfulness"].append(f_score)
        all_scores["answer_relevancy"].append(ar_score)
        all_scores["context_precision"].append(cp_score)
        all_scores["context_recall"].append(cr_score)

        print(f"         faith={f_score:.2f}  relevancy={ar_score:.2f}  "
              f"precision={cp_score:.2f}  recall={cr_score:.2f}")

    final = {
        k: round(sum(v) / len(v), 4)
        for k, v in all_scores.items()
    }
    return final


# ── 4. Report ─────────────────────────────────────────────────────

def print_report(scores: dict) -> None:
    print("\n" + "="*52)
    print("  EVAL REPORT — LLM-as-Judge (Groq)")
    print("="*52)
    all_passed = True
    for metric, score in scores.items():
        threshold = THRESHOLDS[metric]
        passed    = score >= threshold
        if not passed:
            all_passed = False
        status = "✓ PASS" if passed else "✗ FAIL"
        bar    = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:22s} {bar} {score:.4f}  {status}  (min: {threshold})")
    print("="*52)
    if all_passed:
        print("  ✓ All metrics passed — CI gate: ALLOW merge")
    else:
        failed = [m for m, s in scores.items() if s < THRESHOLDS[m]]
        print(f"  ✗ Failed: {', '.join(failed)} — CI gate: BLOCK merge")
    print("="*52)


def save_scores(scores: dict, path: str = "tests/last_eval_scores.json") -> None:
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n  Scores saved → {path}")


# ── 5. Pytest CI gate ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def eval_scores():
    limit = 5 if os.getenv("CI") else None
    return run_evaluation(limit=limit)

def test_faithfulness(eval_scores):
    s = eval_scores["faithfulness"]
    assert s >= THRESHOLDS["faithfulness"], \
        f"Faithfulness {s:.4f} < threshold {THRESHOLDS['faithfulness']}"

def test_answer_relevancy(eval_scores):
    s = eval_scores["answer_relevancy"]
    assert s >= THRESHOLDS["answer_relevancy"], \
        f"Answer relevancy {s:.4f} < threshold {THRESHOLDS['answer_relevancy']}"

def test_context_precision(eval_scores):
    s = eval_scores["context_precision"]
    assert s >= THRESHOLDS["context_precision"], \
        f"Context precision {s:.4f} < threshold {THRESHOLDS['context_precision']}"

def test_context_recall(eval_scores):
    s = eval_scores["context_recall"]
    assert s >= THRESHOLDS["context_recall"], \
        f"Context recall {s:.4f} < threshold {THRESHOLDS['context_recall']}"


# ── 6. Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    scores = run_evaluation()
    print_report(scores)
    save_scores(scores)
