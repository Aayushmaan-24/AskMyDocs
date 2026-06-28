"""
pipeline.py — Groq generation with citation enforcement
Every claim in the answer must map back to a source chunk.
"""

import re
import os
from groq import Groq
from dotenv import load_dotenv
from rich.console import Console
from src.retrieval import hybrid_retrieve

load_dotenv()
console = Console()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# ── 1. Prompt builder ──────────────────────────────────────────────

def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(f"[SOURCE {i+1}: {chunk['source']}, page {chunk['page']}]\n{chunk['text']}")
    context = "\n\n".join(context_blocks)
    return f"""You are a precise document assistant. Answer the questions ONLY based on the sources provided below
RULES:
1. Every sentence in your answer MUST end with a citation like [SOURCE 1] or [SOURCE 2].
2. If the answer is not in the sources, say exactly: "I cannot find this in the provided documents."
3. Do not add any information beyond what is in the sources.
4. Be concise and direct.

SOURCES:
{context}

QUESTION : {query}
ANSWER (cite every sentence) :

"""

# ── 2. Citation parser ─────────────────────────────────────────────

def parse_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Extract which sources were actually cited in the answer."""
    cited = []
    pattern = re.compile(r'\[SOURCE (\d+)\]')
    cited_nums = set(int(m) for m in pattern.findall(answer))
    for num in sorted(cited_nums):
        idx = num - 1
        if 0 <= idx < len(chunks):
            cited.append({
                "citation_num" : num,
                "source" : chunks[idx]["source"],
                "page" : chunks[idx]["page"],
                "ce_score" : chunks[idx].get("ce_score", 0),
                "text_preview" : chunks[idx]["text"][:120],
            })
    return cited


# ── 3. Citation validator ──────────────────────────────────────────

def validate_citation(answer: str, chunks: list[dict]) -> list[dict]:
    """
    Check every sentence has a citation.
    Returns a dict with pass/fail + details.
    """
    
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
    uncited = []
    for sentence in sentences:
        if not re.search(r'\[SOURCE \d+\]', sentence):
            uncited.append(sentence)
    
    return {
        "total_sentences" : len(sentence),
        "uncited_sentence" : uncited,
        "citation_rate" : round((len(sentence) - len(uncited)) / max(len(sentence), 1), 2),
        "passed" : len(uncited) == 0,
    }
    
# ── 4. Full RAG pipeline ───────────────────────────────────────────

def ask(query: str, top_k: int = 10, top_n: int = 5) -> dict:
    """
    End-to-end RAG:
    retrieve → build prompt → generate → validate citations
    Returns full result dict with answer + citations + validation.
    """
    
    # retrieve
    chunks = hybrid_retrieve(query, top_k=top_k, top_n=top_n)
    if not chunks:
        return {"answer": "No relevant documents found.", "citations": [], "validation": {}}
    
    # Generate
    prompt = build_prompt(query, chunks)
    response = client.chat.completions.create(
        model = MODEL,
        messages=[{
            "role" : "user",
            "content" : prompt,
        }],
        temperature=0.1,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content.strip()
    
    # parse + validate citations
    citations = parse_citations(answer, chunks)
    validation = validate_citation(answer, chunks)
    
    return {
        "query":      query,
        "answer":     answer,
        "citations":  citations,
        "chunks":     chunks,
        "validation": validation,
        "model":      MODEL,
        "usage": {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }