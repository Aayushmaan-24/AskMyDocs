"""
pipeline.py — Groq generation with citation enforcement
Every claim in the answer must map back to a source chunk.
"""

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
    import re
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