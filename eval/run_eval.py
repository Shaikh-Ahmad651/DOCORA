"""Compare retrieval quality with and without the reranker.

Usage (from the project root, with your .env filled in):

    python eval/run_eval.py
    python eval/run_eval.py --docs eval/sample_docs --questions eval/questions.json

What it does:
1. Indexes every PDF in the docs folder through core.pipeline.
2. For each question, retrieves candidates once (hybrid search + RRF), then
   builds two final lists of TOP_K_FINAL chunks: the first ones in RRF order,
   and the reranker's choice.
3. Scores each list: a "hit" means any expected (filename, page) is in it.

Metrics: Hit@1, Hit@5, MRR (no LLM judge, so the cost is only embeddings and
reranking). Also prints the average rerank time.

Output: a Markdown table to the screen and to eval/results.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import settings  # noqa: E402
from core import pipeline, reranker, retriever  # noqa: E402
from providers.base import ProviderError  # noqa: E402


def first_hit_rank(chunks: list[dict], expected: set[tuple[str, int]]) -> int | None:
    """1-indexed position of the first chunk matching an expected (filename, page), or None."""
    for rank, chunk in enumerate(chunks, start=1):
        if (chunk["filename"], chunk["page_num"]) in expected:
            return rank
    return None


def summarize(ranks: list[int | None]) -> dict:
    """Hit@1, Hit@5 and MRR from a list of first-hit ranks (None = miss)."""
    n = len(ranks)
    return {
        "hit1": sum(1 for r in ranks if r == 1) / n,
        "hit5": sum(1 for r in ranks if r is not None and r <= 5) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def load_questions(path: str) -> list[dict]:
    """Read and validate the questions file. Errors: SystemExit with a clear message."""
    if not os.path.exists(path):
        sys.exit(
            f"Questions file not found: {path}\n"
            "Copy eval/questions.example.json to eval/questions.json and edit it."
        )
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    questions = []
    for i, item in enumerate(data, start=1):
        expected = {(e["filename"], int(e["page"])) for e in item.get("expected", [])}
        if not item.get("question") or not expected:
            print(f"Skipping question #{i}: it needs a 'question' and at least one 'expected' page.")
            continue
        questions.append({"question": item["question"], "expected": expected})
    if not questions:
        sys.exit("No valid questions found.")
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Docora retrieval eval (with vs without reranker)")
    parser.add_argument("--docs", default=os.path.join(ROOT, "eval", "sample_docs"))
    parser.add_argument("--questions", default=os.path.join(ROOT, "eval", "questions.json"))
    parser.add_argument("--out", default=os.path.join(ROOT, "eval", "results.md"))
    args = parser.parse_args()

    pdfs = sorted(glob.glob(os.path.join(args.docs, "*.pdf")))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.docs}. Add your sample PDF(s) there.")
    questions = load_questions(args.questions)

    problems = settings.check_config()
    if problems:
        sys.exit("Fix your .env first:\n- " + "\n- ".join(problems))

    print(f"Indexing {len(pdfs)} PDF(s)...")
    result = pipeline.index_files(pdfs, ocr_enabled=False)
    for name, reason in result.files_failed.items():
        print(f"  FAILED {name}: {reason}")
    if not pipeline.list_indexed():
        sys.exit("Nothing was indexed, so there is nothing to evaluate.")

    rrf_ranks, rerank_ranks, rerank_times = [], [], []
    fallbacks = 0
    for item in questions:
        candidates = retriever.retrieve(item["question"])
        rrf_final = candidates[: settings.TOP_K_FINAL]
        rrf_ranks.append(first_hit_rank(rrf_final, item["expected"]))

        started = time.perf_counter()
        try:
            reranked, warning = reranker.rerank(item["question"], candidates, settings.TOP_K_FINAL)
        except ProviderError as exc:  # rerank() should not raise; be safe
            reranked, warning = rrf_final, str(exc)
        rerank_times.append(time.perf_counter() - started)
        if warning:
            fallbacks += 1
            print(f"  Reranker warning: {warning}")
        rerank_ranks.append(first_hit_rank(reranked, item["expected"]))

    a, b = summarize(rrf_ranks), summarize(rerank_ranks)
    avg_time = sum(rerank_times) / len(rerank_times)
    lines = [
        f"# Docora eval ({len(questions)} questions, final list = top {settings.TOP_K_FINAL})",
        "",
        "| Setup | Hit@1 | Hit@5 | MRR |",
        "|---|---|---|---|",
        f"| RRF only (no reranker) | {a['hit1']:.2f} | {a['hit5']:.2f} | {a['mrr']:.3f} |",
        f"| With reranker ({settings.RERANK_PROVIDER}) | {b['hit1']:.2f} | {b['hit5']:.2f} | {b['mrr']:.3f} |",
        "",
        f"Average rerank time: {avg_time:.2f}s per question.",
    ]
    if fallbacks:
        lines.append(
            f"\n**Warning:** the reranker failed on {fallbacks} question(s); those used RRF order, "
            "so the 'with reranker' row is not a clean result."
        )
    report = "\n".join(lines)
    print("\n" + report)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(report + "\n")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
