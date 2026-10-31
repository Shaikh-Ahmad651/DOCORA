"""In-memory session index: vectors (cosine similarity) plus BM25 keywords.

Everything lives in RAM and disappears when the app stops. One module-level
index is shared by the app; ``pipeline`` is its only caller. All reads and
writes take a lock.

Note: BM25 here is ``rank_bm25.BM25Okapi``. Its IDF formula gives zero or
negative scores to words that appear in half or more of the chunks, so on a
tiny document (a few chunks) keyword search may return nothing and the vector
search does all the work. That is harmless on real documents.
"""

from __future__ import annotations

import re
import threading

import numpy as np
from rank_bm25 import BM25Okapi

# Keeps tokens like "4.2", "q3-2023", "don't"; drops trailing punctuation
# ("revenue," -> "revenue"). Untested starting point: tune after a real run.
_TOKEN_RE = re.compile(r"\w+(?:[.,'’\-]\w+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case a text and split it into BM25 tokens.

    Input: any text. Output: list of tokens. Errors: none.
    The same function is used for chunks and queries.
    """
    return _TOKEN_RE.findall(text.lower())


class SessionIndex:
    """Chunks, their normalized vectors, a BM25 index, and a file table."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reset()

    def _reset(self) -> None:
        self._chunks: list[dict] = []
        self._matrix: np.ndarray | None = None  # N x D, float32, rows L2-normalized
        self._dim: int | None = None
        self._bm25: BM25Okapi | None = None
        self._files: dict[str, dict] = {}  # filename -> {"hash", "pages", "chunk_ids"}

    # ------------------------------------------------------------------ writes
    def add_file(
        self,
        filename: str,
        file_hash: str,
        chunks: list[dict],
        embeddings: list[list[float]],
        pages: int | None = None,
    ) -> None:
        """Add a file's chunks, replacing any earlier file with the same name.

        Input: filename, its content hash, the chunks, one embedding per chunk,
            and optionally the file's page count (default: number of distinct
            pages that have chunks).
        Output: none.
        Errors: ValueError if there are no chunks, the counts differ, the
            vectors are malformed, or their size differs from the vectors
            already in the index. On error nothing is changed.
        """
        if not chunks:
            raise ValueError("Cannot add a file with no chunks.")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(embeddings)} embeddings."
            )
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise ValueError("Embeddings must be a list of equal-length, non-empty vectors.")

        with self._lock:
            own_chunks = len(self._files[filename]["chunk_ids"]) if filename in self._files else 0
            others_exist = len(self._chunks) > own_chunks
            if others_exist and vectors.shape[1] != self._dim:
                raise ValueError(
                    f"Embedding size mismatch: the index holds {self._dim}-number vectors "
                    f"but this file has {vectors.shape[1]}. Did the embedding provider or "
                    "model change? Reset the session and index again."
                )

            self._remove_locked(filename)  # replace semantics

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # a zero vector stays zero instead of dividing by 0
            vectors = vectors / norms

            self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])
            self._dim = vectors.shape[1]
            self._chunks.extend(dict(chunk) for chunk in chunks)
            self._files[filename] = {
                "hash": file_hash,
                "pages": pages if pages is not None else len({c["page_num"] for c in chunks}),
                "chunk_ids": [c["chunk_id"] for c in chunks],
            }
            self._rebuild_bm25()

    def remove_file(self, filename: str) -> None:
        """Remove a file and its chunks. Unknown names are ignored. Errors: none."""
        with self._lock:
            self._remove_locked(filename)

    def clear(self) -> None:
        """Remove everything. Errors: none."""
        with self._lock:
            self._reset()

    def _remove_locked(self, filename: str) -> None:
        info = self._files.pop(filename, None)
        if info is None:
            return
        gone = set(info["chunk_ids"])
        keep = np.array([c["chunk_id"] not in gone for c in self._chunks], dtype=bool)
        self._chunks = [c for c, k in zip(self._chunks, keep) if k]
        if self._chunks:
            self._matrix = self._matrix[keep]
        else:
            self._matrix = None
            self._dim = None  # an empty index may start over with any vector size
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        tokens = [tokenize(c["text"]) for c in self._chunks]
        # BM25 cannot be built from nothing, or from chunks with no words at all.
        self._bm25 = BM25Okapi(tokens) if any(tokens) else None

    # ------------------------------------------------------------------- reads
    def is_empty(self) -> bool:
        """True if no chunks are indexed. Errors: none."""
        with self._lock:
            return not self._chunks

    def has_file(self, filename: str, file_hash: str) -> bool:
        """True if this filename is indexed with exactly this hash. Errors: none."""
        with self._lock:
            info = self._files.get(filename)
            return info is not None and info["hash"] == file_hash

    def file_hash(self, filename: str) -> str | None:
        """The stored hash for a filename, or None if it is not indexed."""
        with self._lock:
            info = self._files.get(filename)
            return None if info is None else info["hash"]

    def stats(self) -> dict:
        """Counts for the UI.

        Output: {"files": int, "chunks": int,
                 "per_file": {filename: {"pages": int, "chunks": int}}}.
        Errors: none.
        """
        with self._lock:
            return {
                "files": len(self._files),
                "chunks": len(self._chunks),
                "per_file": {
                    name: {"pages": info["pages"], "chunks": len(info["chunk_ids"])}
                    for name, info in self._files.items()
                },
            }

    def vector_search(self, query_vec: list[float], k: int) -> list[dict]:
        """Top-k chunks by cosine similarity.

        Input: the query vector and k.
        Output: chunk dicts plus "score" (cosine similarity), best first.
            An empty index, or a query vector of all zeros, gives [].
        Errors: ValueError if the query vector's size differs from the index's.
        """
        with self._lock:
            if self._matrix is None or k <= 0:
                return []
            query = np.asarray(query_vec, dtype=np.float32)
            if query.ndim != 1 or query.shape[0] != self._dim:
                raise ValueError(
                    f"Query vector has size {query.shape[-1] if query.ndim else 0} but the "
                    f"index uses {self._dim}. Did the embedding provider or model change?"
                )
            norm = np.linalg.norm(query)
            if norm == 0:
                return []
            scores = self._matrix @ (query / norm)
            k = min(k, len(scores))
            if k < len(scores):
                top = np.argpartition(-scores, k - 1)[:k]
            else:
                top = np.arange(len(scores))
            top = top[np.argsort(-scores[top])]
            return [{**self._chunks[i], "score": float(scores[i])} for i in top]

    def bm25_search(self, query: str, k: int) -> list[dict]:
        """Top-k chunks by BM25 keyword score, only those with score > 0.

        Input: the query text and k.
        Output: chunk dicts plus "score", best first. May be shorter than k,
            or empty (no query words, empty index, or no matches).
        Errors: none.
        """
        with self._lock:
            tokens = tokenize(query)
            if self._bm25 is None or not tokens or k <= 0:
                return []
            scores = np.asarray(self._bm25.get_scores(tokens))
            positive = np.nonzero(scores > 0)[0]
            order = positive[np.argsort(-scores[positive])][:k]
            return [{**self._chunks[i], "score": float(scores[i])} for i in order]


# The one shared index, and thin functions around it (pipeline uses these).
_session = SessionIndex()


def add_file(filename, file_hash, chunks, embeddings, pages=None) -> None:
    """See SessionIndex.add_file."""
    _session.add_file(filename, file_hash, chunks, embeddings, pages)


def remove_file(filename: str) -> None:
    """See SessionIndex.remove_file."""
    _session.remove_file(filename)


def clear() -> None:
    """See SessionIndex.clear."""
    _session.clear()


def is_empty() -> bool:
    """See SessionIndex.is_empty."""
    return _session.is_empty()


def has_file(filename: str, file_hash: str) -> bool:
    """See SessionIndex.has_file."""
    return _session.has_file(filename, file_hash)


def file_hash(filename: str) -> str | None:
    """See SessionIndex.file_hash."""
    return _session.file_hash(filename)


def stats() -> dict:
    """See SessionIndex.stats."""
    return _session.stats()


def vector_search(query_vec: list[float], k: int) -> list[dict]:
    """See SessionIndex.vector_search."""
    return _session.vector_search(query_vec, k)


def bm25_search(query: str, k: int) -> list[dict]:
    """See SessionIndex.bm25_search."""
    return _session.bm25_search(query, k)
