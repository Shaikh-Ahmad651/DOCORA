"""Own recursive character splitter. Chunks never span pages.

How it works:
1. Split the page text into small "atoms" using the first separator that
   occurs in it ("\\n\\n", then "\\n", then " ", then single characters).
   An atom that is still longer than the chunk size is split again with the
   next separator.
2. Merge atoms into chunks of at most CHUNK_SIZE characters, joining them
   with the separator they were originally split on.
3. When a chunk is full, the next one starts with a tail of up to
   CHUNK_OVERLAP characters from the full chunk (cut at a word boundary).
   That tail counts towards CHUNK_SIZE, so no chunk is ever longer than it.

Because atoms are re-joined with their original separators, every chunk is
an exact substring of its page text.
"""

from __future__ import annotations

import logging
import re

from config import settings

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s")


def chunk_pages(
    pages: list[dict],
    chunk_size: int | None = None,
    overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[dict]:
    """Split pages into chunks.

    Input: pages from the loader ({"text", "filename", "page_num", ...}).
        The optional arguments default to the values in config/settings.py
        (they exist so tests can use small sizes).
    Output: chunks {"text", "filename", "page_num", "chunk_id"} in page order.
        chunk_id is f"{filename}_p{page_num}_c{i}" with i counting from 0 on
        each page. Chunks are stripped; empty ones are dropped.
    Errors: ValueError if the sizes make no sense (overlap must be smaller
        than chunk_size). Empty input returns [] and logs a warning.
    """
    size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
    lap = settings.CHUNK_OVERLAP if overlap is None else overlap
    seps = list(settings.CHUNK_SEPARATORS if separators is None else separators)
    if size < 1 or lap < 0 or lap >= size:
        raise ValueError(
            f"Invalid chunk settings: size={size}, overlap={lap} "
            "(overlap must be smaller than size)."
        )
    if not seps or seps[-1] != "":
        seps.append("")  # the final separator splits by character

    if not pages:
        logger.warning("chunk_pages received no pages.")
        return []

    chunks: list[dict] = []
    for page in pages:
        atoms = _atoms(page["text"], seps, size, "")
        index = 0
        for text in _merge(atoms, size, lap):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "text": text,
                    "filename": page["filename"],
                    "page_num": page["page_num"],
                    "chunk_id": f"{page['filename']}_p{page['page_num']}_c{index}",
                }
            )
            index += 1
    return chunks


def _atoms(text: str, seps: list[str], size: int, lead: str) -> list[tuple[str, str]]:
    """Break text into (lead, atom) pairs, each atom at most ``size`` long.

    ``lead`` is the separator that came before the atom in the original text.
    Input: the text, the separators still available, the size limit, and the
    separator that preceded this text. Output: list of (lead, atom).
    """
    if len(text) <= size:
        return [(lead, text)]
    position = 0
    for position, sep in enumerate(seps):
        if sep == "" or sep in text:
            break
    rest = seps[position + 1 :]
    pieces = text.split(sep) if sep else list(text)
    atoms: list[tuple[str, str]] = []
    for i, piece in enumerate(pieces):
        atoms.extend(_atoms(piece, rest, size, lead if i == 0 else sep))
    return atoms


def _merge(atoms: list[tuple[str, str]], size: int, overlap: int) -> list[str]:
    """Join atoms into chunks of at most ``size`` characters, with overlap."""
    chunks: list[str] = []
    current = ""
    for lead, atom in atoms:
        joined = current + lead + atom if current else atom
        if len(joined) <= size:
            current = joined
            continue
        chunks.append(current)
        # Start the next chunk with a tail of this one, if it still fits.
        room = size - len(lead) - len(atom)
        carry = _tail(current, min(overlap, room))
        current = carry + lead + atom if carry else atom
    if current:
        chunks.append(current)
    return chunks


def _tail(text: str, n: int) -> str:
    """Last ``n`` characters of text, moved forward to a word boundary.

    Input: text and a length. Output: the tail (may be shorter than n),
    or "" if n is not positive.
    """
    if n <= 0:
        return ""
    tail = text[-n:]
    cut_mid_word = len(tail) < len(text) and not text[-n - 1].isspace()
    if cut_mid_word:
        match = _WHITESPACE.search(tail)
        if match and match.end() < len(tail):
            tail = tail[match.end() :]
    return tail
