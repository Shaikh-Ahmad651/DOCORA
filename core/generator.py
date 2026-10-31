"""Answer generation: build the prompt from the chunks and call the LLM."""

from __future__ import annotations

from config.prompts import (
    DEFAULT_FORMAT,
    EXCERPT_SEPARATOR,
    EXCERPT_TEMPLATE,
    FORMAT_OPTIONS,
    NO_ANSWER_FALLBACK,
    SYSTEM_INSTRUCTION,
    USER_PROMPT_TEMPLATE,
)
from providers.registry import get_llm
from utils.helpers import validate_query


def generate(query: str, chunks: list[dict], format_name: str = DEFAULT_FORMAT) -> str:
    """Write an answer using only the given chunks.

    Input: the question, the chunks to use (best first), and a key of
        FORMAT_OPTIONS (an unknown name falls back to "Auto").
    Output: the answer text, with [Source: <filename>, Page <n>] citations.
        With no chunks the fixed fallback sentence is returned and the model
        is not called.
    Errors: ValueError for an empty query (or a missing API key);
        RuntimeError if the model returns nothing; ProviderError (including
        blocked responses) is passed through for the pipeline to handle.
    """
    query = validate_query(query)
    if not chunks:
        return NO_ANSWER_FALLBACK

    excerpt_block = EXCERPT_SEPARATOR.join(
        EXCERPT_TEMPLATE.format(
            i=i, filename=c["filename"], page_num=c["page_num"], text=c["text"]
        )
        for i, c in enumerate(chunks, start=1)
    )
    format_instruction = FORMAT_OPTIONS.get(format_name, FORMAT_OPTIONS[DEFAULT_FORMAT])
    prompt = USER_PROMPT_TEMPLATE.format(
        excerpt_block=excerpt_block,
        format_instruction=format_instruction,
        query=query,
    )

    answer = get_llm().generate(prompt, system=SYSTEM_INSTRUCTION)
    if not answer or not answer.strip():
        raise RuntimeError("The model returned an empty response.")
    return answer.strip()
