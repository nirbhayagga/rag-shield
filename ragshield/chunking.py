"""Paragraph-aware chunker that meets the eval_harness check_chunker contract: fn(text, chunk_size, overlap) -> list[str]."""
import re


def _pieces(text, size):
    """Split into paragraphs, then sentences, then words, so no piece exceeds size."""
    out = []
    for para in re.split(r'\n\s*\n', text):
        if len(para) <= size:
            out.append(para)
            continue
        for sentence in re.split(r'(?<=[.!?])\s+|\n', para):
            while len(sentence) > size:
                cut = sentence.rfind(' ', 0, size)
                cut = cut if cut > 0 else size
                out.append(sentence[:cut])
                sentence = sentence[cut:].lstrip()
            out.append(sentence)
    return [p for p in out if p.strip()]


def chunk(text, chunk_size=800, overlap=100):
    chunks, current = [], ''
    for piece in _pieces(text, chunk_size):
        joined = f'{current}\n\n{piece}' if current else piece
        if len(joined) <= chunk_size:
            current = joined
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ''
        space = tail.find(' ')
        tail = tail[space + 1:] if 0 <= space < len(tail) - 1 else ''
        current = f'{tail} {piece}'.strip() if tail and len(tail) + 1 + len(piece) <= chunk_size else piece
    if current:
        chunks.append(current)
    return chunks
