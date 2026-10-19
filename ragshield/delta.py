"""What did an edit add? Paragraph-level diff against the last signed version, refined to sentences for modified paragraphs.

A compromised trusted page is, almost by definition, an edit. Scoring only the added text avoids the dilution that hides
a short injected passage inside a long legitimate chunk."""
import difflib
import re

from .provenance import sha256

SENTENCE = re.compile(r'(?<=[.!?])\s+|\n')


def paragraphs(text):
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def sentences(paragraph):
    return [s.strip() for s in SENTENCE.split(paragraph) if s.strip()]


def additions(previous, current):
    """Text present in current but not in previous: whole new paragraphs, or the new sentences of a modified one."""
    known = {sha256(p) for p in previous}
    out = []
    for para in current:
        if sha256(para) in known:
            continue
        best = max(previous, key=lambda old: difflib.SequenceMatcher(None, old, para, autojunk=False).quick_ratio(), default=None)
        if best is not None and difflib.SequenceMatcher(None, best, para, autojunk=False).ratio() > 0.5:
            old_sentences = set(sentences(best))
            added = [s for s in sentences(para) if s not in old_sentences]
            if added:
                out.append(' '.join(added))
        else:
            out.append(para)
    return out


def contains(chunk_text, added_text):
    """Does this chunk carry (part of) the added text? Chunks can split a passage, so match on a leading or trailing window."""
    flat = ' '.join(chunk_text.split())
    flat_added = ' '.join(added_text.split())
    head, tail = flat_added[:60], flat_added[-60:]
    return head in flat or tail in flat
