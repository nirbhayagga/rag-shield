"""Facts an edit can quietly change: links, email addresses, phone numbers, IP addresses and amounts of money.

A harmful edit needs no instructions. Pointing a page's VPN setup link at another site, or changing the support address
or phone number, is enough, and the injection classifier sees nothing wrong with it. changes() reports values of one of
these kinds that an edit removed from a page (all of it, or one of its copies) and replaced with one the page didn't
have. A link counts as replaced only when the new one goes to a site the page didn't link to before, so a moved page on
the same site isn't reported."""
import re
from collections import Counter
from urllib.parse import urlsplit

URL = re.compile(r'https?://[^\s<>()\[\]{}"\'`|\\^]+', re.I)
EMAIL = re.compile(r'(?<![\w.%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b')
PHONE = re.compile(r'(?<![\w.+-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\) ?|\d{3}[ .-])\d{3}[ .-]\d{4}(?![\w-]|\.\d)')
IPV4 = re.compile(r'(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\w.]*\d)')
MONEY = re.compile(r'[$€£]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP)\b')
KINDS = {'link': URL, 'email address': EMAIL, 'phone number': PHONE, 'IP address': IPV4, 'amount': MONEY}


def _url(value):
    value = value.rstrip('.,;:!?*_')
    parts = urlsplit(value)
    host = (parts.hostname or '').lower().removeprefix('www.')
    return host + parts.path.rstrip('/') + (f'?{parts.query}' if parts.query else ''), host


def extract(text, counts=None):
    """{kind: {normalized value: as written}}. Links are normalized to host and path, phone numbers to their digits.
    With counts={}, also fills counts[kind] with how often each value appears."""
    out = {}
    for kind, pattern in KINDS.items():
        values, seen = {}, Counter()
        for match in pattern.finditer(text):
            written = match.group(0)
            if kind == 'link':
                key, host = _url(written)
                if not host:
                    continue
                written = written.rstrip('.,;:!?*_')
            elif kind == 'phone number':
                key = re.sub(r'\D', '', written)[-10:]
            elif kind == 'email address':
                key = written.lower()
            else:
                key = re.sub(r'[\s,]', '', written).upper()
            values.setdefault(key, written)
            seen[key] += 1
        out[kind] = values
        if counts is not None:
            counts[kind] = seen
    return out


def changes(previous, current, added):
    """[(kind, old value, new value)] for each kind that the edit replaced: a value of the previous text that the current
    text has fewer copies of, while the added text brings one the previous text didn't have. Counting copies matters: an
    attacker can change one of two links to the same page and leave the other."""
    had, has = {}, {}
    before, new = extract(previous, had), extract(added)
    extract(current, has)
    hosts = {_url(v)[1] for v in before['link'].values()}
    out = []
    for kind in KINDS:
        removed = [before[kind][k] for k in before[kind] if has[kind][k] < had[kind][k]]
        if kind == 'link':
            brought = [v for k, v in new[kind].items() if k not in before[kind] and _url(v)[1] not in hosts]
        else:
            brought = [v for k, v in new[kind].items() if k not in before[kind]]
        if removed and brought:
            out.append((kind, removed[0], brought[0]))
    return out


def describe(kind, old, new):
    if kind == 'link':
        return f"points a link to a site the page didn't link to before ({old} → {new})"
    return f'replaces {"an" if kind[0] in "aeiouIA" else "a"} {kind} the page already had ({old} → {new})'
