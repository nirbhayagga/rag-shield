"""Hidden and look-alike characters: text that reads one way to a person and another way to a model.

Invisible format characters (zero-width spaces, word joiners, bidi controls) split words into odd tokens, Unicode tag
characters can carry a whole instruction that no editor shows, and letters from other scripts or styled Unicode blocks
can stand in for Latin ones. Injection classifiers react to these unpredictably: on BIPIA's test attacks the base
classifier flagged 10% of the plain text, but 100% with zero-width spaces and 90% with Cyrillic look-alikes. A score that
moves that much is a false-alarm risk on benign pages, and an evasion risk for any classifier that learns to ignore them.

So the check is explicit. inspect() returns the text a reader would see (invisible characters removed, tag characters
decoded, look-alikes mapped back to Latin) and counts what it changed. The shield scores the cleaned text and reports the
counts as their own reason. Ordinary multilingual text is left alone: a Russian or Persian word, an emoji sequence, a
right-to-left paragraph with its direction marks, or fullwidth letters in Japanese text is not counted.
"""
import re
import unicodedata
from functools import lru_cache

# Letters from Cyrillic, Greek and Armenian that render like Latin letters (a subset of Unicode's confusables data).
LOOKALIKE = {
    # Cyrillic
    # (small-capital look-alikes such as в, м, т are left out: they are common in Russian words and only resemble B/M/T)
    'а': 'a', 'с': 'c', 'ԁ': 'd', 'е': 'e', 'һ': 'h', 'і': 'i', 'ј': 'j', 'ӏ': 'l', 'о': 'o', 'р': 'p', 'ԛ': 'q',
    'ѕ': 's', 'ԝ': 'w', 'х': 'x', 'у': 'y', 'ү': 'y',
    'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H', 'І': 'I', 'Ј': 'J', 'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'P',
    'Ѕ': 'S', 'Т': 'T', 'Х': 'X', 'У': 'Y',
    # Greek
    'α': 'a', 'γ': 'y', 'ι': 'i', 'κ': 'k', 'ν': 'v', 'ο': 'o', 'ρ': 'p', 'υ': 'u', 'χ': 'x',
    'Α': 'A', 'Β': 'B', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'H', 'Ι': 'I', 'Κ': 'K', 'Μ': 'M', 'Ν': 'N', 'Ο': 'O', 'Ρ': 'P',
    'Τ': 'T', 'Υ': 'Y', 'Χ': 'X',
    # Armenian
    'հ': 'h', 'ո': 'n', 'ս': 'u', 'օ': 'o',
}
BIDI = {0x061C, 0x200E, 0x200F} | set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A))
JOINERS = {0x200C, 0x200D}  # legitimate inside emoji sequences and many non-Latin scripts
SOFT_HYPHEN, BOM, BLACK_FLAG = 0x00AD, 0xFEFF, 0x1F3F4
FILLERS = {0x115F, 0x1160, 0x3164, 0xFFA0, 0x2800}  # render as blank space but are not whitespace
LETTERS = re.compile(r'[^\W\d_]+')
COUNTS = ('invisible', 'bidi', 'tags', 'lookalike', 'styled')


@lru_cache(maxsize=4096)
def _latin(ch):
    """A Latin letter, including styled forms (fullwidth, mathematical, circled) that a reader takes for one."""
    if ch.isascii():
        return ch.isalpha()
    cp = ord(ch)
    if 0x1D400 <= cp <= 0x1D7FF or 0x24B6 <= cp <= 0x24E9:
        return True
    return ch.isalpha() and 'LATIN' in unicodedata.name(ch, '')


@lru_cache(maxsize=4096)
def _rtl(ch):
    return unicodedata.bidirectional(ch) in ('R', 'AL')


def _styled(ch):
    """Fullwidth, mathematical or circled letters and digits, mapped to ASCII; None for anything else."""
    cp = ord(ch)
    if 0xFF10 <= cp <= 0xFF19 or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
        return chr(cp - 0xFEE0)
    if 0x1D400 <= cp <= 0x1D7FF or 0x24B6 <= cp <= 0x24E9:
        mapped = unicodedata.normalize('NFKC', ch)
        return mapped if mapped.isascii() else None
    return None


def _joiner_is_legit(text, i):
    """A ZWJ or ZWNJ between emoji, or next to a letter of a script that uses joiners (Arabic, Indic, ...)."""
    for j in (i - 1, i + 1):
        if 0 <= j < len(text):
            ch = text[j]
            if unicodedata.category(ch) in ('So', 'Sk', 'Mn') or 0x1F000 <= ord(ch) <= 0x1FAFF:
                return True
            if ch.isalpha() and not _latin(ch):
                return True
    return False


def _inside_word(text, i):
    before = next((c for c in reversed(text[:i]) if unicodedata.category(c) != 'Cf'), '')
    after = next((c for c in text[i + 1:] if unicodedata.category(c) != 'Cf'), '')
    return before.isalnum() and after.isalnum()


def inspect(text):
    """Return (clean_text, counts). clean_text is what a reader sees; counts says what was hidden or disguised.
    Plain ASCII text is returned unchanged, so existing scores and caches are unaffected."""
    counts = dict.fromkeys(COUNTS, 0)
    if text.isascii():
        return text, counts
    letters = [ch for ch in text if ch.isalpha()]
    latin = sum(map(_latin, letters))
    latin_share = latin / len(letters) if letters else 0.0
    # Latin text in disguise has Latin letters and look-alikes, but almost no other foreign letters (Russian has и, б, д...)
    foreign = sum(1 for ch in letters if not _latin(ch) and ch not in LOOKALIKE)
    latin_dominant = latin > 0 and foreign / (latin + foreign) < 0.2
    has_rtl = any(_rtl(ch) for ch in letters)

    out, in_flag = [], False
    for i, ch in enumerate(text):
        cp = ord(ch)
        if cp == BLACK_FLAG:
            in_flag = True
            out.append(ch)
            continue
        if 0xE0000 <= cp <= 0xE007F:  # tag characters: invisible, but models read them as ASCII
            if in_flag:  # subdivision flag emoji (England, Scotland, Wales) are legitimate tag sequences
                out.append(ch)
                continue
            counts['tags'] += 1
            if 0xE0020 <= cp <= 0xE007E:
                out.append(chr(cp - 0xE0000))
            continue
        in_flag = False
        if cp in BIDI:
            if not has_rtl:
                counts['bidi'] += 1
            continue
        if cp == SOFT_HYPHEN or (cp == BOM and i == 0):
            continue
        if cp in JOINERS:
            if _joiner_is_legit(text, i):
                out.append(ch)
            else:
                counts['invisible'] += 1
            continue
        if 0xE0100 <= cp <= 0xE01EF:  # variation selectors 17-256: only CJK ideographic variants use them
            prev = text[i - 1] if i else ''
            if prev and 'CJK' in unicodedata.name(prev, ''):
                out.append(ch)
            else:
                counts['invisible'] += 1
            continue
        if cp in FILLERS:
            if latin_share > 0.5:
                counts['invisible'] += 1
                out.append(' ')
            else:
                out.append(ch)
            continue
        if unicodedata.category(ch) == 'Cf':
            # Only inside a word does an invisible character change what a model reads. Between words it is usually
            # copy-paste debris (a stray zero-width space after a space), so it is removed but not counted.
            if _inside_word(text, i):
                counts['invisible'] += 1
            continue
        mapped = _styled(ch)
        if mapped is not None:
            if mapped.isalpha() and latin_share > 0.5:
                counts['styled'] += 1
            out.append(mapped)
            continue
        out.append(ch)
    clean = ''.join(out)

    if any(ch in LOOKALIKE for ch in clean):
        def fix(match):
            word = match.group(0)
            hits = sum(ch in LOOKALIKE for ch in word)
            if not hits:
                return word
            mixed = any(_latin(ch) for ch in word)
            disguised = latin_dominant and hits == len(word) >= 3  # a whole look-alike word inside Latin text
            if not (mixed or disguised):
                return word
            counts['lookalike'] += hits
            return ''.join(LOOKALIKE.get(ch, ch) for ch in word)
        clean = LETTERS.sub(fix, clean)
        if counts['lookalike'] and latin_dominant:  # the text is disguised: map the short all-look-alike words too
            def fix_short(match):
                word = match.group(0)
                if all(ch in LOOKALIKE for ch in word):
                    counts['lookalike'] += len(word)
                    return ''.join(LOOKALIKE[ch] for ch in word)
                return word
            clean = LETTERS.sub(fix_short, clean)
    return clean, counts


def clean(text):
    return inspect(text)[0]


def total(counts):
    return sum(counts.values()) if counts else 0


def describe(counts):
    """Short reason text, e.g. 'hidden or look-alike characters (12 invisible, 3 look-alike)'."""
    labels = {'invisible': 'invisible', 'bidi': 'direction control', 'tags': 'tag', 'lookalike': 'look-alike',
              'styled': 'styled'}
    parts = [f'{n} {labels[k]}' for k, n in counts.items() if n]
    return 'hidden or look-alike characters (' + ', '.join(parts) + ')'
