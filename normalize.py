"""Format normalisation applied to BOTH responses before judging (our fix).

The paper (arXiv 2604.23178, Sec. 5.1) recommends normalising formatting before judging
but never tests it. Two variants:

  strip       remove all markdown syntax: headers, bold/italic, bullet markers,
              horizontal rules, table pipes, blockquotes, and code fences.
  keep_code   the same, but fenced code blocks are left exactly as written, because
              for coding questions the fence marks real content boundaries.

Text, line breaks, numbers in numbered lists, and LaTeX ($...$) are kept: only
presentation syntax is removed, never words.
"""
import re

FENCE = re.compile(r"```.*?```", re.S)

_RULES = [
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$", re.M), r"\1"),     # # Header -> Header
    (re.compile(r"^[ \t]{0,3}(?:-{3,}|\*{3,}|_{3,})[ \t]*$", re.M), ""),       # --- rules
    (re.compile(r"^[ \t]{0,3}>[ \t]?", re.M), ""),                             # > quote
    (re.compile(r"^([ \t]*)[-*+][ \t]+", re.M), r"\1"),                        # - bullet
    (re.compile(r"\*\*(.+?)\*\*", re.S), r"\1"),                         # **bold**
    (re.compile(r"__(.+?)__", re.S), r"\1"),                             # __bold__
    (re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])"), r"\1"),  # *italic*
    (re.compile(r"(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?![\w_])"), r"\1"),    # _italic_
    (re.compile(r"`([^`\n]+)`"), r"\1"),                                 # `inline code`
    (re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$", re.M), ""),  # table rule
    (re.compile(r"^[ \t]*\|(.*)\|[ \t]*$", re.M), lambda m: " ; ".join(c.strip() for c in m.group(1).split("|"))),
]


def _strip_block(text):
    for pat, rep in _RULES:
        text = pat.sub(rep, text)
    text = text.replace("**", "")                 # unpaired bold markers
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse blank lines left behind
    return text.strip()


def strip(text):
    """Remove all markdown, including code fences (the code text itself is kept)."""
    text = re.sub(r"```[^\n]*\n?", "", text)      # drop fence lines, keep code
    return _strip_block(text)


def keep_code(text):
    """Remove markdown outside fenced code blocks; leave code blocks untouched."""
    out, last = [], 0
    for m in FENCE.finditer(text):
        out.append(_strip_block(text[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_strip_block(text[last:]))
    return "\n\n".join(p for p in out if p)


def identity(text):
    return text


NORMALISERS = {"none": identity, "strip": strip, "keep_code": keep_code}


def markdown_features(text):
    """Which markdown constructs remain (for checking the normaliser)."""
    pats = {"header": r"^#{1,6} ", "bold": r"\*\*", "bullet": r"^[ \t]*[-*+] ",
            "fence": r"```", "table": r"^\|"}
    return [k for k, p in pats.items() if re.search(p, text, re.M)]


def has_markdown(text):
    """Heuristic used for the 'formatting differs' subset of MT-Bench."""
    return bool(re.search(r"(^#{1,6} |\*\*|^\s*[-*+] |^\s*\d+\. |```|^\|)", text, re.M))
