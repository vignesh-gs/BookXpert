# Text normalization for name matching. Punctuation -> spaces so "Vignesh.G.S" -> "vignesh g s" (keeps initials).

import re
from typing import List, Optional


PUNCT_PATTERN = re.compile(r'[.,;:()\[\]{}\-_/\'"]+')


def normalize_text(s: str) -> str:
    """Lowercase, replace punctuation with spaces, collapse spaces, strip."""
    if not s:
        return ""
    
    text = s.lower()
    
    text = PUNCT_PATTERN.sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def tokenize(normalized: str) -> List[str]:
    """Split normalized string into tokens."""
    if not normalized:
        return []
    return normalized.split()


def is_valid_name(normalized: str) -> bool:
    """True if non-empty and has at least one letter."""
    if not normalized:
        return False
    return any(c.isalpha() for c in normalized)


def normalize_and_tokenize(raw: str) -> tuple[str, List[str]]:
    """Normalize and tokenize in one call. Returns (normalized_string, tokens)."""
    normalized = normalize_text(raw)
    tokens = tokenize(normalized)
    return normalized, tokens
