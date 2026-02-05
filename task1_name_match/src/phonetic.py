# Phonetic rewrite for Indian name transliteration (Geetha/Gita, Pooja/Puja, etc.). Rules applied in order; longer patterns first.

from typing import List
from src.config import PHONETIC_KEY_LENGTH


PHONETIC_RULES = [
    ('ee', 'i'),
    ('aa', 'a'),
    ('oo', 'u'),
    ('th', 't'),
    ('ph', 'f'),
    ('sh', 's'),
    ('tt', 't'),
    ('nn', 'n'),
    ('kk', 'k'),
    ('rr', 'r'),
    ('mm', 'm'),
    ('ll', 'l'),
]


def phonetic_rewrite(token: str) -> str:
    """Apply phonetic rules to a token (length > 1). e.g. geetha -> gita."""
    if len(token) <= 1:
        return token
    
    result = token.lower()
    
    for pattern, replacement in PHONETIC_RULES:
        result = result.replace(pattern, replacement)
    
    return result


def phonetic_key_for_index(first_core_token: str) -> str:
    """First N chars of phonetically rewritten first core token; padded if short. Used for index bucket."""
    if not first_core_token:
        return "___"
    
    rewritten = phonetic_rewrite(first_core_token)
    
    if len(rewritten) < PHONETIC_KEY_LENGTH:
        rewritten = rewritten + '_' * (PHONETIC_KEY_LENGTH - len(rewritten))
    
    return rewritten[:PHONETIC_KEY_LENGTH]


def phonetic_core_string(core_tokens: List[str]) -> str:
    """Join phonetically rewritten core tokens for sound comparison."""
    return ' '.join(phonetic_rewrite(t) for t in core_tokens)


def first_letter_key(first_core_token: str) -> str:
    """First letter of first core token; used when phonetic bucket is empty or small."""
    if not first_core_token:
        return "_"
    return first_core_token[0].lower()
