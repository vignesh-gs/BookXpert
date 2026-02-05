# Token classification: core (full name parts), initials (single letter), merged initials (gs -> g, s).

from typing import List, NamedTuple
from dataclasses import dataclass


@dataclass
class ClassifiedTokens:
    """Classified tokens: core, initials, merged initials."""
    core_tokens: List[str]        # Full name parts (len > 1)
    initial_tokens: List[str]     # Single letter initials
    merged_initials: List[str]    # 2-letter possible initials (gs, bs)
    all_tokens: List[str]         # Original token list for reference
    
    @property
    def first_core(self) -> str:
        return self.core_tokens[0] if self.core_tokens else ""
    
    @property
    def remaining_core(self) -> List[str]:
        return self.core_tokens[1:] if len(self.core_tokens) > 1 else []
    
    @property
    def all_initials_expanded(self) -> List[str]:
        """Initials plus expanded merged (gs -> g, s)."""
        result = list(self.initial_tokens)
        for merged in self.merged_initials:
            result.extend(expand_merged_initial(merged))
        return result


def is_core_token(token: str) -> bool:
    """Length > 2 and mostly alphabetic (so gs/bs stay merged initials)."""
    if len(token) <= 2:
        return False
    # Must have at least one alpha and be mostly alpha
    alpha_count = sum(1 for c in token if c.isalpha())
    return alpha_count > 0 and alpha_count >= len(token) * 0.5


def is_initial_token(token: str) -> bool:
    return len(token) == 1 and token.isalpha()


def is_merged_initial(token: str) -> bool:
    """Two letters, could be two initials (gs -> G S)."""
    return len(token) == 2 and token.isalpha()


def expand_merged_initial(token: str) -> List[str]:
    """gs -> [g, s], bs -> [b, s]."""
    if len(token) != 2:
        return [token]
    return [token[0], token[1]]


def classify_tokens(tokens: List[str]) -> ClassifiedTokens:
    """Split tokens into core, initials, merged initials."""
    core = []
    initials = []
    merged = []
    
    for token in tokens:
        if is_initial_token(token):
            initials.append(token)
        elif is_merged_initial(token):
            merged.append(token)
        elif is_core_token(token):
            core.append(token)
        # Skip tokens that don't fit any category (rare edge cases)
    
    return ClassifiedTokens(
        core_tokens=core,
        initial_tokens=initials,
        merged_initials=merged,
        all_tokens=tokens
    )


def get_first_letters_of_cores(core_tokens: List[str]) -> List[str]:
    """
    Get first letter of each core token.
    
    Used for abbreviation matching: checking if an initial
    matches any core token's first letter.
    """
    return [t[0] for t in core_tokens if t]
