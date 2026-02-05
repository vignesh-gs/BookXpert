# Multi-component scoring: first_name, edit_distance, other_core, initials, phonetic, full_string; minus penalties (missing/extra initials, missing cores, length).

from typing import Dict, List, Any
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from src.phonetic import phonetic_rewrite, phonetic_core_string
from src.initials import ClassifiedTokens, get_first_letters_of_cores
from src.config import (
    WEIGHT_FIRST_NAME, WEIGHT_OTHER_CORE, WEIGHT_INITIALS,
    WEIGHT_PHONETIC_CORE, WEIGHT_FULL_STRING, WEIGHT_EDIT_DISTANCE,
    PENALTY_MISSING_INITIAL, PENALTY_EXTRA_INITIAL,
    PENALTY_MISSING_CORE, PENALTY_OVERLONG_CANDIDATE, PENALTY_LENGTH_DIFF,
    FIRST_NAME_STRONG_THRESHOLD, MISSING_INITIAL_PENALTY_CAP_WHEN_STRONG,
    EXTRA_INITIAL_PENALTY_CAP_WHEN_STRONG, LENGTH_DIFF_PENALTY_CAP_WHEN_STRONG,
    ABBREV_MATCH_CAP
)


@dataclass
class ScoreBreakdown:
    """All score components and penalties for one candidate."""
    first_name_score: float
    edit_distance_score: float
    other_core_score: float
    initials_score: float
    phonetic_core_score: float
    full_string_score: float
    
    missing_initial_penalty: float
    extra_initial_penalty: float
    missing_core_penalty: float
    overlong_penalty: float
    length_diff_penalty: float
    
    final_score: float
    
    # Debug info
    query_normalized: str
    candidate_normalized: str
    query_cores: List[str]
    candidate_cores: List[str]
    query_initials: List[str]
    candidate_initials: List[str]
    missing_initials: List[str]
    extra_initials: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict for JSON."""
        return {
            'first_name_score': round(self.first_name_score, 1),
            'edit_distance_score': round(self.edit_distance_score, 1),
            'other_core_score': round(self.other_core_score, 1),
            'initials_score': round(self.initials_score, 1),
            'phonetic_core_score': round(self.phonetic_core_score, 1),
            'full_string_score': round(self.full_string_score, 1),
            'missing_initial_penalty': round(self.missing_initial_penalty, 1),
            'extra_initial_penalty': round(self.extra_initial_penalty, 1),
            'missing_core_penalty': round(self.missing_core_penalty, 1),
            'overlong_penalty': round(self.overlong_penalty, 1),
            'length_diff_penalty': round(self.length_diff_penalty, 1),
            'final_score': round(self.final_score, 1),
            'query_cores': self.query_cores,
            'candidate_cores': self.candidate_cores,
            'query_initials': self.query_initials,
            'candidate_initials': self.candidate_initials,
            'missing_initials': self.missing_initials,
            'extra_initials': self.extra_initials,
        }


def compute_first_name_score(q_first: str, c_first: str) -> float:
    """First-name match: raw WRatio + phonetic WRatio blend."""
    if not q_first or not c_first:
        return 0.0
    
    raw_score = fuzz.WRatio(q_first, c_first)
    
    # Phonetic similarity (handles transliteration variants)
    phonetic_score = fuzz.WRatio(
        phonetic_rewrite(q_first),
        phonetic_rewrite(c_first)
    )
    
    # Combine: favor raw match but boost phonetic matches
    return 0.7 * raw_score + 0.3 * phonetic_score


def compute_other_core_score(
    q_remaining: List[str],
    c_remaining: List[str],
    c_initials: List[str]
) -> float:
    """Match non-first cores; best match per query core; K can match Kumar."""
    if not q_remaining:
        return 50.0
    
    if not c_remaining:
        # partial match if query cores' first letters appear in candidate initials
        matches = 0
        for q_core in q_remaining:
            first_letter = q_core[0].lower()
            if first_letter in c_initials:
                matches += 1
        if matches > 0:
            return ABBREV_MATCH_CAP * (matches / len(q_remaining))
        return 0.0
    
    # Match each query core to best candidate core
    scores = []
    c_first_letters = get_first_letters_of_cores(c_remaining)
    
    for q_core in q_remaining:
        # Find best matching candidate core
        best_score = 0.0
        for c_core in c_remaining:
            score = fuzz.WRatio(q_core, c_core)
            # Also try phonetic
            phon_score = fuzz.WRatio(
                phonetic_rewrite(q_core),
                phonetic_rewrite(c_core)
            )
            best_score = max(best_score, score, phon_score)
        
        # Abbreviation fallback: if query core is short, check initials
        q_first_letter = q_core[0].lower()
        if best_score < ABBREV_MATCH_CAP and q_first_letter in c_initials:
            best_score = max(best_score, ABBREV_MATCH_CAP)
        
        scores.append(best_score)
    
    return sum(scores) / len(scores) if scores else 0.0


def compute_initials_score(
    q_initials: List[str],
    c_initials: List[str],
    c_remaining_cores: List[str]
) -> tuple[float, List[str], List[str]]:
    """Initials match score; returns (score, missing_initials, extra_initials)."""
    if not q_initials:
        return 70.0, [], []
    
    q_set = set(i.lower() for i in q_initials)
    c_set = set(i.lower() for i in c_initials)
    
    # First letters of candidate's remaining cores (for abbreviation matching)
    c_core_first_letters = set(
        core[0].lower() for core in c_remaining_cores if core
    )
    
    # Find exact matches
    exact_matches = q_set & c_set
    
    # Find missing initials (in query but not candidate)
    missing = q_set - c_set
    
    # Check if missing initials match any candidate core first letters
    # This is abbreviation logic: "K" matches "Kumar"
    abbrev_matches = missing & c_core_first_letters
    partial_matches = len(abbrev_matches)
    truly_missing = list(missing - abbrev_matches)
    
    # Extra initials (in candidate but not query)
    extra = list(c_set - q_set)
    
    # Calculate score
    total_matches = len(exact_matches) + (partial_matches * 0.7)  # Partial credit
    base_score = 100 * (total_matches / len(q_initials))
    
    return base_score, truly_missing, extra


def compute_phonetic_core_score(q_cores: List[str], c_cores: List[str]) -> float:
    """WRatio on phonetically normalized full core strings (e.g. geetha vs gita)."""
    q_phonetic = phonetic_core_string(q_cores)
    c_phonetic = phonetic_core_string(c_cores)
    
    if not q_phonetic or not c_phonetic:
        return 0.0
    
    return fuzz.WRatio(q_phonetic, c_phonetic)


def compute_full_string_score(q_normalized: str, c_normalized: str) -> float:
    """Overall string similarity (WRatio)."""
    if not q_normalized or not c_normalized:
        return 0.0
    
    return fuzz.WRatio(q_normalized, c_normalized)


def compute_edit_distance_score(q_str: str, c_str: str) -> float:
    """Levenshtein-based score: 0 distance = 100, scaled by max length."""
    if not q_str or not c_str:
        return 0.0
    
    distance = Levenshtein.distance(q_str, c_str)
    max_len = max(len(q_str), len(c_str))
    
    # Normalize: 0 distance = 100, max distance = 0
    return 100 * (1 - distance / max_len)


def score_candidate(
    q_normalized: str,
    q_classified: ClassifiedTokens,
    c_normalized: str,
    c_classified: ClassifiedTokens
) -> ScoreBreakdown:
    """Full score breakdown for one query vs one candidate."""
    # Extract components
    q_first = q_classified.first_core
    c_first = c_classified.first_core
    q_remaining = q_classified.remaining_core
    c_remaining = c_classified.remaining_core
    q_initials = q_classified.all_initials_expanded
    c_initials = c_classified.all_initials_expanded
    
    # Compute component scores
    first_name_score = compute_first_name_score(q_first, c_first)
    edit_distance_score = compute_edit_distance_score(q_normalized, c_normalized)
    other_core_score = compute_other_core_score(q_remaining, c_remaining, c_initials)
    initials_score, missing_initials, extra_initials = compute_initials_score(
        q_initials, c_initials, c_remaining
    )
    phonetic_core_score = compute_phonetic_core_score(
        q_classified.core_tokens, c_classified.core_tokens
    )
    full_string_score = compute_full_string_score(q_normalized, c_normalized)
    
    # Compute penalties
    missing_initial_penalty = PENALTY_MISSING_INITIAL * len(missing_initials)
    # When first name is strong but not exact (e.g. Ganu-Ganesh ~74), soften initial penalties so close first names can outrank initials-only matches.
    # Do not soften when first name is exact (100) so that e.g. "Vignesh Kumar" is still penalized for missing R vs "Vignesh Kumar R".
    first_name_strong_not_exact = (
        FIRST_NAME_STRONG_THRESHOLD <= first_name_score < 99.0
    )
    if first_name_strong_not_exact:
        missing_initial_penalty = min(missing_initial_penalty, MISSING_INITIAL_PENALTY_CAP_WHEN_STRONG)
        if missing_initials:
            initials_score = max(initials_score, 100.0)  # treat initials as satisfied when first name is strong (typo/variant)
            missing_initial_penalty = 0.0
    
    # Extra initial penalty only if query has initials
    if q_initials:
        extra_initial_penalty = PENALTY_EXTRA_INITIAL * len(extra_initials)
    else:
        extra_initial_penalty = 0.0
    if first_name_strong_not_exact:
        extra_initial_penalty = min(extra_initial_penalty, EXTRA_INITIAL_PENALTY_CAP_WHEN_STRONG)
    
    # Missing core penalty: if query has more core tokens than candidate
    missing_cores = max(0, len(q_classified.core_tokens) - len(c_classified.core_tokens))
    # Only penalize beyond first core
    missing_core_penalty = PENALTY_MISSING_CORE * max(0, missing_cores)
    
    # Overlong penalty: if candidate has many more cores than query
    extra_cores = max(0, len(c_classified.core_tokens) - len(q_classified.core_tokens) - 1)
    overlong_penalty = PENALTY_OVERLONG_CANDIDATE * extra_cores
    
    # Length difference penalty: penalize candidates with different first name length
    # This helps prefer same-length first names (e.g., "aman" over "ajmal" for query "amal")
    # Only applies to first core token to avoid penalizing multi-word names unfairly
    q_first_len = len(q_first) if q_first else 0
    c_first_len = len(c_first) if c_first else 0
    length_diff = abs(q_first_len - c_first_len)
    length_diff_penalty = PENALTY_LENGTH_DIFF * length_diff
    if first_name_strong_not_exact:
        length_diff_penalty = min(length_diff_penalty, LENGTH_DIFF_PENALTY_CAP_WHEN_STRONG)
    
    # Weighted combination
    weighted_score = (
        WEIGHT_FIRST_NAME * first_name_score +
        WEIGHT_EDIT_DISTANCE * edit_distance_score +
        WEIGHT_OTHER_CORE * other_core_score +
        WEIGHT_INITIALS * initials_score +
        WEIGHT_PHONETIC_CORE * phonetic_core_score +
        WEIGHT_FULL_STRING * full_string_score
    )
    
    # Apply penalties
    final_score = weighted_score - missing_initial_penalty - extra_initial_penalty
    final_score = final_score - missing_core_penalty - overlong_penalty - length_diff_penalty
    
    # Clamp to [0, 100]
    final_score = max(0.0, min(100.0, final_score))
    
    return ScoreBreakdown(
        first_name_score=first_name_score,
        edit_distance_score=edit_distance_score,
        other_core_score=other_core_score,
        initials_score=initials_score,
        phonetic_core_score=phonetic_core_score,
        full_string_score=full_string_score,
        missing_initial_penalty=missing_initial_penalty,
        extra_initial_penalty=extra_initial_penalty,
        missing_core_penalty=missing_core_penalty,
        overlong_penalty=overlong_penalty,
        length_diff_penalty=length_diff_penalty,
        final_score=final_score,
        query_normalized=q_normalized,
        candidate_normalized=c_normalized,
        query_cores=q_classified.core_tokens,
        candidate_cores=c_classified.core_tokens,
        query_initials=q_initials,
        candidate_initials=c_initials,
        missing_initials=missing_initials,
        extra_initials=extra_initials,
    )
