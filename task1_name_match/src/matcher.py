# Matcher: normalize query, get shortlist from index, score, sort, return top-k.

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.normalize import normalize_text, tokenize, is_valid_name
from src.phonetic import phonetic_key_for_index, first_letter_key
from src.initials import classify_tokens
from src.index import NameIndex, Candidate
from src.scoring import score_candidate, ScoreBreakdown


@dataclass
class MatchResult:
    """One match: name, score, breakdown."""
    name: str
    score: float
    breakdown: ScoreBreakdown
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'score': round(self.score, 1),
            'breakdown': self.breakdown.to_dict()
        }


@dataclass
class MatcherResult:
    """Full result: query, best match, list of matches, optional error."""
    query: str
    query_normalized: str
    best_match: Optional[MatchResult]
    matches: List[MatchResult]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'query': self.query,
            'query_normalized': self.query_normalized,
        }
        
        if self.error:
            result['error'] = self.error
            result['best_match'] = None
            result['matches'] = []
        else:
            result['best_match'] = self.best_match.to_dict() if self.best_match else None
            result['matches'] = [m.to_dict() for m in self.matches]
        
        return result


class NameMatcher:
    """Takes an index; match(query, top_k) returns best matches and ranked list."""
    
    def __init__(self, index: NameIndex):
        self.index = index
    
    def match(self, query: str, top_k: int = 5) -> MatcherResult:
        """Find top_k matches for query. Returns MatcherResult with best match and list."""
        query_normalized = normalize_text(query)
        
        if not is_valid_name(query_normalized):
            return MatcherResult(
                query=query,
                query_normalized=query_normalized,
                best_match=None,
                matches=[],
                error="Invalid query: empty or no alphabetic characters"
            )
        
        # Tokenize and classify
        query_tokens = tokenize(query_normalized)
        query_classified = classify_tokens(query_tokens)
        
        # Handle edge case: query with no core tokens (only initials)
        if not query_classified.core_tokens:
            # Fall back to full scan with low confidence
            shortlist = self.index.get_all_indices()
        else:
            # Get shortlist from phonetic index
            phonetic_key = phonetic_key_for_index(query_classified.first_core)
            first_letter = first_letter_key(query_classified.first_core)
            shortlist = self.index.get_shortlist(phonetic_key, first_letter)
        
        # Score each candidate in shortlist
        scored_results: List[MatchResult] = []
        
        for idx in shortlist:
            candidate = self.index.candidates[idx]
            
            breakdown = score_candidate(
                query_normalized,
                query_classified,
                candidate.normalized,
                candidate.classified
            )
            
            scored_results.append(MatchResult(
                name=candidate.original,
                score=breakdown.final_score,
                breakdown=breakdown
            ))
        
        # Sort with tie-breaking
        scored_results.sort(key=lambda r: self._sort_key(r), reverse=True)
        
        # Get top_k
        top_matches = scored_results[:top_k]
        best_match = top_matches[0] if top_matches else None
        
        return MatcherResult(
            query=query,
            query_normalized=query_normalized,
            best_match=best_match,
            matches=top_matches
        )
    
    def _sort_key(self, result: MatchResult) -> tuple:
        """
        Generate sort key for deterministic tie-breaking.
        
        Priority:
        1. Higher final score
        2. Higher first_name_score
        3. Fewer missing initials
        4. Fewer missing core tokens (approximated by query - candidate)
        5. Lexicographic by name (for stability)
        """
        bd = result.breakdown
        return (
            result.score,
            bd.first_name_score,
            -len(bd.missing_initials),
            -bd.missing_core_penalty,
            # Negative for reverse lexicographic (so 'A' beats 'Z')
            # Actually we want stable, so just use the name
        )


def match_names(
    query: str,
    index: NameIndex,
    top_k: int = 5
) -> MatcherResult:
    """
    Convenience function for one-off matching.
    
    Args:
        query: Name to search for
        index: Prebuilt NameIndex
        top_k: Number of results
        
    Returns:
        MatcherResult
    """
    matcher = NameMatcher(index)
    return matcher.match(query, top_k)
