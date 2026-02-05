# Phonetic index: key -> candidate indices. Query hits a bucket instead of scanning all. Fallback: first-letter bucket, then full scan.

import csv
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

from src.normalize import normalize_text, tokenize, is_valid_name
from src.phonetic import phonetic_key_for_index, first_letter_key, phonetic_core_string
from src.initials import classify_tokens, ClassifiedTokens
from src.config import MIN_SHORTLIST, MAX_SHORTLIST


@dataclass
class Candidate:
    """One preprocessed candidate (original, normalized, classified, keys)."""
    original: str
    normalized: str
    tokens: List[str]
    classified: ClassifiedTokens
    phonetic_core: str
    phonetic_key: str
    first_letter: str


class NameIndex:
    """In-memory index: load from CSV, then shortlist by phonetic key (or fallback)."""
    
    def __init__(self):
        self.candidates: List[Candidate] = []
        self.phonetic_index: Dict[str, List[int]] = {}
        self.first_letter_index: Dict[str, List[int]] = {}
        self._normalized_set: Set[str] = set()  # For deduplication
    
    def load_from_csv(self, csv_path: str) -> int:
        """Load candidates from CSV (expects 'name' or 'Name' column). Returns count."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Names file not found: {csv_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both 'name' and 'Name' column names (case-insensitive)
                name = row.get('name') or row.get('Name') or ''
                name = name.strip()
                if name:
                    self._add_candidate(name)
        
        return len(self.candidates)
    
    def load_from_list(self, names: List[str]) -> int:
        """Load candidates from a list (for tests)."""
        for name in names:
            if name and name.strip():
                self._add_candidate(name.strip())
        return len(self.candidates)
    
    def _add_candidate(self, name: str) -> Optional[int]:
        """Add one candidate. Returns index or None if invalid/duplicate."""
        normalized = normalize_text(name)
        
        if not is_valid_name(normalized):
            return None
        
        if normalized in self._normalized_set:
            return None
        self._normalized_set.add(normalized)
        
        tokens = tokenize(normalized)
        classified = classify_tokens(tokens)
        
        # Compute phonetic representations
        first_core = classified.first_core
        phonetic_core = phonetic_core_string(classified.core_tokens)
        phonetic_key = phonetic_key_for_index(first_core) if first_core else "___"
        first_letter = first_letter_key(first_core) if first_core else "_"
        
        # Create candidate
        candidate = Candidate(
            original=name,
            normalized=normalized,
            tokens=tokens,
            classified=classified,
            phonetic_core=phonetic_core,
            phonetic_key=phonetic_key,
            first_letter=first_letter
        )
        
        idx = len(self.candidates)
        self.candidates.append(candidate)
        
        # Add to phonetic index
        if phonetic_key not in self.phonetic_index:
            self.phonetic_index[phonetic_key] = []
        self.phonetic_index[phonetic_key].append(idx)
        
        # Add to first-letter index
        if first_letter not in self.first_letter_index:
            self.first_letter_index[first_letter] = []
        self.first_letter_index[first_letter].append(idx)
        
        return idx
    
    def get_shortlist(self, query_phonetic_key: str, query_first_letter: str) -> List[int]:
        """Indices to score: phonetic bucket; if small add first-letter; if empty full scan; cap at MAX_SHORTLIST."""
        indices: Set[int] = set()
        
        if query_phonetic_key in self.phonetic_index:
            indices.update(self.phonetic_index[query_phonetic_key])
        
        if len(indices) < MIN_SHORTLIST:
            if query_first_letter in self.first_letter_index:
                indices.update(self.first_letter_index[query_first_letter])
        
        if len(indices) == 0:
            indices = set(range(len(self.candidates)))
        
        result = list(indices)
        if len(result) > MAX_SHORTLIST:
            result = result[:MAX_SHORTLIST]
        
        return result
    
    def get_all_indices(self) -> List[int]:
        return list(range(len(self.candidates)))
    
    def __len__(self) -> int:
        return len(self.candidates)


_global_index: Optional[NameIndex] = None


def get_or_create_index(csv_path: str) -> NameIndex:
    """Load index from CSV once (CLI singleton)."""
    global _global_index
    if _global_index is None:
        _global_index = NameIndex()
        _global_index.load_from_csv(csv_path)
    return _global_index


def create_fresh_index() -> NameIndex:
    """Empty index for tests."""
    return NameIndex()
