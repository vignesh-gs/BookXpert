"""Phonetic normalization."""

import pytest
from src.phonetic import (
    phonetic_rewrite, 
    phonetic_key_for_index, 
    phonetic_core_string,
    first_letter_key
)


class TestPhoneticRewrite:
    """Phonetic rewrite rules."""
    
    def test_ee_to_i(self):
        """ee -> i."""
        assert phonetic_rewrite("geetha") == "gita"
        assert phonetic_rewrite("preet") == "prit"
    
    def test_aa_to_a(self):
        """Long 'aa' becomes short 'a'."""
        assert phonetic_rewrite("raama") == "rama"
        assert phonetic_rewrite("saagar") == "sagar"
    
    def test_oo_to_u(self):
        """oo -> u."""
        assert phonetic_rewrite("pooja") == "puja"
    
    def test_th_to_t(self):
        """th -> t (ee->i first, then th->t)."""
        assert phonetic_rewrite("geetha") == "gita"
        assert phonetic_rewrite("karthik") == "kartik"
    
    def test_combined_rules(self):
        """Multiple rules (aa->a, sh->s)."""
        assert phonetic_rewrite("raakesh") == "rakes"
    
    def test_short_tokens_unchanged(self):
        """Single-letter tokens unchanged."""
        assert phonetic_rewrite("a") == "a"
        assert phonetic_rewrite("s") == "s"
    
    def test_double_consonant_collapse(self):
        """Double consonants collapse (kannada -> kanada)."""
        assert phonetic_rewrite("kannada") == "kanada"


class TestPhoneticKeyForIndex:
    """Phonetic key for index (first 3 chars of rewrite)."""
    
    def test_basic_key(self):
        """First 3 chars of phonetic rewrite."""
        assert phonetic_key_for_index("geetha") == "git"
        assert phonetic_key_for_index("vignesh") == "vig"
    
    def test_short_token_padded(self):
        """Short tokens padded (an -> an_)."""
        assert phonetic_key_for_index("an") == "an_"
    
    def test_empty_returns_placeholder(self):
        assert phonetic_key_for_index("") == "___"


class TestPhoneticCoreString:
    """Full phonetic core string (e.g. geetha sajeev -> gita sajiv)."""
    
    def test_multiple_cores(self):
        result = phonetic_core_string(["geetha", "sajeev"])
        assert result == "gita sajiv"
    
    def test_single_core(self):
        result = phonetic_core_string(["geetha"])
        assert result == "gita"
    
    def test_empty_list(self):
        assert phonetic_core_string([]) == ""


class TestFirstLetterKey:
    """First-letter fallback key."""
    
    def test_basic(self):
        assert first_letter_key("geetha") == "g"
        assert first_letter_key("vignesh") == "v"
    
    def test_empty(self):
        assert first_letter_key("") == "_"
