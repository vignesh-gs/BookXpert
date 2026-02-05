"""Token classification and initial expansion."""

import pytest
from src.initials import (
    is_core_token,
    is_initial_token,
    is_merged_initial,
    expand_merged_initial,
    classify_tokens,
    get_first_letters_of_cores
)


class TestTokenClassification:
    """Token classification helpers."""
    
    def test_is_core_token(self):
        """Core = length > 2, alphabetic."""
        assert is_core_token("vignesh") is True
        assert is_core_token("kumar") is True
        assert is_core_token("rao") is True  # 3 chars is core
        assert is_core_token("gs") is False  # 2 chars treated as merged initials
        assert is_core_token("ab") is False  # 2 chars treated as merged initials
        assert is_core_token("g") is False
        assert is_core_token("") is False
    
    def test_is_initial_token(self):
        """Initial = single alphabetic char."""
        assert is_initial_token("g") is True
        assert is_initial_token("s") is True
        assert is_initial_token("gs") is False
        assert is_initial_token("1") is False
        assert is_initial_token("") is False
    
    def test_is_merged_initial(self):
        """Merged = exactly 2 alphabetic chars."""
        assert is_merged_initial("gs") is True
        assert is_merged_initial("bs") is True
        assert is_merged_initial("g") is False
        assert is_merged_initial("gsk") is False


class TestExpandMergedInitial:
    """Expanding merged initials (gs -> g, s)."""
    
    def test_expand_two_chars(self):
        assert expand_merged_initial("gs") == ["g", "s"]
        assert expand_merged_initial("bs") == ["b", "s"]
    
    def test_non_two_chars_unchanged(self):
        assert expand_merged_initial("g") == ["g"]
        assert expand_merged_initial("gsk") == ["gsk"]


class TestClassifyTokens:
    """Full token classification."""
    
    def test_mixed_tokens(self):
        """Mix of cores, initials, merged."""
        result = classify_tokens(["vignesh", "g", "s"])
        assert result.core_tokens == ["vignesh"]
        assert result.initial_tokens == ["g", "s"]
        assert result.merged_initials == []
    
    def test_with_merged_initials(self):
        """'gs' detected as merged initials."""
        result = classify_tokens(["vignesh", "gs"])
        assert result.core_tokens == ["vignesh"]
        assert result.initial_tokens == []
        assert result.merged_initials == ["gs"]
    
    def test_all_initials_expanded(self):
        """all_initials_expanded includes expanded merged."""
        result = classify_tokens(["vignesh", "gs"])
        assert set(result.all_initials_expanded) == {"g", "s"}
        
        result2 = classify_tokens(["vignesh", "g", "s"])
        assert set(result2.all_initials_expanded) == {"g", "s"}
    
    def test_first_core_property(self):
        result = classify_tokens(["vignesh", "kumar", "r"])
        assert result.first_core == "vignesh"
        assert result.remaining_core == ["kumar"]
    
    def test_empty_tokens(self):
        result = classify_tokens([])
        assert result.core_tokens == []
        assert result.first_core == ""


class TestGetFirstLettersOfCores:
    """First letters of core tokens."""
    
    def test_basic(self):
        assert get_first_letters_of_cores(["kumar", "rao"]) == ["k", "r"]
    
    def test_empty(self):
        assert get_first_letters_of_cores([]) == []
