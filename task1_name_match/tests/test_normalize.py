"""Text normalization."""

import pytest
from src.normalize import normalize_text, tokenize, is_valid_name, normalize_and_tokenize


class TestNormalizeText:
    """normalize_text."""
    
    def test_basic_lowercase(self):
        assert normalize_text("Vignesh") == "vignesh"
        assert normalize_text("GEETHA") == "geetha"
    
    def test_dot_to_space(self):
        """Dots become spaces so initials stay (Vignesh.G.S -> vignesh g s)."""
        assert normalize_text("Vignesh.G.S") == "vignesh g s"
        assert normalize_text("A.P.J.") == "a p j"
    
    def test_hyphen_to_space(self):
        """Hyphens become spaces."""
        assert normalize_text("Anne-Marie Johnson") == "anne marie johnson"
    
    def test_mixed_punctuation(self):
        """Punctuation becomes spaces."""
        assert normalize_text("Geetha.B.S") == "geetha b s"
        assert normalize_text("Geetha B.S.") == "geetha b s"
        assert normalize_text("A.P.J. Abdul Kalam") == "a p j abdul kalam"
    
    def test_collapse_multiple_spaces(self):
        """Multiple spaces collapse to one."""
        assert normalize_text("Vignesh  G   S") == "vignesh g s"
        assert normalize_text("  Geetha  ") == "geetha"
    
    def test_empty_and_whitespace(self):
        """Empty and whitespace-only strings."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""
    
    def test_preserves_numbers(self):
        """Numbers preserved."""
        assert normalize_text("Test123") == "test123"


class TestTokenize:
    """tokenize."""
    
    def test_basic_split(self):
        assert tokenize("vignesh g s") == ["vignesh", "g", "s"]
    
    def test_empty_string(self):
        assert tokenize("") == []
    
    def test_single_token(self):
        assert tokenize("geetha") == ["geetha"]


class TestIsValidName:
    """Test cases for is_valid_name function."""
    
    def test_valid_names(self):
        assert is_valid_name("vignesh") is True
        assert is_valid_name("vignesh g s") is True
    
    def test_invalid_names(self):
        assert is_valid_name("") is False
        assert is_valid_name("   ") is False
        assert is_valid_name("123") is False


class TestNormalizeAndTokenize:
    """normalize + tokenize together."""
    
    def test_combined(self):
        norm, tokens = normalize_and_tokenize("Vignesh.G.S")
        assert norm == "vignesh g s"
        assert tokens == ["vignesh", "g", "s"]
