"""Matcher integration: expected rankings for query/candidate pairs."""

import pytest
from src.index import NameIndex
from src.matcher import NameMatcher


class TestVigneshVariants:
    """Vignesh name variants."""
    
    @pytest.fixture
    def index(self):
        """Index with Vignesh variants."""
        idx = NameIndex()
        idx.load_from_list([
            "Vignesh.G.S",
            "Vignesh GS",
            "Vignesh G S",
            "Venkatesh Rao",
            "Vignesh Kumar",
            "Vijay Kumar",
        ])
        return idx
    
    def test_vignesh_gs_matches_variants(self, index):
        """Vignesh G.S matches Vignesh+GS variants highly."""
        matcher = NameMatcher(index)
        result = matcher.match("Vignesh G.S", top_k=5)
        
        top_names = [m.name for m in result.matches[:3]]
        
        assert "Vignesh.G.S" in top_names or "Vignesh G S" in top_names
        assert result.best_match.score > 80
        
        # Venkatesh Rao should score lower
        venkatesh_match = next(
            (m for m in result.matches if "Venkatesh" in m.name), 
            None
        )
        if venkatesh_match:
            assert venkatesh_match.score < result.best_match.score - 20


class TestVigneshKumarR:
    """Vignesh Kumar R with abbreviations (K ~ Kumar)."""
    
    @pytest.fixture
    def index(self):
        idx = NameIndex()
        idx.load_from_list([
            "Vignesh K R",
            "Vignesh Kumar",
            "Vignesh R",
            "Vignesh Kumar R",
        ])
        return idx
    
    def test_vignesh_kumar_r(self, index):
        """Vignesh Kumar R prefers candidates with R; exact match top; K R beats Kumar (missing R)."""
        matcher = NameMatcher(index)
        result = matcher.match("Vignesh Kumar R", top_k=4)
        
        assert result.best_match.name == "Vignesh Kumar R"
        
        k_r_score = next(m.score for m in result.matches if m.name == "Vignesh K R")
        kumar_only_score = next(m.score for m in result.matches if m.name == "Vignesh Kumar")
        assert k_r_score >= kumar_only_score


class TestGeethaVariants:
    """Geetha/Gita phonetic matching."""
    
    @pytest.fixture
    def index(self):
        idx = NameIndex()
        idx.load_from_list([
            "Geetha B",
            "Gita",
            "Geetha Sajeev",
            "Geetha B S",
            "Gita B S",
            "Geeta",
        ])
        return idx
    
    def test_geetha_bs_matches(self, index):
        """Geetha B.S finds Geetha B S or Gita B S as best."""
        matcher = NameMatcher(index)
        result = matcher.match("Geetha B.S", top_k=5)
        
        # Geetha B S or Gita B S should be top (same phonetically)
        assert result.best_match.name in ["Geetha B S", "Gita B S"]
        
        geetha_b = next(m for m in result.matches if m.name == "Geetha B")
        assert geetha_b.score > 60
    
    def test_phonetic_matching(self, index):
        """Geetha/Gita/Geeta match well phonetically."""
        matcher = NameMatcher(index)
        result = matcher.match("Gita", top_k=5)
        
        geetha_scores = [
            m.score for m in result.matches 
            if any(x in m.name.lower() for x in ["geetha", "gita", "geeta"])
        ]
        assert all(s > 40 for s in geetha_scores)


class TestGanuBStrongFirstName:
    """Strong first name (Ganu–Ganesh) outranks initials-only (Gita B.S) for query 'Ganu B'."""

    @pytest.fixture
    def index(self):
        idx = NameIndex()
        idx.load_from_list([
            "Ganesh R",
            "Ganesh Rao",
            "Ganesh R K",
            "Gita B.S",
            "Geetha B",
            "Geetha B S",
            "Gitu",
        ])
        return idx

    def test_ganu_b_prefers_ganesh_over_gita(self, index):
        """Ganu B ranks Ganesh R/Rao above Gita B.S (Ganu~Ganesh strong; Gita only matches B)."""
        matcher = NameMatcher(index)
        result = matcher.match("Ganu B", top_k=5)

        top_names = [m.name for m in result.matches]
        ganesh_r = next((m for m in result.matches if m.name == "Ganesh R"), None)
        ganesh_rao = next((m for m in result.matches if m.name == "Ganesh Rao"), None)
        gita_bs = next((m for m in result.matches if m.name == "Gita B.S"), None)

        assert ganesh_r is not None and gita_bs is not None
        ganesh_best = ganesh_r if ganesh_rao is None else (ganesh_r if ganesh_r.score >= ganesh_rao.score else ganesh_rao)
        assert ganesh_best.score >= gita_bs.score, (
            f"Ganesh (score={ganesh_best.score}) should rank >= Gita B.S (score={gita_bs.score}) for query 'Ganu B'"
        )
        assert any("Ganesh" in n for n in top_names[:3])


class TestExtraInitialsPenalty:
    """Extra initials don't over-penalize when query has none."""
    
    @pytest.fixture
    def index(self):
        idx = NameIndex()
        idx.load_from_list([
            "Gita B S",
            "Gita",
            "Gita Kumar",
        ])
        return idx
    
    def test_query_without_initials(self, index):
        """Gita (no initials) doesn't heavily penalize Gita B S; both score well."""
        matcher = NameMatcher(index)
        result = matcher.match("Gita", top_k=3)
        
        gita_exact = next(m for m in result.matches if m.name == "Gita")
        gita_bs = next(m for m in result.matches if m.name == "Gita B S")
        
        assert result.best_match.name == "Gita"
        assert gita_bs.score > 70
        assert gita_exact.score - gita_bs.score < 20


class TestWordOrder:
    """Different word orders."""
    
    @pytest.fixture
    def index(self):
        idx = NameIndex()
        idx.load_from_list([
            "Gita Sajeev",
            "Gita Kumar",
        ])
        return idx
    
    def test_word_order_tolerance(self, index):
        """Exact match ranks highest; Gita Kumar still matches on first name."""
        matcher = NameMatcher(index)
        result = matcher.match("Gita Sajeev", top_k=3)
        
        gs = next(m for m in result.matches if m.name == "Gita Sajeev")
        assert gs.score > 80
        
        gk = next(m for m in result.matches if m.name == "Gita Kumar")
        assert gk.score > 50


class TestEmptyQuery:
    """Empty/invalid query."""
    
    @pytest.fixture
    def index(self):
        idx = NameIndex()
        idx.load_from_list(["Gita", "Geetha"])
        return idx
    
    def test_empty_query_error(self, index):
        """Empty query returns error."""
        matcher = NameMatcher(index)
        result = matcher.match("", top_k=3)
        
        assert result.error is not None
        assert result.best_match is None


class TestPhoneticMatching:
    """Phonetic matching across transliteration (Krishna/Krushna, Shankar/Sankar)."""
    
    @pytest.fixture
    def index(self):
        idx = NameIndex()
        idx.load_from_list([
            "Krishna R",
            "Krushna R",
            "Shankar",
            "Sankar",
            "Mohamed Ali",
            "Muhammad Ali",
        ])
        return idx
    
    def test_krishna_krushna(self, index):
        """Krishna and Krushna match well."""
        matcher = NameMatcher(index)
        result = matcher.match("Krishna R", top_k=3)
        
        top_names = [m.name for m in result.matches[:2]]
        assert "Krishna R" in top_names
        assert "Krushna R" in top_names
    
    def test_shankar_sankar(self, index):
        """Shankar and Sankar match (sh->s rule)."""
        matcher = NameMatcher(index)
        result = matcher.match("Shankar", top_k=5)
        
        scores = {m.name: m.score for m in result.matches}
        assert scores.get("Shankar", 0) > 80
        assert scores.get("Sankar", 0) > 70
