# Weights, penalties, index constants (all tunable in one place)

# Indexing
MIN_SHORTLIST = 30  # Minimum candidates to consider before fallback
MAX_SHORTLIST = 2000  # Cap shortlist to avoid performance issues

# Scoring weights (sum to 1.0)
WEIGHT_FIRST_NAME = 0.30
WEIGHT_EDIT_DISTANCE = 0.15
WEIGHT_OTHER_CORE = 0.20
WEIGHT_INITIALS = 0.15
WEIGHT_PHONETIC_CORE = 0.10
WEIGHT_FULL_STRING = 0.10

# Penalties
PENALTY_MISSING_INITIAL = 12
PENALTY_EXTRA_INITIAL = 4
PENALTY_MISSING_CORE = 6
PENALTY_OVERLONG_CANDIDATE = 3
PENALTY_LENGTH_DIFF = 8

# When first name is strong but not exact (e.g. Ganu vs Ganesh), cap these so close first names outrank initial-only matches
FIRST_NAME_STRONG_THRESHOLD = 70
MISSING_INITIAL_PENALTY_CAP_WHEN_STRONG = 6
EXTRA_INITIAL_PENALTY_CAP_WHEN_STRONG = 0
LENGTH_DIFF_PENALTY_CAP_WHEN_STRONG = 8

# Abbreviation: max score when an initial matches a core token by first letter
ABBREV_MATCH_CAP = 70

# Phonetic index key length
PHONETIC_KEY_LENGTH = 3
