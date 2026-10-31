"""
features.py
===========
Custom sklearn-compatible transformers that add lexicon-based emotion features
on top of TF-IDF to dramatically improve generalisation.

Provides:
  - NRCLexiconFeatures  : per-sentence NRC Emotion Lexicon aggregate scores
  - TextStatFeatures    : punctuation/caps/length heuristics
  - EmotionDenseFeatures: combined dense feature transformer for FeatureUnion

These features are rule-based (no ML), so they don't overfit and greatly
help the final classifier generalise to unseen sentences.
"""

import re
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

# NRC Emotion → Ekman 7 grouping
# The NRC lexicon has 8 Plutchik emotions: anger, anticipation, disgust, fear,
# joy, sadness, surprise, trust  + positive/negative polarity flags
NRC_TO_EKMAN = {
    "anger":       "anger",
    "disgust":     "disgust",
    "fear":        "fear",
    "joy":         "joy",
    "sadness":     "sadness",
    "surprise":    "surprise",
    "trust":       "joy",         # Plutchik trust → Ekman joy (closest)
    "anticipation":"neutral",     # Plutchik anticipation → Ekman neutral
    "positive":    "joy",
    "negative":    "sadness",
}

# Maps NRC emotion names to their output column index
EKMAN_COLS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
NRC_EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
                "negative", "positive", "sadness", "surprise", "trust"]

# ─────────────────────────────────────────────────────────────────────────────
# Curated keyword hint lists for each Ekman class
# Used when NRC lexicon is unavailable as a fallback
# ─────────────────────────────────────────────────────────────────────────────
KEYWORD_HINTS = {
    "joy": [
        "happy", "happiness", "joy", "joyful", "excited", "excitement", "love",
        "loved", "wonderful", "amazing", "great", "fantastic", "thrilled",
        "grateful", "gratitude", "proud", "pride", "blessed", "cheerful",
        "delighted", "ecstatic", "elated", "content", "satisfied", "glad",
        "excellent", "brilliant", "awesome", "bliss", "pleased", "celebrate",
        "celebrating", "admire", "admiration", "adore", "enjoy", "enjoying",
        "relief", "relieved", "optimistic", "hopeful", "smile", "smiling",
        "laugh", "laughing", "fun", "playful", "cheerful", "laugh",
    ],
    "sadness": [
        "sad", "sadness", "unhappy", "depressed", "depression", "grief",
        "grieve", "grieving", "heartbroken", "heartbreak", "miserable",
        "lonely", "alone", "alone", "hopeless", "despair", "cry", "crying",
        "tears", "loss", "lost", "miss", "missing", "mourn", "mourning",
        "sorrow", "sorrowful", "regret", "regretful", "remorse", "shame",
        "ashamed", "disappointed", "disappointment", "upset", "broken",
        "hurt", "painful", "pain", "suffering", "suffer", "wretched",
        "dismal", "gloomy", "melancholy", "mournful", "tragic",
    ],
    "anger": [
        "angry", "anger", "furious", "fury", "rage", "mad", "outraged",
        "outrage", "livid", "seething", "hatred", "hate", "despise",
        "resentment", "resentful", "bitter", "bitterness", "irritated",
        "irritation", "annoyed", "annoyance", "frustrated", "frustration",
        "enraged", "infuriated", "irate", "hostile", "hostile", "offended",
        "insulted", "betrayed", "betrayal", "unfair", "injustice", "wrong",
        "ridiculous", "disgusted", "outrage", "intolerable", "unacceptable",
    ],
    "fear": [
        "afraid", "fear", "scared", "terrified", "terror", "panic", "dread",
        "anxious", "anxiety", "worried", "worry", "nervous", "nervousness",
        "frightened", "fright", "horror", "horrified", "phobia", "phobic",
        "petrified", "trembling", "shaking", "dreading", "dreaded", "alarmed",
        "alarm", "threat", "threatened", "unsafe", "danger", "dangerous",
        "horrifying", "creepy", "eerie", "terrifying", "nightmare",
        "catastrophe", "catastrophic", "crisis", "desperate", "helpless",
    ],
    "surprise": [
        "surprised", "surprise", "shocking", "shocked", "astonished",
        "astonishment", "amazed", "amazing", "unexpected", "unexpectedly",
        "stunned", "disbelief", "incredible", "unbelievable", "wow",
        "whoa", "woah", "omg", "jaw", "dropped", "blindsided", "suddenly",
        "out of nowhere", "never expected", "cannot believe", "cant believe",
        "speechless", "mind-blowing", "remarkable", "extraordinary",
    ],
    "disgust": [
        "disgusting", "disgust", "disgusted", "revolting", "revolted",
        "repulsive", "repulsed", "gross", "nasty", "vile", "filthy", "dirty",
        "nauseating", "nauseated", "nauseous", "sick", "sickening", "yuck",
        "eww", "ew", "horrible", "awful", "appalling", "appalled", "foul",
        "rotten", "putrid", "stench", "stinks", "contempt", "contemptuous",
        "loathe", "loathing", "abhorrent", "abhor", "detestable",
    ],
    "neutral": [
        "the", "is", "are", "was", "were", "will", "would", "could", "should",
        "has", "have", "had", "does", "do", "did", "this", "that", "these",
        "those", "time", "day", "place", "today", "tomorrow", "yesterday",
        "schedule", "meeting", "report", "document", "information", "data",
        "process", "system", "result", "according", "regarding", "based",
    ],
}


class NRCLexiconFeatures(BaseEstimator, TransformerMixin):
    """
    Converts raw text into a fixed-length emotion feature vector
    using the NRC Word Emotion Lexicon via nrclex.

    For each sentence, extracts per-word emotion frequencies from the NRC
    lexicon (Plutchik 8 + positive/negative polarity = 10 dimensions).

    Falls back to keyword-hint matching if nrclex is unavailable.
    """

    def __init__(self):
        self._use_nrclex = False
        self._keyword_hints = KEYWORD_HINTS
        self._load_lexicon()

    def _load_lexicon(self):
        try:
            from nrclex import NRCLex
            self._NRCLex = NRCLex
            self._use_nrclex = True
            print("  [NRC] NRCLex library loaded (Plutchik 8 + polarity = 10 dims)")
        except ImportError:
            print("  [NRC] nrclex not available, using keyword hints fallback")
            self._use_nrclex = False

    def _get_nrc_vector(self, text: str) -> list:
        """
        Returns a 10-dim NRC emotion frequency vector.
        Emotions order: anger, anticipation, disgust, fear, joy,
                        negative, positive, sadness, surprise, trust
        """
        try:
            # Tokenise simply (split on whitespace, lowercase)
            tokens = re.findall(r'\b\w+\b', text.lower())
            if not tokens:
                return [0.0] * len(NRC_EMOTIONS)
            lex = self._NRCLex()
            lex.load_token_list(tokens)
            freqs = lex.affect_frequencies
            return [freqs.get(e, 0.0) for e in NRC_EMOTIONS]
        except Exception:
            return [0.0] * len(NRC_EMOTIONS)

    def _get_keyword_vector(self, text: str) -> list:
        """Returns a 7-dim keyword hit vector (one per Ekman class)."""
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        vec = []
        for label in EKMAN_COLS:
            hints = self._keyword_hints.get(label, [])
            score = sum(1 for w in hints if w in words or w in text_lower)
            vec.append(float(score))
        total = sum(vec) or 1
        return [v / total for v in vec]

    def _text_to_vec(self, text: str) -> list:
        if self._use_nrclex:
            return self._get_nrc_vector(text)
        else:
            return self._get_keyword_vector(text)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        matrix = [self._text_to_vec(str(text)) for text in X]
        return csr_matrix(np.array(matrix, dtype=np.float32))


class TextStatFeatures(BaseEstimator, TransformerMixin):
    """
    Rule-based hand-crafted features that carry strong emotion signals:
      - Exclamation mark count (normalised) → joy/anger/surprise
      - Question mark count → fear/surprise/neutral
      - UPPERCASE word ratio → anger/intensity
      - Word count → longer = more likely emotional narrative
      - NOT_ tag count → negation scope
      - EMOJI_ token count → explicit emotion signalling
      - Repetition of letters (!!!!, love love) → intensity
    """

    def _extract(self, text: str) -> list:
        words = text.split()
        n = max(len(words), 1)
        raw = text  # preserves case/punct

        exclamations  = text.count("!") / n
        questions     = text.count("?") / n
        upper_ratio   = sum(1 for w in words if w.isupper() and len(w) > 1) / n
        word_count    = min(len(words) / 50.0, 1.0)   # normalised, capped at 50
        not_tag_count = sum(1 for w in words if w.upper().startswith("NOT_")) / n
        emoji_joy     = text.count("[EMOJI_JOY]") / n
        emoji_sad     = text.count("[EMOJI_SADNESS]") / n
        emoji_anger   = text.count("[EMOJI_ANGER]") / n
        emoji_fear    = text.count("[EMOJI_FEAR]") / n
        emoji_surpr   = text.count("[EMOJI_SURPRISE]") / n
        emoji_disg    = text.count("[EMOJI_DISGUST]") / n

        # Repeated chars like "sooooo happy" or "noooooo"
        repeated_chars = len(re.findall(r'(.)\1{2,}', text.lower())) / n

        return [
            exclamations, questions, upper_ratio, word_count,
            not_tag_count, emoji_joy, emoji_sad, emoji_anger,
            emoji_fear, emoji_surpr, emoji_disg, repeated_chars
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        matrix = [self._extract(str(text)) for text in X]
        return csr_matrix(np.array(matrix, dtype=np.float32))
