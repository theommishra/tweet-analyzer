import re

class TextPreprocessor:
    """
    Advanced text preprocessor for emotion classification.
    Handles lowercasing, contraction expansion, emoji translation,
    negation scope tagging, and punctuation normalization.
    """

    CONTRACTIONS = {
        "i'm": "i am",
        "im": "i am",
        "don't": "do not",
        "dont": "do not",
        "can't": "cannot",
        "cant": "cannot",
        "isn't": "is not",
        "isnt": "is not",
        "won't": "will not",
        "wont": "will not",
        "it's": "it is",
        "its": "it is",
        "that's": "that is",
        "thats": "that is",
        "i've": "i have",
        "ive": "i have",
        "i'll": "i will",
        "ill": "i will",
        "couldn't": "could not",
        "couldnt": "could not",
        "shouldn't": "should not",
        "shouldnt": "should not",
        "wouldn't": "would not",
        "wouldnt": "would not",
        "haven't": "have not",
        "havent": "have not",
        "hasn't": "has not",
        "hasnt": "has not",
        "hadn't": "had not",
        "hadnt": "had not",
        "wasn't": "was not",
        "wasnt": "was not",
        "weren't": "were not",
        "werent": "were not",
        "aren't": "are not",
        "arent": "are not",
        "you're": "you are",
        "youre": "you are",
        "they're": "they are",
        "we're": "we are",
    }

    EMOJI_MAP = {
        # Joy / Happiness
        "😊": " [EMOJI_JOY] ", "😃": " [EMOJI_JOY] ", "😄": " [EMOJI_JOY] ", "😁": " [EMOJI_JOY] ",
        "😆": " [EMOJI_JOY] ", "🥰": " [EMOJI_JOY] ", "😍": " [EMOJI_JOY] ", "🥳": " [EMOJI_JOY] ",
        "🎉": " [EMOJI_JOY] ", "✨": " [EMOJI_JOY] ", "❤️": " [EMOJI_JOY] ", "💙": " [EMOJI_JOY] ",
        "💖": " [EMOJI_JOY] ", "🙂": " [EMOJI_JOY] ", "😀": " [EMOJI_JOY] ", "😎": " [EMOJI_JOY] ",
        # Sadness
        "😢": " [EMOJI_SADNESS] ", "😭": " [EMOJI_SADNESS] ", "😿": " [EMOJI_SADNESS] ",
        "💔": " [EMOJI_SADNESS] ", "😞": " [EMOJI_SADNESS] ", "😔": " [EMOJI_SADNESS] ",
        "😟": " [EMOJI_SADNESS] ", "🥺": " [EMOJI_SADNESS] ", "☹️": " [EMOJI_SADNESS] ",
        "🙁": " [EMOJI_SADNESS] ", "💧": " [EMOJI_SADNESS] ",
        # Anger
        "😡": " [EMOJI_ANGER] ", "🤬": " [EMOJI_ANGER] ", "😠": " [EMOJI_ANGER] ",
        "👿": " [EMOJI_ANGER] ", "💢": " [EMOJI_ANGER] ", "😤": " [EMOJI_ANGER] ",
        # Fear
        "😨": " [EMOJI_FEAR] ", "😱": " [EMOJI_FEAR] ", "😰": " [EMOJI_FEAR] ",
        "😳": " [EMOJI_FEAR] ", "😬": " [EMOJI_FEAR] ", "💀": " [EMOJI_FEAR] ",
        "🙀": " [EMOJI_FEAR] ", "😨": " [EMOJI_FEAR] ",
        # Surprise
        "😲": " [EMOJI_SURPRISE] ", "😮": " [EMOJI_SURPRISE] ", "😯": " [EMOJI_SURPRISE] ",
        "🤯": " [EMOJI_SURPRISE] ", "❗": " [EMOJI_SURPRISE] ", "⁉️": " [EMOJI_SURPRISE] ",
        "‼️": " [EMOJI_SURPRISE] ",
        # Disgust
        "🤢": " [EMOJI_DISGUST] ", "🤮": " [EMOJI_DISGUST] ", "😷": " [EMOJI_DISGUST] ",
        "💩": " [EMOJI_DISGUST] ", "😒": " [EMOJI_DISGUST] ",
    }

    NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "cannot", "without", "hardly", "barely", "lack"}

    def __init__(self):
        escaped_emojis = [re.escape(e) for e in self.EMOJI_MAP.keys()]
        self.emoji_pattern = re.compile("|".join(escaped_emojis))

    def extract_emojis(self, text: str) -> list[str]:
        """Finds all emojis present in the text."""
        return [match.group() for match in self.emoji_pattern.finditer(text)]

    def translate_emojis(self, text: str) -> str:
        """Replaces emojis with textual emotion tokens."""
        for emoji, token in self.EMOJI_MAP.items():
            if emoji in text:
                text = text.replace(emoji, token)
        return text

    def expand_contractions(self, text: str) -> str:
        """Expands common English contractions."""
        words = text.split()
        expanded_words = []
        for word in words:
            clean_word = word.lower().strip(".,!?;:\"'()[]{}")
            if clean_word in self.CONTRACTIONS:
                expanded_words.append(self.CONTRACTIONS[clean_word])
            else:
                expanded_words.append(word)
        return " ".join(expanded_words)

    def tag_negations(self, text: str) -> str:
        """
        Tags words following a negation word with NOT_ prefix to capture sentiment inversion.
        e.g., 'not happy' -> 'not NOT_happy'
        """
        words = text.split()
        tagged_words = []
        in_negation_scope = False
        negation_count = 0

        for word in words:
            clean = word.lower().strip(".,!?;:\"'()[]{}")
            
            if any(char in word for char in [".", "!", "?", ";"]):
                in_negation_scope = False

            if clean in self.NEGATION_WORDS or clean.endswith("n't"):
                in_negation_scope = True
                negation_count = 0
                tagged_words.append(word)
            elif in_negation_scope:
                if negation_count < 3: # Scope window of 3 words
                    tagged_words.append(f"NOT_{word}")
                    negation_count += 1
                else:
                    in_negation_scope = False
                    tagged_words.append(word)
            else:
                tagged_words.append(word)

        return " ".join(tagged_words)

    def clean_text(self, text: str) -> str:
        """
        Applies full cleaning pipeline:
        1. Extract & translate emojis
        2. Lowercase
        3. Expand contractions
        4. Tag negations
        5. Normalize whitespace
        """
        if not text or not isinstance(text, str):
            return ""

        text_with_emojis = self.translate_emojis(text)
        lower_text = text_with_emojis.lower()
        expanded_text = self.expand_contractions(lower_text)
        negated_text = self.tag_negations(expanded_text)

        cleaned = re.sub(r"[^\w\s\[\]]", " ", negated_text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def get_preprocessing_details(self, text: str) -> dict:
        """Returns step-by-step breakdown for UI debugging."""
        emojis = self.extract_emojis(text)
        translated = self.translate_emojis(text)
        expanded = self.expand_contractions(translated.lower())
        negated = self.tag_negations(expanded)
        final_cleaned = self.clean_text(text)

        return {
            "raw_text": text,
            "emojis_found": emojis,
            "translated_emojis": translated,
            "expanded_contractions": expanded,
            "negations_tagged": negated,
            "final_cleaned": final_cleaned
        }
