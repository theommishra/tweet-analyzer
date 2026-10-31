"""
model.py — Ekman 7-class emotion classifier inference wrapper
=============================================================
Supports v4 format (SentenceTransformer + NRC Lexicon + TextStat + Logistic Regression),
v3 format (TF-IDF word/char + NRC + TextStat), and legacy single-vectorizer format.
"""

import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocessor import TextPreprocessor

EKMAN_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

EMOTION_EMOJI = {
    "joy": "😊", "sadness": "😢", "anger": "😡",
    "fear": "😱", "surprise": "😲", "disgust": "🤢", "neutral": "😐"
}


class EmotionClassifier:
    """
    Unified local emotion classifier for Ekman's 7-class taxonomy.
    Uses SentenceTransformer("all-MiniLM-L6-v2") 384-dim contextual embeddings
    combined with NRC Lexicon features and TextStat features.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "emotion_model.joblib")

        self.model_path = model_path
        self.preprocessor = TextPreprocessor()
        self.classifier = None
        self.classes = EKMAN_EMOTIONS
        self.backend = "scikit-learn + SentenceTransformers"
        self.cv_macro_f1 = None
        self.model_version = "unknown"

        # Feature components
        self.embedder = None
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self._nrc_tf   = None
        self._stat_tf  = None
        
        # Legacy TF-IDF components
        self._word_vec = None
        self._char_vec = None
        self._legacy_vec = None

        self._load_or_train_model()

    def _load_or_train_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model not found at '{self.model_path}'. Training now...")
            from train import train_model
            train_model()

        try:
            model_data = joblib.load(self.model_path)
            self.classifier     = model_data["classifier"]
            self.classes        = model_data.get("classes", EKMAN_EMOTIONS)
            self.preprocessor   = model_data.get("preprocessor", self.preprocessor)
            self.cv_macro_f1    = model_data.get("cv_macro_f1_mean", None)
            self.model_version  = model_data.get("model_version", "legacy")

            if "embedding_model_name" in model_data or self.model_version == "v4_sentence_transformer_nrc":
                self.embedding_model_name = model_data.get("embedding_model_name", "all-MiniLM-L6-v2")
                print(f"Loading SentenceTransformer('{self.embedding_model_name}')...")
                self.embedder = SentenceTransformer(self.embedding_model_name)
                self._nrc_tf  = model_data["nrc_features"]
                self._stat_tf = model_data["stat_features"]
                self.backend  = f"SentenceTransformers ({self.embedding_model_name}) + LR + NRC"
            elif "word_vectorizer" in model_data:
                # v3 format
                self._word_vec  = model_data["word_vectorizer"]
                self._char_vec  = model_data["char_vectorizer"]
                self._nrc_tf    = model_data["nrc_features"]
                self._stat_tf   = model_data["stat_features"]
                self.backend    = "scikit-learn (TF-IDF + LR + NRC Lexicon)"
            else:
                # Legacy format
                self._legacy_vec = model_data.get("vectorizer")
                self.backend     = "scikit-learn (TF-IDF)"

            cv_pct = f"{self.cv_macro_f1 * 100:.1f}%" if self.cv_macro_f1 else "N/A"
            print(f"Emotion Classifier loaded [{self.model_version}]. CV Macro-F1: {cv_pct}")
        except Exception as e:
            print(f"Error loading model from '{self.model_path}': {e}. Retraining...")
            from train import train_model
            train_model()
            self._load_or_train_model()

    def _build_feature_matrix(self, raw_text: str, cleaned_text: str):
        """Builds the combined feature vector for inference."""
        if self.embedder is not None:
            # v4 format: SentenceTransformer + NRC + TextStat
            emb = self.embedder.encode([cleaned_text], show_progress_bar=False)
            nrc_sparse = self._nrc_tf.transform([raw_text])
            nrc_arr = nrc_sparse.toarray() if hasattr(nrc_sparse, "toarray") else np.array(nrc_sparse)
            stat_sparse = self._stat_tf.transform([cleaned_text])
            stat_arr = stat_sparse.toarray() if hasattr(stat_sparse, "toarray") else np.array(stat_sparse)
            return np.hstack([emb, nrc_arr, stat_arr])
        elif self._word_vec is not None:
            from scipy.sparse import hstack
            X_word = self._word_vec.transform([cleaned_text])
            X_char = self._char_vec.transform([cleaned_text])
            X_nrc  = self._nrc_tf.transform([raw_text])
            X_stat = self._stat_tf.transform([cleaned_text])
            return hstack([X_word, X_char, X_nrc, X_stat], format="csr")
        else:
            return self._legacy_vec.transform([cleaned_text])

    def predict(self, text: str) -> dict:
        """
        Runs full emotion prediction pipeline on a sentence.
        Returns primary emotion, confidence %, 7-class breakdown, preprocessing trace.
        """
        if not text or not text.strip():
            return {
                "error": "Empty text provided.",
                "primary_emotion": "neutral",
                "confidence": 1.0,
                "confidence_percentage": 100.0,
                "breakdown": [
                    {"emotion": e, "score": 1.0 if e == "neutral" else 0.0,
                     "percentage": 100.0 if e == "neutral" else 0.0}
                    for e in EKMAN_EMOTIONS
                ]
            }

        details = self.preprocessor.get_preprocessing_details(text)
        cleaned = details["final_cleaned"]
        X = self._build_feature_matrix(text, cleaned)

        probabilities = self.classifier.predict_proba(X)[0]

        emotion_scores = {}
        for cls_name, prob in zip(self.classes, probabilities):
            emotion_scores[cls_name.lower()] = float(prob)

        for e in EKMAN_EMOTIONS:
            if e not in emotion_scores:
                emotion_scores[e] = 0.0

        total = sum(emotion_scores.values()) or 1.0
        for e in emotion_scores:
            emotion_scores[e] /= total

        sorted_breakdown = sorted(
            [
                {"emotion": e, "score": round(s, 4), "percentage": round(s * 100, 1)}
                for e, s in emotion_scores.items()
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        top = sorted_breakdown[0]
        return {
            "text": text,
            "cleaned_text": cleaned,
            "primary_emotion": top["emotion"],
            "confidence": top["score"],
            "confidence_percentage": top["percentage"],
            "breakdown": sorted_breakdown,
            "preprocessing_details": details,
            "backend": self.backend,
            "model_version": self.model_version,
        }


if __name__ == "__main__":
    classifier = EmotionClassifier()
    sample_tests = [
        ("I am so happy and grateful for this wonderful day!", "joy"),
        ("I feel so lonely and heartbroken, nothing makes sense anymore.", "sadness"),
        ("Stop lying to me! I am absolutely furious at your behaviour!", "anger"),
        ("I am terrified about what the doctor might say tomorrow.", "fear"),
        ("Wait, WHAT?! I cannot believe that just happened!", "surprise"),
        ("That is utterly disgusting, how could anyone do that?", "disgust"),
        ("The meeting is scheduled for 3pm in conference room B.", "neutral"),
        ("I am not happy at all with how this turned out.", "sadness"),
        ("She passed away last night. I just got the call.", "sadness"),
        ("My hands are shaking, I am so anxious about the exam.", "fear"),
        ("that game hurt", "sadness"),
    ]

    correct = 0
    for sentence, expected in sample_tests:
        res = classifier.predict(sentence)
        predicted = res["primary_emotion"]
        ok = "OK" if predicted == expected else "!!"
        safe_s = sentence.encode("ascii", errors="ignore").decode()
        print(f"{ok} [{expected:8s}->{predicted:8s}] {safe_s[:60]}")
        if predicted == expected:
            correct += 1

    print(f"\nTest accuracy: {correct}/{len(sample_tests)} = {correct/len(sample_tests)*100:.0f}%")
