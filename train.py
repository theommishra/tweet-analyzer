"""
train.py — Contextual Sentence-Transformer + NRC Lexicon + TextStat Emotion Classifier
========================================================================================
Upgraded Pipeline:
  1. SentenceTransformer("all-MiniLM-L6-v2") — 384-dimensional contextual sentence embeddings
  2. NRC Emotion Lexicon features — rule-based emotion frequencies (10 dims)
  3. TextStat features — punctuation, caps, emoji, negation counts (12 dims)
  4. Combined Feature Matrix X: np.hstack([embeddings_384, nrc_features, textstat_features]) -> (N, 406)
  5. Logistic Regression (C=2.0, max_iter=2000, class_weight="balanced", solver="saga", random_state=42)
  6. 5-Fold Stratified Cross-Validation reporting CV Accuracy & Macro F1 vs 75.99% / 0.7824 baseline
"""

import os
import json
import time
import joblib
import numpy as np
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from preprocessor import TextPreprocessor
from features import NRCLexiconFeatures, TextStatFeatures

EKMAN_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "emotion_model.joblib")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "emotion_dataset.json")


def load_or_build_dataset() -> list:
    if os.path.exists(DATASET_PATH):
        print(f"Loading cached dataset from '{DATASET_PATH}'...")
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
        print(f"  Loaded {len(records):,} samples.")
        return records
    else:
        print("Dataset not found — downloading/building now...")
        from dataset_builder import build_dataset
        return build_dataset(max_per_class=3000, output_path=DATASET_PATH)


def preprocess_texts(texts: list, preprocessor: TextPreprocessor) -> list:
    cleaned = []
    for i, t in enumerate(texts):
        if i % 3000 == 0 and i > 0:
            print(f"  Preprocessed {i:,}/{len(texts):,}...")
        cleaned.append(preprocessor.clean_text(t))
    return cleaned


def build_feature_pipeline(cleaned_texts: list, raw_texts: list, model_name: str = "all-MiniLM-L6-v2"):
    """
    Build 3-component dense feature matrix:
      1. SentenceTransformer("all-MiniLM-L6-v2") embeddings (384 dims) — contextual semantics
      2. NRC Lexicon emotion scores (10 dims) — rule-based lexicons
      3. TextStat features (12 dims) — punctuation, caps, emoji, negation counts
    """
    print(f"\nComputing 384-dim sentence embeddings using SentenceTransformer('{model_name}')...")
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(cleaned_texts, show_progress_bar=True, batch_size=64)
    print(f"  Sentence Embeddings shape: {embeddings.shape}")

    print("Extracting NRC Emotion Lexicon features...")
    nrc_tf = NRCLexiconFeatures()
    X_nrc_sparse = nrc_tf.transform(raw_texts)
    X_nrc = X_nrc_sparse.toarray() if hasattr(X_nrc_sparse, "toarray") else np.array(X_nrc_sparse)
    print(f"  NRC Lexicon shape: {X_nrc.shape}")

    print("Extracting text stat features...")
    stat_tf = TextStatFeatures()
    X_stat_sparse = stat_tf.transform(cleaned_texts)
    X_stat = X_stat_sparse.toarray() if hasattr(X_stat_sparse, "toarray") else np.array(X_stat_sparse)
    print(f"  TextStat shape: {X_stat.shape}")

    # Stack all features horizontally
    X = np.hstack([embeddings, X_nrc, X_stat])
    print(f"  Combined feature matrix shape: {X.shape}")

    return X, embedder, nrc_tf, stat_tf


def train_model():
    t0 = time.time()
    print("=" * 70)
    print("  Ekman 7-Class Emotion Classifier — SentenceTransformer Upgrade")
    print("=" * 70)

    # 1. Load dataset
    records = load_or_build_dataset()
    preprocessor = TextPreprocessor()
    raw_texts = [r[0] for r in records]
    labels    = [r[1] for r in records]

    dist = Counter(labels)
    print(f"\nDataset: {len(labels):,} samples")
    for lbl in EKMAN_LABELS:
        bar = "#" * (dist.get(lbl, 0) // 100)
        print(f"  {lbl:10s} {dist.get(lbl, 0):>5,}  {bar}")

    # 2. Preprocess texts
    print(f"\nPreprocessing {len(raw_texts):,} texts...")
    cleaned_texts = preprocess_texts(raw_texts, preprocessor)

    # 3. Build features (384-dim Sentence Transformer + NRC + TextStat)
    model_name = "all-MiniLM-L6-v2"
    X, embedder, nrc_tf, stat_tf = build_feature_pipeline(cleaned_texts, raw_texts, model_name=model_name)
    y = np.array(labels)

    # 4. 5-Fold Stratified Cross-Validation
    print("\nRunning 5-fold stratified cross-validation...")
    lr_clf = LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        solver="saga",
        random_state=42
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_f1  = cross_val_score(lr_clf, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    cv_acc = cross_val_score(lr_clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)

    print(f"  CV Macro F1:   {cv_f1.round(4)}  mean={cv_f1.mean():.4f}")
    print(f"  CV Accuracy:   {cv_acc.round(4)}  mean={cv_acc.mean():.4f}")

    # 5. Final fit on full dataset
    print("\nFitting final Logistic Regression on full dataset...")
    lr_clf.fit(X, y)

    preds = lr_clf.predict(X)
    print(f"\n  Train accuracy (in-sample): {accuracy_score(y, preds) * 100:.2f}%")
    print(f"  CV accuracy (generalisation): {cv_acc.mean() * 100:.2f}% (vs 75.99% baseline)")
    print(f"  CV Macro F1 (generalisation): {cv_f1.mean():.4f} (vs 0.7824 baseline)")
    print("\n--- Per-Class Classification Report (on full training dataset) ---")
    print(classification_report(y, preds, target_names=sorted(set(labels)), digits=4))

    # 6. Save model artifacts
    model_data = {
        "embedding_model_name": model_name,
        "nrc_features": nrc_tf,
        "stat_features": stat_tf,
        "classifier": lr_clf,
        "classes": sorted(set(labels)),
        "preprocessor": preprocessor,
        "cv_macro_f1_mean": float(cv_f1.mean()),
        "cv_macro_f1_std":  float(cv_f1.std()),
        "cv_accuracy_mean": float(cv_acc.mean()),
        "model_version": "v4_sentence_transformer_nrc",
    }
    joblib.dump(model_data, MODEL_PATH, compress=3)

    elapsed = time.time() - t0
    print(f"\nModel saved to '{MODEL_PATH}'  ({elapsed:.1f}s total)")
    print(f"Estimated CV Accuracy: {cv_acc.mean() * 100:.2f}%  |  Macro-F1: {cv_f1.mean():.4f}")
    print("=" * 70)


if __name__ == "__main__":
    train_model()
