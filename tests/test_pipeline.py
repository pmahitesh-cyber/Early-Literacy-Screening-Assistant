"""
Unit and Integration Tests for Early Literacy Screening Assistant Pipeline
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio_features import (
    compute_mfcc, compute_deltas, extract_acoustic_prosody,
    extract_mfcc_feature_vector, get_feature_names
)
from src.speech_embeddings import extract_speech_embedding, get_pretrained_speech_encoder
from src.inference import LiteracyScorer


def test_mfcc_feature_dimensions():
    """Verify that MFCC extraction returns exact expected feature vector shape."""
    sr = 16000
    t = np.linspace(0, 1.5, int(1.5 * sr), endpoint=False)
    synthetic_signal = 0.5 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.random.randn(len(t))
    
    feats = extract_mfcc_feature_vector(synthetic_signal, sample_rate=sr, num_cepstral=13)
    names = get_feature_names(num_cepstral=13)
    
    assert len(feats) == 179
    assert len(names) == 179
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()


def test_prosody_features():
    """Verify pause ratio and acoustic prosody descriptors."""
    sr = 16000
    # Create signal with 50% silence
    signal = np.ones(sr)
    signal[int(sr * 0.5):] = 0.0
    
    prosody = extract_acoustic_prosody(signal, sample_rate=sr)
    assert "pause_ratio" in prosody
    assert "mean_centroid" in prosody
    assert "duration_sec" in prosody
    assert prosody["duration_sec"] == 1.0
    assert 0.4 <= prosody["pause_ratio"] <= 0.6


def test_speech_embeddings():
    """Verify pretrained speech embedding dimensions and L2 normalization."""
    sr = 16000
    signal = np.random.randn(sr * 2).astype(np.float32)
    emb = extract_speech_embedding(signal, sample_rate=sr)
    
    assert emb.shape == (128,)
    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-3)


def test_dataset_schema():
    """Verify metadata and ground truth labels match DrivenData schema."""
    data_dir = "e:/Mahi Proj/data"
    meta_path = os.path.join(data_dir, "train_metadata.csv")
    labels_path = os.path.join(data_dir, "train_labels.csv")
    
    assert os.path.exists(meta_path)
    assert os.path.exists(labels_path)
    
    df_meta = pd.read_csv(meta_path)
    df_labels = pd.read_csv(labels_path)
    
    expected_meta_cols = {"filename", "task", "expected_text", "grade", "student_id"}
    expected_label_cols = {"filename", "score"}
    
    assert expected_meta_cols.issubset(set(df_meta.columns))
    assert expected_label_cols.issubset(set(df_labels.columns))
    assert len(df_meta) == len(df_labels)


def test_held_out_speaker_integrity():
    """Verify that train and validation sets have mutually exclusive student IDs."""
    data_dir = "e:/Mahi Proj/data"
    df_meta = pd.read_csv(os.path.join(data_dir, "train_metadata.csv"))
    
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    groups = df_meta['student_id'].values
    
    for train_idx, val_idx in gkf.split(df_meta, groups=groups):
        train_speakers = set(groups[train_idx])
        val_speakers = set(groups[val_idx])
        # Intersection must be completely empty!
        assert len(train_speakers.intersection(val_speakers)) == 0


def test_scorer_inference():
    """Test full inference pipeline with real clip."""
    audio_path = "e:/Mahi Proj/data/audio/clip_0001.wav"
    scorer = LiteracyScorer()
    res = scorer.score_audio(audio_path, task="blending", grade="1", student_id="student_test")
    
    assert "status" in res
    assert "status_code" in res
    assert res["status_code"] in ["proficient", "borderline", "flagged"]
    assert "scores" in res
    assert "hybrid_probability" in res["scores"]
    assert 0.0 <= res["scores"]["hybrid_probability"] <= 1.0
    assert len(res["pedagogical_recommendations"]) > 0
    assert len(res["spectrogram_base64"]) > 1000
