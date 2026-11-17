"""
Model Training, Cross-Validation, and Evaluation Pipeline
Evaluates on Held-Out Speakers using GroupKFold.
Benchmarks:
1. Majority Class Baseline
2. Handcrafted MFCC + Prosodic Features (Logistic Regression & HistGradientBoosting)
3. Pretrained Speech Embeddings
4. Multimodal Hybrid Fusion (MFCC + Pretrained Embeddings + Metadata)
Reports Log Loss and ROC-AUC across all models.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.audio_features import load_audio, extract_mfcc_feature_vector, get_feature_names
from src.speech_embeddings import extract_speech_embedding


def extract_dataset_features(data_dir: str = "e:/Mahi Proj/data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load dataset and extract both MFCC and Pretrained Speech Embeddings for all clips.
    Returns:
    - X_mfcc: (N, 179) array of acoustic MFCC features
    - X_emb: (N, 128) array of pretrained neural speech embeddings
    - y: (N,) binary targets (1=correct, 0=incorrect)
    - df: DataFrame containing metadata (filename, student_id, task, grade, expected_text)
    """
    metadata_path = os.path.join(data_dir, "train_metadata.csv")
    labels_path = os.path.join(data_dir, "train_labels.csv")
    audio_dir = os.path.join(data_dir, "audio")
    
    if not os.path.exists(metadata_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("Dataset files missing. Run dataset_generator.py first.")
        
    df_meta = pd.read_csv(metadata_path)
    df_labels = pd.read_csv(labels_path)
    df = pd.merge(df_meta, df_labels, on="filename")
    
    cache_file = os.path.join(data_dir, "extracted_features_cache.npz")
    if os.path.exists(cache_file):
        print(f"Loading cached features from {cache_file}...")
        cached = np.load(cache_file, allow_pickle=True)
        return cached['X_mfcc'], cached['X_emb'], cached['y'], df
        
    print(f"Extracting features from {len(df)} audio files...")
    mfcc_list = []
    emb_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row['filename'])
        if not os.path.exists(audio_path):
            continue
            
        signal, sr = load_audio(audio_path, target_sr=16000)
        
        # 1. MFCC features
        mfcc_feats = extract_mfcc_feature_vector(signal, sample_rate=sr)
        
        # 2. Pretrained speech embeddings
        speech_emb = extract_speech_embedding(signal, sample_rate=sr)
        
        mfcc_list.append(mfcc_feats)
        emb_list.append(speech_emb)
        valid_indices.append(idx)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
            print(f"Processed {idx + 1}/{len(df)} clips...")
            
    X_mfcc = np.array(mfcc_list, dtype=np.float32)
    X_emb = np.array(emb_list, dtype=np.float32)
    y = df.loc[valid_indices, 'score'].values.astype(int)
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    
    np.savez_compressed(cache_file, X_mfcc=X_mfcc, X_emb=X_emb, y=y)
    print(f"Features extracted and cached: MFCC={X_mfcc.shape}, Pretrained={X_emb.shape}")
    return X_mfcc, X_emb, y, df_valid


def evaluate_models_on_held_out_speakers(
    X_mfcc: np.ndarray,
    X_emb: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform GroupKFold cross-validation grouped strictly by student_id (held-out speakers).
    Evaluates:
    - Baseline: Majority Class
    - Model 1: MFCC + Logistic Regression (L2 regularized)
    - Model 2: MFCC + HistGradientBoosting
    - Model 3: Pretrained Speech Embedding + Logistic Regression
    - Model 4: Pretrained Speech Embedding + HistGradientBoosting
    - Model 5: Hybrid Multimodal (MFCC + Pretrained + Metadata)
    """
    groups = df['student_id'].values
    gkf = GroupKFold(n_splits=n_splits)
    
    # Metadata one-hot features (task and grade)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    meta_feats = encoder.fit_transform(df[['task', 'grade']])
    
    # Combined hybrid features
    X_hybrid = np.hstack([X_mfcc, X_emb, meta_feats])
    
    model_configs = {
        "Majority Class Baseline": None,
        "MFCC (LogReg)": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, max_iter=500, random_state=random_state))]),
        "MFCC (GradientBoost)": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, random_state=random_state),
        "Pretrained Embedding (LogReg)": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1.0, max_iter=500, random_state=random_state))]),
        "Pretrained Embedding (GradientBoost)": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, random_state=random_state),
        "Hybrid Multimodal (Ensemble)": HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, random_state=random_state),
    }
    
    predictions = {name: np.zeros(len(y), dtype=float) for name in model_configs}
    fold_metrics = {name: {"log_loss": [], "roc_auc": [], "accuracy": [], "balanced_acc": [], "f1": []} for name in model_configs}
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_mfcc, y, groups)):
        y_train, y_val = y[train_idx], y[val_idx]
        train_speakers = set(groups[train_idx])
        val_speakers = set(groups[val_idx])
        assert len(train_speakers.intersection(val_speakers)) == 0, "Speaker leakage detected in fold!"
        
        # 1. Majority Class Baseline
        majority_prior = np.mean(y_train)
        pred_majority = np.full(len(y_val), majority_prior)
        predictions["Majority Class Baseline"][val_idx] = pred_majority
        
        # 2. MFCC LogReg
        clf_mfcc_lr = model_configs["MFCC (LogReg)"]
        clf_mfcc_lr.fit(X_mfcc[train_idx], y_train)
        pred_mfcc_lr = clf_mfcc_lr.predict_proba(X_mfcc[val_idx])[:, 1]
        predictions["MFCC (LogReg)"][val_idx] = pred_mfcc_lr
        
        # 3. MFCC GradientBoost
        clf_mfcc_gb = model_configs["MFCC (GradientBoost)"]
        clf_mfcc_gb.fit(X_mfcc[train_idx], y_train)
        pred_mfcc_gb = clf_mfcc_gb.predict_proba(X_mfcc[val_idx])[:, 1]
        predictions["MFCC (GradientBoost)"][val_idx] = pred_mfcc_gb
        
        # 4. Pretrained Embedding LogReg
        clf_emb_lr = model_configs["Pretrained Embedding (LogReg)"]
        clf_emb_lr.fit(X_emb[train_idx], y_train)
        pred_emb_lr = clf_emb_lr.predict_proba(X_emb[val_idx])[:, 1]
        predictions["Pretrained Embedding (LogReg)"][val_idx] = pred_emb_lr
        
        # 5. Pretrained Embedding GradientBoost
        clf_emb_gb = model_configs["Pretrained Embedding (GradientBoost)"]
        clf_emb_gb.fit(X_emb[train_idx], y_train)
        pred_emb_gb = clf_emb_gb.predict_proba(X_emb[val_idx])[:, 1]
        predictions["Pretrained Embedding (GradientBoost)"][val_idx] = pred_emb_gb
        
        # 6. Hybrid Multimodal
        clf_hybrid = model_configs["Hybrid Multimodal (Ensemble)"]
        clf_hybrid.fit(X_hybrid[train_idx], y_train)
        pred_hybrid = clf_hybrid.predict_proba(X_hybrid[val_idx])[:, 1]
        predictions["Hybrid Multimodal (Ensemble)"][val_idx] = pred_hybrid
        
        # Compute fold-level metrics
        fold_preds = {
            "Majority Class Baseline": pred_majority,
            "MFCC (LogReg)": pred_mfcc_lr,
            "MFCC (GradientBoost)": pred_mfcc_gb,
            "Pretrained Embedding (LogReg)": pred_emb_lr,
            "Pretrained Embedding (GradientBoost)": pred_emb_gb,
            "Hybrid Multimodal (Ensemble)": pred_hybrid,
        }
        
        for name, p in fold_preds.items():
            p_clipped = np.clip(p, 1e-15, 1 - 1e-15)
            ll = log_loss(y_val, p_clipped)
            try:
                auc = roc_auc_score(y_val, p) if len(np.unique(y_val)) > 1 else 0.5
            except Exception:
                auc = 0.5
            pred_binary = (p >= 0.5).astype(int)
            acc = accuracy_score(y_val, pred_binary)
            bacc = balanced_accuracy_score(y_val, pred_binary)
            f1 = f1_score(y_val, pred_binary, zero_division=0)
            
            fold_metrics[name]["log_loss"].append(ll)
            fold_metrics[name]["roc_auc"].append(auc)
            fold_metrics[name]["accuracy"].append(acc)
            fold_metrics[name]["balanced_acc"].append(bacc)
            fold_metrics[name]["f1"].append(f1)
            
    # Compute overall out-of-fold metrics
    results = {}
    print("\n" + "=" * 80)
    print("HELD-OUT SPEAKER CROSS-VALIDATION RESULTS (GroupKFold on student_id)")
    print("=" * 80)
    print(f"{'Model Name':<32} | {'Log Loss':<10} | {'ROC-AUC':<9} | {'Accuracy':<9} | {'Bal Acc':<9} | {'F1-Score':<8}")
    print("-" * 80)
    
    for name in model_configs:
        p_oof = predictions[name]
        p_clipped = np.clip(p_oof, 1e-15, 1 - 1e-15)
        oof_ll = log_loss(y, p_clipped)
        try:
            oof_auc = roc_auc_score(y, p_oof)
        except Exception:
            oof_auc = 0.5
        p_binary = (p_oof >= 0.5).astype(int)
        oof_acc = accuracy_score(y, p_binary)
        oof_bacc = balanced_accuracy_score(y, p_binary)
        oof_f1 = f1_score(y, p_binary, zero_division=0)
        
        mean_ll = float(np.mean(fold_metrics[name]["log_loss"]))
        std_ll = float(np.std(fold_metrics[name]["log_loss"]))
        mean_auc = float(np.mean(fold_metrics[name]["roc_auc"]))
        std_auc = float(np.std(fold_metrics[name]["roc_auc"]))
        
        results[name] = {
            "oof_log_loss": round(float(oof_ll), 4),
            "oof_roc_auc": round(float(oof_auc), 4),
            "oof_accuracy": round(float(oof_acc), 4),
            "oof_balanced_acc": round(float(oof_bacc), 4),
            "oof_f1": round(float(oof_f1), 4),
            "cv_mean_log_loss": round(mean_ll, 4),
            "cv_std_log_loss": round(std_ll, 4),
            "cv_mean_roc_auc": round(mean_auc, 4),
            "cv_std_roc_auc": round(std_auc, 4),
            "confusion_matrix": confusion_matrix(y, p_binary).tolist()
        }
        
        print(f"{name:<32} | {oof_ll:.4f}     | {oof_auc:.4f}    | {oof_acc:.4f}    | {oof_bacc:.4f}    | {oof_f1:.4f}")
        
    print("=" * 80 + "\n")
    return results, predictions


def train_and_save_production_pipeline(output_models_dir: str = "e:/Mahi Proj/models",
                                       data_dir: str = "e:/Mahi Proj/data"):
    """
    Train final models on all data and save pipeline artifacts for the teacher dashboard.
    """
    os.makedirs(output_models_dir, exist_ok=True)
    os.makedirs("e:/Mahi Proj/reports", exist_ok=True)
    
    X_mfcc, X_emb, y, df = extract_dataset_features(data_dir=data_dir)
    
    # 1. Run held-out speaker benchmark
    metrics, oof_preds = evaluate_models_on_held_out_speakers(X_mfcc, X_emb, y, df)
    
    # Save metrics report to JSON
    report_path = "e:/Mahi Proj/reports/evaluation_metrics.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {report_path}")
    
    # 2. Fit production models on full dataset
    print("Fitting production models on full training dataset...")
    # Pipeline 1: Standalone Handcrafted MFCC Scorer
    mfcc_scaler = StandardScaler()
    X_mfcc_scaled = mfcc_scaler.fit_transform(X_mfcc)
    
    mfcc_clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=42)
    mfcc_clf.fit(X_mfcc, y)
    
    mfcc_lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    mfcc_lr.fit(X_mfcc_scaled, y)
    
    # Pipeline 2: Pretrained Speech Embedding Scorer
    emb_scaler = StandardScaler()
    X_emb_scaled = emb_scaler.fit_transform(X_emb)
    emb_clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=42)
    emb_clf.fit(X_emb, y)
    
    # Pipeline 3: Hybrid Multimodal Model
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    meta_feats = encoder.fit_transform(df[['task', 'grade']])
    X_hybrid = np.hstack([X_mfcc, X_emb, meta_feats])
    hybrid_clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, random_state=42)
    hybrid_clf.fit(X_hybrid, y)
    
    # Save artifacts
    artifacts = {
        "mfcc_clf": mfcc_clf,
        "mfcc_lr": mfcc_lr,
        "mfcc_scaler": mfcc_scaler,
        "emb_clf": emb_clf,
        "emb_scaler": emb_scaler,
        "hybrid_clf": hybrid_clf,
        "meta_encoder": encoder,
        "majority_prior": float(np.mean(y)),
        "feature_names": get_feature_names(),
        "tasks": list(df['task'].unique()),
        "grades": list(df['grade'].unique()),
    }
    
    model_save_path = os.path.join(output_models_dir, "literacy_screener_models.joblib")
    joblib.dump(artifacts, model_save_path)
    print(f"Production models successfully saved to {model_save_path}")
    
    # Generate teacher cohort report from out-of-fold evaluations
    df_cohort = df.copy()
    df_cohort['predicted_prob_mfcc'] = oof_preds["MFCC (GradientBoost)"]
    df_cohort['predicted_prob_pretrained'] = oof_preds["Pretrained Embedding (GradientBoost)"]
    df_cohort['predicted_prob_hybrid'] = oof_preds["Hybrid Multimodal (Ensemble)"]
    
    # Triage flag logic
    def get_flag(p):
        if p >= 0.70:
            return "Proficient"
        elif p >= 0.45:
            return "Monitor / Borderline"
        else:
            return "Flagged for Teacher Review"
            
    df_cohort['screening_status'] = df_cohort['predicted_prob_hybrid'].apply(get_flag)
    cohort_path = "e:/Mahi Proj/reports/screening_summary.csv"
    df_cohort.to_csv(cohort_path, index=False)
    print(f"Teacher cohort summary saved to {cohort_path}")
    
    return metrics


if __name__ == "__main__":
    train_and_save_production_pipeline()
