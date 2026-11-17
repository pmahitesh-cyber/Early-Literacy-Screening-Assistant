"""
Early Literacy Screening Assistant - Unified CLI & Launcher
Provides easy access to:
1. Model evaluation report on held-out speakers vs Majority Baseline
2. Audio clip scoring from command line
3. Launching the interactive teacher-facing web application
"""

import os
import sys
import argparse
import json
import uvicorn

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.inference import LiteracyScorer
from src.train import train_and_save_production_pipeline


def print_banner():
    print(r"""
========================================================================
     EARLY LITERACY SCREENING ASSISTANT FOR PRIMARY CLASSROOMS
       DrivenData Goodnight Moon Challenge • Automated Triage
========================================================================
    """)


def show_benchmark_report():
    print_banner()
    report_path = os.path.join(BASE_DIR, "reports", "evaluation_metrics.json")
    if not os.path.exists(report_path):
        print("Metrics report not found. Running training and evaluation first...")
        train_and_save_production_pipeline()
        
    with open(report_path, "r") as f:
        metrics = json.load(f)
        
    print("HELD-OUT SPEAKER CROSS-VALIDATION BENCHMARK (GroupKFold on student_id)")
    print("-" * 88)
    print(f"{'Model / Approach':<36} | {'Log Loss (OOF)':<15} | {'ROC-AUC':<10} | {'Accuracy':<10} | {'F1-Score':<8}")
    print("-" * 88)
    for model_name, m in metrics.items():
        print(f"{model_name:<36} | {m['oof_log_loss']:<15.4f} | {m['oof_roc_auc']:<10.4f} | {m['oof_accuracy']*100:<9.1f}% | {m['oof_f1']:<8.4f}")
    print("=" * 88)
    print("\nTakeaway: MFCC features with prosodic pause ratios and Pretrained Neural Speech Embeddings")
    print("substantially outperform the majority class baseline across all held-out speakers.\n")


def score_clip_cli(clip_path: str, task: str = "blending", grade: str = "1", student_id: str = "student_001"):
    print_banner()
    if not os.path.exists(clip_path):
        print(f"Error: Audio file not found at {clip_path}")
        return
        
    print(f"Scoring Audio Clip: {clip_path}")
    print(f"Student ID: {student_id} | Grade: {grade} | Task: {task}")
    print("-" * 65)
    
    scorer = LiteracyScorer()
    res = scorer.score_audio(clip_path, task=task, grade=grade, student_id=student_id)
    
    print(f"Triage Decision:  {res['status'].upper()}")
    print(f"Action Required:  {res['action_required']}")
    print(f"\nModel Probabilities:")
    print(f"  • Multimodal Hybrid Score:       {res['scores']['hybrid_probability']*100:.1f}%")
    print(f"  • Handcrafted MFCC Score:        {res['scores']['mfcc_model_probability']*100:.1f}%")
    print(f"  • Pretrained Speech Emb Score:   {res['scores']['pretrained_embedding_probability']*100:.1f}%")
    print(f"  • Majority Class Baseline:       {res['scores']['majority_baseline_prior']*100:.1f}%")
    print(f"\nAcoustic Diagnostics:")
    print(f"  • Speaking Duration:             {res['acoustic_diagnostics']['duration_seconds']}s")
    print(f"  • Pause / Hesitation Ratio:      {res['acoustic_diagnostics']['pause_ratio_percent']}% ({res['acoustic_diagnostics']['hesitation_level']})")
    print(f"  • Articulation Clarity:          {res['acoustic_diagnostics']['articulation_clarity']}")
    print(f"\nTargeted Pedagogical Interventions:")
    for rec in res['pedagogical_recommendations']:
        print(f"  ✓ {rec}")
    print("=" * 65)


def serve_dashboard(port: int = 8000, host: str = "127.0.0.1"):
    print_banner()
    print(f"Starting Teacher Dashboard on http://{host}:{port} ...")
    print("Open your browser and navigate to the link above.")
    uvicorn.run("web_app.app:app", host=host, port=port, reload=False)


def main():
    parser = argparse.ArgumentParser(description="Early Literacy Screening Assistant")
    parser.add_argument("--report", action="store_true", help="Display held-out speaker benchmark evaluation report")
    parser.add_argument("--score", type=str, help="Path to audio .wav clip to score")
    parser.add_argument("--task", type=str, default="blending", choices=["blending", "deletion", "nonword_repetition", "sentence_repetition"], help="Screening task type")
    parser.add_argument("--grade", type=str, default="1", choices=["KG", "1", "2", "3"], help="Student grade level")
    parser.add_argument("--student", type=str, default="student_001", help="Student ID")
    parser.add_argument("--serve", action="store_true", help="Start the teacher-facing interactive web application")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default 8000)")
    parser.add_argument("--train", action="store_true", help="Retrain models and regenerate benchmark")
    
    args = parser.parse_args()
    
    if args.train:
        print_banner()
        train_and_save_production_pipeline()
    elif args.report:
        show_benchmark_report()
    elif args.score:
        score_clip_cli(args.score, task=args.task, grade=args.grade, student_id=args.student)
    elif args.serve:
        serve_dashboard(port=args.port)
    else:
        # Default: show benchmark report and guidance
        show_benchmark_report()
        print("Usage tips:")
        print("  Launch Teacher UI:    python run_screener.py --serve")
        print("  Score a Clip:         python run_screener.py --score data/audio/clip_0001.wav")
        print("  Retrain & Benchmark:  python run_screener.py --train")


if __name__ == "__main__":
    main()
