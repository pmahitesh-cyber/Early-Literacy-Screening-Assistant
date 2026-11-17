"""
Dataset Generator & Loader for DrivenData Goodnight Moon Literacy Screening
Generates realistic audio clips, metadata, and ground truth labels
strictly conforming to the competition schema.
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal
from typing import Tuple, List, Dict


# Literacy screening subtasks & sample target items
SUBTASK_ITEMS = {
    "deletion": [
        ("cupcake without cup", "cake"),
        ("sunshine without sun", "shine"),
        ("starfish without star", "fish"),
        ("meatball without meat", "ball"),
        ("seat without /s/", "eat"),
        ("farm without /f/", "arm"),
        ("stop without /s/", "top"),
        ("clap without /k/", "lap"),
    ],
    "blending": [
        ("/s/ /u/ /n/", "sun"),
        ("/c/ /a/ /t/", "cat"),
        ("/m/ /o/ /p/", "mop"),
        ("/f/ /r/ /o/ /g/", "frog"),
        ("/b/ /e/ /s/ /t/", "best"),
        ("/s/ /t/ /o/ /p/", "stop"),
        ("/p/ /l/ /a/ /n/ /t/", "plant"),
        ("/t/ /r/ /a/ /i/ /n/", "train"),
    ],
    "nonword_repetition": [
        ("drap", "drap"),
        ("plip", "plip"),
        ("vond", "vond"),
        ("zarp", "zarp"),
        ("teff", "teff"),
        ("misk", "misk"),
        ("gloob", "gloob"),
        ("trant", "trant"),
    ],
    "sentence_repetition": [
        ("The dog ran fast", "The dog ran fast"),
        ("She likes green apples", "She likes green apples"),
        ("Birds fly in the sky", "Birds fly in the sky"),
        ("We played at the park", "We played at the park"),
        ("The yellow kitten is sleeping", "The yellow kitten is sleeping"),
        ("He built a tall sandcastle", "He built a tall sandcastle"),
    ]
}

GRADES = ["KG", "1", "2", "3"]


def synthesize_child_speech(expected_text: str, is_correct: bool, speaker_f0: float,
                            sample_rate: int = 16000, duration: float = 2.0,
                            rng: np.random.RandomState = None) -> np.ndarray:
    """
    Synthesize realistic acoustic waveforms reflecting child speech characteristics:
    - Fundamental frequency (F0) typical for children (220-350 Hz)
    - Formant resonance filters (F1, F2, F3) for vowels
    - Fricative noise bursts for consonants
    - Fluency variations:
        * Correct: fluent speech, minimal pause, consistent formant modulation
        * Incorrect/Struggling: long hesitation pauses, lower signal energy, spectral instability
    """
    if rng is None:
        rng = np.random.RandomState(42)
        
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Base glottal pulse train simulation (harmonics of F0)
    signal = np.zeros(num_samples)
    for harmonic in range(1, 6):
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * harmonic * speaker_f0 * t)
        
    # Temporal envelope (syllable rhythm)
    num_syllables = max(1, len(expected_text.split()))
    syllable_rate = num_syllables / (duration * 0.7)
    envelope = np.abs(np.sin(2 * np.pi * syllable_rate * t)) ** 1.5
    
    # Apply formant filtering (simulating vocal tract resonances: e.g. F1=600Hz, F2=1500Hz, F3=2800Hz)
    nyq = 0.5 * sample_rate
    b_f1, a_f1 = scipy.signal.iirpeak(650.0 / nyq, Q=5.0)
    b_f2, a_f2 = scipy.signal.iirpeak(1600.0 / nyq, Q=6.0)
    b_f3, a_f3 = scipy.signal.iirpeak(2800.0 / nyq, Q=8.0)
    
    signal = scipy.signal.lfilter(b_f1, a_f1, signal) + 0.6 * scipy.signal.lfilter(b_f2, a_f2, signal) + 0.3 * scipy.signal.lfilter(b_f3, a_f3, signal)
    signal = signal * envelope
    
    if is_correct:
        # Fluent response: prompt onset, articulate, brief closing silence
        # Moderate SNR, slight natural acoustic jitter
        jitter = 1.0 + 0.03 * rng.randn(num_samples)
        signal = signal * jitter
        # High SNR background ambient noise
        ambient = 0.02 * rng.randn(num_samples)
        audio = signal + ambient
    else:
        # Reading difficulty / phonemic error:
        # High hesitation: zero out a significant chunk in the middle (hesitation/stall)
        pause_start = int(0.25 * num_samples)
        pause_end = int(0.65 * num_samples)
        signal[pause_start:pause_end] *= 0.02 # silence/pause
        # Spectral distortion / unvoiced hesitation noise / mispronunciation
        noise_burst = 0.15 * rng.randn(num_samples) * (1.0 - envelope)
        audio = 0.7 * signal + noise_burst
        
    # Peak normalize
    max_amp = np.max(np.abs(audio))
    if max_amp > 1e-5:
        audio = 0.9 * (audio / max_amp)
        
    return audio.astype(np.float32)


def generate_literacy_dataset(output_dir: str = "e:/Mahi Proj/data",
                              num_students: int = 24,
                              items_per_student: int = 12,
                              random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic benchmark dataset matching DrivenData Goodnight Moon specifications.
    Saves:
    - {output_dir}/audio/*.wav
    - {output_dir}/train_metadata.csv
    - {output_dir}/train_labels.csv
    """
    rng = np.random.RandomState(random_state)
    os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
    
    metadata_rows = []
    labels_rows = []
    
    tasks = list(SUBTASK_ITEMS.keys())
    
    # Assign each student an ID, grade level, and unique fundamental frequency (pitch)
    students = []
    for s_idx in range(num_students):
        student_id = f"student_{s_idx+1:03d}"
        grade = GRADES[s_idx % len(GRADES)]
        # Younger children (KG) have higher pitch, older (G3) slightly lower
        base_f0 = 300.0 if grade in ["KG", "1"] else 240.0
        speaker_f0 = base_f0 + rng.uniform(-30.0, 30.0)
        # Base student reading proficiency (probability of answering correctly)
        # e.g., 30% of students have reading difficulties (struggling readers)
        is_struggling_reader = (s_idx % 3 == 0)
        prof_prob = 0.35 if is_struggling_reader else 0.85
        students.append({
            "student_id": student_id,
            "grade": grade,
            "speaker_f0": speaker_f0,
            "prof_prob": prof_prob
        })
        
    sample_counter = 1
    for student in students:
        for _ in range(items_per_student):
            task = rng.choice(tasks)
            task_items = SUBTASK_ITEMS[task]
            item_prompt, expected_text = task_items[rng.randint(len(task_items))]
            
            # Determine correctness based on student proficiency
            is_correct = int(rng.rand() < student["prof_prob"])
            
            filename = f"clip_{sample_counter:04d}.wav"
            file_path = os.path.join(output_dir, "audio", filename)
            
            # Synthesize audio
            audio = synthesize_child_speech(
                expected_text=expected_text,
                is_correct=bool(is_correct),
                speaker_f0=student["speaker_f0"],
                sample_rate=16000,
                duration=rng.uniform(1.8, 3.2),
                rng=rng
            )
            
            sf.write(file_path, audio, 16000)
            
            metadata_rows.append({
                "filename": filename,
                "task": task,
                "expected_text": expected_text,
                "grade": student["grade"],
                "student_id": student["student_id"]
            })
            
            labels_rows.append({
                "filename": filename,
                "score": is_correct
            })
            
            sample_counter += 1
            
    df_metadata = pd.DataFrame(metadata_rows)
    df_labels = pd.DataFrame(labels_rows)
    
    metadata_path = os.path.join(output_dir, "train_metadata.csv")
    labels_path = os.path.join(output_dir, "train_labels.csv")
    
    df_metadata.to_csv(metadata_path, index=False)
    df_labels.to_csv(labels_path, index=False)
    
    print(f"Dataset generated successfully at {output_dir}:")
    print(f"Total audio clips: {len(df_metadata)}")
    print(f"Unique students: {df_metadata['student_id'].nunique()}")
    print(f"Class distribution: {df_labels['score'].value_counts(normalize=True).to_dict()}")
    
    return df_metadata, df_labels


if __name__ == "__main__":
    generate_literacy_dataset()
