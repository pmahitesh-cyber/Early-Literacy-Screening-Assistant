"""
MFCC and Acoustic Feature Extraction Pipeline
Designed for Early Literacy Speech Assessment (DrivenData Goodnight Moon challenge).
Extracts MFCCs, Delta, Delta-Delta, and acoustic prosody features.
"""

import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.stats
import soundfile as sf
import os
from typing import Tuple, Dict, Any, Optional


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono, and ensure standard float32 format.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    data, sr = sf.read(file_path, dtype='float32')
    
    # Convert stereo to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    # Simple linear resampling if sample rate differs
    if sr != target_sr and target_sr is not None:
        num_samples = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num_samples)
        sr = target_sr
        
    # Peak normalization
    max_val = np.max(np.abs(data))
    if max_val > 1e-6:
        data = data / max_val
        
    return data.astype(np.float32), sr


def preemphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to amplify high frequencies."""
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])


def frame_signal(signal: np.ndarray, frame_len: int, frame_step: int, win_func=np.hamming) -> np.ndarray:
    """Slice 1D signal into overlapping frames with window function."""
    slen = len(signal)
    if slen < frame_len:
        # Pad short signals
        pad = np.zeros(frame_len - slen, dtype=signal.dtype)
        signal = np.concatenate([signal, pad])
        slen = len(signal)
        
    num_frames = 1 + int(np.ceil((slen - frame_len) / frame_step))
    pad_len = int((num_frames - 1) * frame_step + frame_len)
    
    if pad_len > slen:
        pad = np.zeros(pad_len - slen, dtype=signal.dtype)
        signal = np.concatenate([signal, pad])
        
    indices = (
        np.tile(np.arange(0, frame_len), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    )
    
    frames = signal[indices.astype(np.int32, copy=False)]
    if win_func is not None:
        window = win_func(frame_len)
        frames = frames * window
    return frames


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Convert Mel scale to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def get_filterbanks(num_filters: int = 26, nfft: int = 512, sample_rate: int = 16000,
                    low_freq: float = 0.0, high_freq: Optional[float] = None) -> np.ndarray:
    """Construct triangular Mel-filterbanks."""
    if high_freq is None:
        high_freq = sample_rate / 2.0
        
    low_mel = hz_to_mel(np.array([low_freq]))[0]
    high_mel = hz_to_mel(np.array([high_freq]))[0]
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)
    
    fbank = np.zeros((num_filters, int(np.floor(nfft / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        
        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - f_m)
                
    return fbank


def compute_mfcc(signal: np.ndarray, sample_rate: int = 16000,
                 win_size: float = 0.025, win_stride: float = 0.010,
                 num_cepstral: int = 13, num_filters: int = 26,
                 nfft: int = 512, preemph_coeff: float = 0.97,
                 cep_lifter: int = 22) -> np.ndarray:
    """
    Compute 2D array of MFCC coefficients over time frames: shape (num_frames, num_cepstral).
    """
    emphasized = preemphasis(signal, preemph_coeff)
    frame_len = int(round(win_size * sample_rate))
    frame_step = int(round(win_stride * sample_rate))
    
    frames = frame_signal(emphasized, frame_len, frame_step, np.hamming)
    
    # Power spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, nfft))
    pow_frames = ((1.0 / nfft) * (mag_frames ** 2))
    
    # Filterbanks
    fbanks = get_filterbanks(num_filters, nfft, sample_rate)
    filter_banks = np.dot(pow_frames, fbanks.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_fbanks = np.log(filter_banks)
    
    # Discrete Cosine Transform (DCT)
    mfcc = scipy.fftpack.dct(log_fbanks, type=2, axis=1, norm='ortho')[:, :num_cepstral]
    
    # Cepstral liftering
    if cep_lifter > 0:
        n = np.arange(num_cepstral)
        lift = 1.0 + (cep_lifter / 2.0) * np.sin(np.pi * n / cep_lifter)
        mfcc = mfcc * lift
        
    return mfcc


def compute_deltas(features: np.ndarray, n: int = 2) -> np.ndarray:
    """Compute 1st derivative (velocity) or 2nd derivative (acceleration) of features."""
    num_frames, dim = features.shape
    deltas = np.zeros_like(features)
    denom = 2 * sum(i ** 2 for i in range(1, n + 1))
    
    padded = np.pad(features, ((n, n), (0, 0)), mode='edge')
    for t in range(num_frames):
        d_val = np.zeros(dim)
        for i in range(1, n + 1):
            d_val += i * (padded[t + n + i] - padded[t + n - i])
        deltas[t] = d_val / denom
    return deltas


def extract_acoustic_prosody(signal: np.ndarray, sample_rate: int = 16000,
                             win_size: float = 0.025, win_stride: float = 0.010) -> Dict[str, float]:
    """
    Extract key literacy-specific acoustic markers:
    - Pause ratio / Hesitation index (long silences during reading)
    - Zero-Crossing Rate (ZCR)
    - Root-Mean-Square Energy (RMS)
    - Spectral Centroid (timbral brightness/articulation)
    - Spectral Rolloff (frequency energy distribution)
    - Duration of utterance
    """
    frame_len = int(round(win_size * sample_rate))
    frame_step = int(round(win_stride * sample_rate))
    frames = frame_signal(signal, frame_len, frame_step, None)
    
    # Frame RMS energy
    rms_energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    max_energy = np.max(rms_energy) if len(rms_energy) > 0 else 1.0
    rel_energy = rms_energy / (max_energy + 1e-8)
    
    # Silence / Pause ratio: frames with relative energy < 0.1
    silence_frames = np.sum(rel_energy < 0.1)
    total_frames = len(rel_energy)
    pause_ratio = float(silence_frames / total_frames) if total_frames > 0 else 0.0
    
    # Zero Crossing Rate
    zcr = np.mean(0.5 * np.abs(np.diff(np.sign(frames), axis=1)), axis=1)
    
    # Spectral Centroid & Rolloff
    mag_frames = np.absolute(np.fft.rfft(frames, 512))
    freqs = np.fft.rfftfreq(512, 1.0 / sample_rate)
    sum_mag = np.sum(mag_frames, axis=1) + 1e-10
    spectral_centroid = np.sum(mag_frames * freqs, axis=1) / sum_mag
    
    # Rolloff (85% energy point)
    cum_energy = np.cumsum(mag_frames, axis=1)
    rolloff_thresh = 0.85 * cum_energy[:, -1:]
    rolloff_bins = np.argmax(cum_energy >= rolloff_thresh, axis=1)
    spectral_rolloff = freqs[rolloff_bins]
    
    duration = float(len(signal) / sample_rate)
    
    return {
        'duration_sec': duration,
        'pause_ratio': pause_ratio,
        'mean_rms': float(np.mean(rms_energy)),
        'std_rms': float(np.std(rms_energy)),
        'mean_zcr': float(np.mean(zcr)),
        'std_zcr': float(np.std(zcr)),
        'mean_centroid': float(np.mean(spectral_centroid)),
        'std_centroid': float(np.std(spectral_centroid)),
        'mean_rolloff': float(np.mean(spectral_rolloff)),
        'std_rolloff': float(np.std(spectral_rolloff)),
    }


def extract_mfcc_feature_vector(signal: np.ndarray, sample_rate: int = 16000,
                                num_cepstral: int = 13) -> np.ndarray:
    """
    Extract full aggregated statistical feature vector:
    - MFCC (13 cepstrals): mean, std, skewness, min, max, p25, p75
    - Delta MFCC (13): mean, std, min, max
    - Delta-Delta MFCC (13): mean, std
    - Prosody/Acoustic descriptors (10)
    Total dimensions: 13*7 + 13*4 + 13*2 + 10 = 91 + 52 + 26 + 10 = 179 features.
    """
    mfcc = compute_mfcc(signal, sample_rate=sample_rate, num_cepstral=num_cepstral)
    delta = compute_deltas(mfcc, n=2)
    delta_delta = compute_deltas(delta, n=2)
    
    # Statistical pooling across time
    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)
    mfcc_skew = scipy.stats.skew(mfcc, axis=0)
    mfcc_min = np.min(mfcc, axis=0)
    mfcc_max = np.max(mfcc, axis=0)
    mfcc_p25 = np.percentile(mfcc, 25, axis=0)
    mfcc_p75 = np.percentile(mfcc, 75, axis=0)
    
    delta_mean = np.mean(delta, axis=0)
    delta_std = np.std(delta, axis=0)
    delta_min = np.min(delta, axis=0)
    delta_max = np.max(delta, axis=0)
    
    delta2_mean = np.mean(delta_delta, axis=0)
    delta2_std = np.std(delta_delta, axis=0)
    
    prosody = extract_acoustic_prosody(signal, sample_rate=sample_rate)
    prosody_vec = np.array(list(prosody.values()), dtype=np.float32)
    
    feat_vec = np.concatenate([
        mfcc_mean, mfcc_std, mfcc_skew, mfcc_min, mfcc_max, mfcc_p25, mfcc_p75,
        delta_mean, delta_std, delta_min, delta_max,
        delta2_mean, delta2_std,
        prosody_vec
    ])
    
    # Handle NaN or Inf values gracefully
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_vec.astype(np.float32)


def get_feature_names(num_cepstral: int = 13) -> list:
    """Generate human-readable names for all features."""
    names = []
    stats_mfcc = ['mean', 'std', 'skew', 'min', 'max', 'p25', 'p75']
    for stat in stats_mfcc:
        for i in range(num_cepstral):
            names.append(f"mfcc_{i}_{stat}")
            
    stats_delta = ['mean', 'std', 'min', 'max']
    for stat in stats_delta:
        for i in range(num_cepstral):
            names.append(f"delta_{i}_{stat}")
            
    stats_delta2 = ['mean', 'std']
    for stat in stats_delta2:
        for i in range(num_cepstral):
            names.append(f"delta2_{i}_{stat}")
            
    prosody_keys = [
        'duration_sec', 'pause_ratio', 'mean_rms', 'std_rms',
        'mean_zcr', 'std_zcr', 'mean_centroid', 'std_centroid',
        'mean_rolloff', 'std_rolloff'
    ]
    names.extend(prosody_keys)
    return names
