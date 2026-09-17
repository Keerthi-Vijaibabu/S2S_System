import soundfile as sf
from scipy.signal import resample_poly
import numpy as np

INPUT_FILE = "tamil_sample.flac"
OUTPUT_FILE = "tamil_sample_16k.wav"
TARGET_SR = 16000

# 1. Decode FLAC
audio, sample_rate = sf.read(INPUT_FILE, dtype="float32")

print("Original shape:", audio.shape)
print("Original sample rate:", sample_rate)

# 2. Stereo -> mono
if audio.ndim == 2:
    audio = audio.mean(axis=1)

print("Mono shape:", audio.shape)

# 3. Resample to 16 kHz
if sample_rate != TARGET_SR:
    from math import gcd

    g = gcd(sample_rate, TARGET_SR)

    audio = resample_poly(
        audio,
        TARGET_SR // g,
        sample_rate // g
    ).astype(np.float32)

# 4. Save as WAV
sf.write(
    OUTPUT_FILE,
    audio,
    TARGET_SR,
    subtype="PCM_16"
)

print("\nFinal shape:", audio.shape)
print("Final sample rate:", TARGET_SR)
print("Duration:", len(audio) / TARGET_SR)
print("Saved:", OUTPUT_FILE)