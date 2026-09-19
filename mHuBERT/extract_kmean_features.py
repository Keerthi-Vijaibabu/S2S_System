import io
import json
import random
from math import gcd

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset, Audio
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "utter-project/mHuBERT-147"
SPLIT_FILE = "indicvoices_tamil_split_full.json"

TARGET_SR = 16000

# Test run first
MAX_UTTERANCES = 10_000

# Keep at most this many frames from EACH utterance.
# This prevents long utterances from dominating the sample.
FRAMES_PER_UTTERANCE = 50

SEED = 42

OUTPUT_FILE = "tamil_kmeans_features_test.npy"

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD TRAIN SPEAKERS
# ============================================================

print("\nLoading speaker split...")

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    splits = json.load(f)

train_speakers = {
    item["speaker_id"]
    for item in splits["train"]
}

print("Training speakers:", len(train_speakers))


# ============================================================
# LOAD mHuBERT
# ============================================================

print("\nLoading mHuBERT-147...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(device)
model.eval()

print("mHuBERT loaded")


# ============================================================
# LOAD INDICVOICES
# ============================================================

print("\nLoading IndicVoices Tamil stream...")

dataset = load_dataset(
    "ai4bharat/IndicVoices",
    "tamil",
    split="train",
    streaming=True
)

dataset = dataset.cast_column(
    "audio_filepath",
    Audio(decode=False)
)

print("Dataset ready")


# ============================================================
# AUDIO DECODER
# ============================================================

def decode_audio(audio_bytes):

    audio, sr = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    # Stereo -> mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Resample -> 16 kHz
    if sr != TARGET_SR:

        g = gcd(sr, TARGET_SR)

        audio = resample_poly(
            audio,
            TARGET_SR // g,
            sr // g
        ).astype(np.float32)

    return audio


# ============================================================
# FEATURE EXTRACTION
# ============================================================

sampled_features = []

processed = 0
skipped = 0

print("\nStarting feature extraction...\n")

for sample in dataset:

    # Ignore validation/test speakers
    if sample["speaker_id"] not in train_speakers:
        continue

    try:

        audio_bytes = sample["audio_filepath"]["bytes"]

        if audio_bytes is None:
            skipped += 1
            continue

        audio = decode_audio(audio_bytes)

        inputs = feature_extractor(
            audio,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        )

        input_values = inputs.input_values.to(device)

        with torch.inference_mode():

            outputs = model(
                input_values,
                output_hidden_states=True
            )

        # Transformer Layer 11
        features = (
            outputs.hidden_states[11]
            .squeeze(0)
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Randomly sample frames from this utterance
        # ----------------------------------------------------

        n_frames = features.shape[0]

        take = min(
            FRAMES_PER_UTTERANCE,
            n_frames
        )

        indices = np.random.choice(
            n_frames,
            size=take,
            replace=False
        )

        sampled_features.append(
            features[indices]
        )

        processed += 1

        if processed % 100 == 0:

            total_frames = sum(
                x.shape[0]
                for x in sampled_features
            )

            print(
                f"Processed: {processed:,} | "
                f"Frames collected: {total_frames:,} | "
                f"Skipped: {skipped:,}"
            )

        if processed >= MAX_UTTERANCES:
            break

    except Exception as e:

        skipped += 1

        print(
            f"Skipping utterance due to error: {e}"
        )


# ============================================================
# COMBINE FEATURES
# ============================================================

print("\nCombining sampled features...")

features = np.concatenate(
    sampled_features,
    axis=0
).astype(np.float32)


# ============================================================
# VALIDATION
# ============================================================

print("\n-----------------------------")
print("FEATURE STATISTICS")
print("-----------------------------")

print("Utterances processed:", processed)
print("Utterances skipped:", skipped)
print("Feature matrix:", features.shape)
print("dtype:", features.dtype)

print("NaN:", np.isnan(features).sum())
print("Inf:", np.isinf(features).sum())

print("Mean:", features.mean())
print("Std:", features.std())


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_FILE,
    features
)

print("\nSaved:", OUTPUT_FILE)