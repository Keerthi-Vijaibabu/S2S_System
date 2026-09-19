import io

import numpy as np
import soundfile as sf
import torch

from datasets import load_dataset, Audio
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModel
from math import gcd


MODEL_NAME = "utter-project/mHuBERT-147"
NUM_UTTERANCES = 100
TARGET_SR = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# --------------------------------------------------
# 1. Load mHuBERT
# --------------------------------------------------

print("Loading mHuBERT-147...")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

print("Model loaded")


# --------------------------------------------------
# 2. Load IndicVoices Tamil
# --------------------------------------------------

print("\nLoading IndicVoices Tamil...")

dataset = load_dataset(
    "ai4bharat/IndicVoices",
    "tamil",
    split="train",
    streaming=True
)

# IMPORTANT:
# Don't let Hugging Face/TorchCodec decode the audio.
dataset = dataset.cast_column(
    "audio_filepath",
    Audio(decode=False)
)

print("Dataset loaded")


# --------------------------------------------------
# 3. Audio preprocessing
# --------------------------------------------------

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


# --------------------------------------------------
# 4. Extract mHuBERT features
# --------------------------------------------------

all_features = []

print(f"\nExtracting {NUM_UTTERANCES} utterances...\n")

for i, sample in enumerate(dataset):

    if i >= NUM_UTTERANCES:
        break

    try:

        audio_bytes = sample["audio_filepath"]["bytes"]

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
        layer11 = outputs.hidden_states[11]

        # [1, T, 768] -> [T, 768]
        features = layer11.squeeze(0).cpu().numpy()

        all_features.append(features)

        print(
            f"{i + 1:03d}/{NUM_UTTERANCES} | "
            f"{sample['duration']:.2f}s | "
            f"{features.shape}"
        )

    except Exception as e:

        print(
            f"Skipping sample {i + 1}: {e}"
        )


# --------------------------------------------------
# 5. Concatenate
# --------------------------------------------------

features = np.concatenate(
    all_features,
    axis=0
)

print("\n-----------------------------")
print("Extraction complete")
print("-----------------------------")

print("Utterances:", len(all_features))
print("Feature matrix:", features.shape)
print("dtype:", features.dtype)


# --------------------------------------------------
# 6. Save
# --------------------------------------------------

np.save(
    "tamil_mhubert_layer11_100.npy",
    features
)

print("\nSaved tamil_mhubert_layer11_100.npy")