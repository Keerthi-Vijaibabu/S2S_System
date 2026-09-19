import numpy as np
import torch
import soundfile as sf
import joblib

from transformers import AutoFeatureExtractor, AutoModel


MODEL_NAME = "utter-project/mHuBERT-147"
AUDIO_FILE = "tamil_sample_16k.wav"
KMEANS_FILE = "tamil_kmeans_1000_test.joblib"


# ------------------------------------------------
# Device
# ------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ------------------------------------------------
# Load mHuBERT
# ------------------------------------------------

print("\nLoading mHuBERT...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()


# ------------------------------------------------
# Load KMeans
# ------------------------------------------------

print("Loading KMeans...")

kmeans = joblib.load(KMEANS_FILE)

print(
    "KMeans clusters:",
    kmeans.cluster_centers_.shape
)


# ------------------------------------------------
# Load audio
# ------------------------------------------------

audio, sr = sf.read(
    AUDIO_FILE,
    dtype="float32"
)

assert sr == 16000, f"Expected 16 kHz audio, got {sr}"


# ------------------------------------------------
# mHuBERT
# ------------------------------------------------

inputs = feature_extractor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

input_values = inputs.input_values.to(device)

with torch.inference_mode():

    outputs = model(
        input_values,
        output_hidden_states=True
    )

features = (
    outputs.hidden_states[11]
    .squeeze(0)
    .cpu()
    .numpy()
)

print("\nmHuBERT features:", features.shape)


# ------------------------------------------------
# Quantize
# ------------------------------------------------

units = kmeans.predict(features)

print("Unit sequence:", units.shape)

print("\nFirst 100 units:")
print(units[:100])


# ------------------------------------------------
# Collapse consecutive duplicates
# ------------------------------------------------

collapsed = [
    int(units[0])
]

for unit in units[1:]:

    if unit != collapsed[-1]:
        collapsed.append(int(unit))

collapsed = np.array(collapsed)


# ------------------------------------------------
# Statistics
# ------------------------------------------------

unique_units, counts = np.unique(
    units,
    return_counts=True
)

print("\n-----------------------------")
print("UNIT STATISTICS")
print("-----------------------------")

print("Total frames:", len(units))
print("Unique units used:", len(unique_units))
print("Collapsed length:", len(collapsed))

print(
    "Duplicate reduction:",
    round(
        (1 - len(collapsed) / len(units)) * 100,
        2
    ),
    "%"
)


# Top units

order = np.argsort(counts)[::-1]

print("\n10 most common units:")

for idx in order[:10]:

    print(
        f"Unit {unique_units[idx]}: "
        f"{counts[idx]} frames"
    )


# ------------------------------------------------
# Save
# ------------------------------------------------

np.savetxt(
    "tamil_sample.units",
    units,
    fmt="%d"
)

np.savetxt(
    "tamil_sample_collapsed.units",
    collapsed,
    fmt="%d"
)

print("\nSaved:")
print("tamil_sample.units")
print("tamil_sample_collapsed.units")