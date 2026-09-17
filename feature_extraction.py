import torch
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModel

MODEL_NAME = "utter-project/mHuBERT-147"
AUDIO_FILE = "tamil_sample_16k.wav"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

print("Loading feature extractor...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

print("Loading mHuBERT-147...")

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(device)
model.eval()

print("Model loaded")

audio, sample_rate = sf.read(
    AUDIO_FILE,
    dtype="float32"
)

print("\nAudio information")
print("Sample rate:", sample_rate)
print("Audio samples:", len(audio))
print("Duration:", len(audio) / sample_rate)

assert sample_rate == 16000, "Audio must be 16 kHz"


inputs = feature_extractor(
    audio,
    sampling_rate=sample_rate,
    return_tensors="pt"
)

input_values = inputs.input_values.to(device)

print("\nInput tensor:", input_values.shape)

print("\nRunning mHuBERT...")

with torch.no_grad():

    outputs = model(
        input_values,
        output_hidden_states=True
    )

print("\nNumber of hidden states:", len(outputs.hidden_states))

for i, hidden_state in enumerate(outputs.hidden_states):
    print(
        f"Hidden state {i}:",
        hidden_state.shape
    )


layer11 = outputs.hidden_states[11]

print("\n--- LAYER 11 ---")
print("Shape:", layer11.shape)
print("Device:", layer11.device)
print("Data type:", layer11.dtype)