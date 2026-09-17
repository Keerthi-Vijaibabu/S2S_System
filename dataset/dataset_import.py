from datasets import load_dataset, Audio

print("Loading IndicVoices Tamil...")

ds = load_dataset(
    "ai4bharat/IndicVoices",
    "tamil",
    split="train",
    streaming=True
)

print("Dataset loaded")
print("Audio feature:", ds.features["audio_filepath"])

# Prevent Hugging Face from automatically decoding with TorchCodec
ds = ds.cast_column(
    "audio_filepath",
    Audio(decode=False)
)

print("Reading one sample...")

sample = next(iter(ds))

print("\n--- SAMPLE ---")
print("Text:", sample["text"])
print("Duration:", sample["duration"])
print("Language:", sample["lang"])
print("Speaker:", sample["speaker_id"])

audio = sample["audio_filepath"]

print("Audio keys:", audio.keys())
print("Path:", audio.get("path"))
print("Number of bytes:", len(audio["bytes"]))

with open("tamil_sample.flac", "wb") as f:
    f.write(audio["bytes"])

print("\nSaved tamil_sample.flac!")