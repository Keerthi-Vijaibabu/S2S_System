import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

from datasets import load_dataset, Audio
from collections import defaultdict
import json
import random

SEED = config.SEED

# None  -> scan the entire dataset and write config.SPLIT_FILE
# 5000  -> quick test on the first 5,000 utterances; writes
#          indicvoices_tamil_split_5000.json instead, so the full
#          split file is never overwritten by a test run.
MAX_SCAN = None

if MAX_SCAN is None:
    OUTPUT_FILE = config.SPLIT_FILE
else:
    OUTPUT_FILE = (
        config.PROJECT_ROOT
        / f"indicvoices_tamil_split_{MAX_SCAN}.json"
    )

random.seed(SEED)

print("Loading IndicVoices Tamil...")

ds = load_dataset(
    "ai4bharat/IndicVoices",
    "tamil",
    split="train",
    streaming=True
)

# Prevent TorchCodec decoding
ds = ds.cast_column(
    "audio_filepath",
    Audio(decode=False)
)

print("Dataset loaded")


# --------------------------------------------------
# 1. Scan metadata
# --------------------------------------------------

speakers = defaultdict(list)

print(f"\nScanning dataset utterances...")

for i, sample in enumerate(ds):

    if MAX_SCAN is not None and i >= MAX_SCAN:
        break

    if (i + 1) % 1000 == 0:
        print(f"Scanned {i + 1:,} utterances...")

    speaker_id = sample["speaker_id"]

    # We don't store audio bytes!
    # The Tamil transcript is kept: later stages (data filtering,
    # translation, auxiliary CTC loss) all need it.
    metadata = {
        "speaker_id": speaker_id,
        "duration": sample["duration"],
        "text": sample["text"],
        "audio_path": sample["audio_filepath"]["path"],
    }

    speakers[speaker_id].append(metadata)

print("\nScan complete")
print("Utterances:", sum(len(x) for x in speakers.values()))
print("Unique speakers:", len(speakers))


# --------------------------------------------------
# 2. Shuffle speakers
# --------------------------------------------------

speaker_ids = list(speakers.keys())

random.shuffle(speaker_ids)

n = len(speaker_ids)

train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_speakers = set(speaker_ids[:train_end])
val_speakers = set(speaker_ids[train_end:val_end])
test_speakers = set(speaker_ids[val_end:])


# --------------------------------------------------
# 3. Construct split metadata
# --------------------------------------------------

splits = {
    "train": [],
    "validation": [],
    "test": []
}

for speaker_id, samples in speakers.items():

    if speaker_id in train_speakers:
        splits["train"].extend(samples)

    elif speaker_id in val_speakers:
        splits["validation"].extend(samples)

    else:
        splits["test"].extend(samples)


# --------------------------------------------------
# 4. Print statistics
# --------------------------------------------------

print("\n-----------------------------")
print("SPLIT STATISTICS")
print("-----------------------------")

for split_name, samples in splits.items():

    duration = sum(x["duration"] for x in samples)

    unique_speakers = len(
        set(x["speaker_id"] for x in samples)
    )

    print(f"\n{split_name.upper()}")
    print("Utterances:", len(samples))
    print("Speakers:", unique_speakers)
    print("Duration:", round(duration / 3600, 2), "hours")


# --------------------------------------------------
# 5. Verify speaker separation
# --------------------------------------------------

assert train_speakers.isdisjoint(val_speakers)
assert train_speakers.isdisjoint(test_speakers)
assert val_speakers.isdisjoint(test_speakers)

print("\nSpeaker-disjoint split verified!")


# --------------------------------------------------
# 6. Save metadata
# --------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        splits,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"\nSaved {OUTPUT_FILE}")