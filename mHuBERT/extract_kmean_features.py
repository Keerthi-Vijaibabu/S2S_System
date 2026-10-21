import io
import os
import gc
import json
import random
import time
from math import gcd

import numpy as np
import soundfile as sf
import torch

from datasets import load_dataset, Audio
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModel

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = config.MHUBERT_MODEL

SPLIT_FILE = config.SPLIT_FILE

# Which split's speakers to extract features for.
#
#   "train"      -> K-means training features
#                   (chunks go to tamil_kmeans_chunks)
#   "validation" -> held-out features
#                   (chunks go to tamil_kmeans_chunks_validation)
#   "test"       -> held-out features
#                   (chunks go to tamil_kmeans_chunks_test)
SPLIT_NAME = "train"

TARGET_SR = config.TARGET_SR

# Limit the number of utterances processed in one run.
# None = process the entire split.
#
# NOTE: the full train split is ~336k utterances. At 50 frames x
# 768 floats each, that is roughly 50 GB of chunk files. K-means
# with 1000 clusters does not need that many frames; a cap of
# 100,000 - 150,000 utterances is usually plenty.
MAX_UTTERANCES = None

# Normal GPU batch size (an 8 GB GPU is fine with fp16 autocast;
# the OOM handler below halves the batch automatically if needed)
BATCH_SIZE = 8

# Maximum number of frames sampled from each utterance
FRAMES_PER_UTTERANCE = config.FRAMES_PER_UTTERANCE

# Layer used for K-means features
HIDDEN_LAYER = config.HIDDEN_LAYER

# Also merge all chunks into one huge .npy file at the end.
# train_kmeans.py reads the chunks directly, so this is normally
# unnecessary and doubles the disk usage.
COMBINE_CHUNKS = False

SEED = config.SEED


# ============================================================
# RESUME CONFIGURATION
# ============================================================

# To resume an interrupted run, set both values below:
#
#   RESUME_FROM_UTTERANCE -> number of utterances (of this split)
#                            already processed
#   START_CHUNK_NUMBER    -> number of the next chunk file to write
#
# Example: if chunks 0, 1 and 2 were saved with 1000 utterances
# each, use RESUME_FROM_UTTERANCE = 3000 and START_CHUNK_NUMBER = 3.
#
# For a fresh run keep both at 0. Existing chunk files are never
# overwritten; the script stops with an error instead.

RESUME_FROM_UTTERANCE = 0

START_CHUNK_NUMBER = 0

# Utterances per saved chunk file
CHUNK_UTTERANCES = 1000


# ============================================================
# OUTPUT
# ============================================================

CHUNK_DIR = config.chunk_dir(SPLIT_NAME)

OUTPUT_FILE = (
    config.PROJECT_ROOT
    / f"tamil_kmeans_features_{SPLIT_NAME}.npy"
)


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEVICE INFORMATION")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    # Improve matrix multiplication performance
    torch.backends.cuda.matmul.allow_tf32 = True

    torch.set_float32_matmul_precision(
        "high"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    CHUNK_DIR,
    exist_ok=True
)


# ============================================================
# VERIFY EXISTING CHUNKS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING EXISTING CHUNKS")
print("=" * 70)

existing_chunks = sorted([
    f
    for f in os.listdir(CHUNK_DIR)
    if (
        f.startswith("features_chunk_")
        and f.endswith(".npy")
    )
])

if existing_chunks:

    print(
        f"Found {len(existing_chunks)} "
        f"existing chunks:"
    )

    for filename in existing_chunks:

        path = os.path.join(
            CHUNK_DIR,
            filename
        )

        arr = np.load(
            path,
            mmap_mode="r"
        )

        print(
            filename,
            arr.shape
        )

else:

    print("No existing chunks found.")


# ============================================================
# SAFETY CHECK
# ============================================================

for i in range(1, START_CHUNK_NUMBER):

    expected = (
        f"features_chunk_{i:05d}.npy"
    )

    path = os.path.join(
        CHUNK_DIR,
        expected
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Expected existing chunk missing: "
            f"{path}"
        )

print(
    "\nExisting chunks verified."
)

print(
    f"Will resume after "
    f"{RESUME_FROM_UTTERANCE:,} "
    f"training utterances."
)

print(
    f"Next chunk number: "
    f"{START_CHUNK_NUMBER}"
)


# ============================================================
# LOAD TRAIN SPEAKER IDS
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAIN SPLIT")
print("=" * 70)

with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8"
) as f:

    splits = json.load(f)


if SPLIT_NAME not in splits:

    raise KeyError(
        f"Split '{SPLIT_NAME}' not found in {SPLIT_FILE}. "
        f"Available: {list(splits.keys())}"
    )


split_speakers = {
    item["speaker_id"]
    for item in splits[SPLIT_NAME]
}

print(
    f"Split: {SPLIT_NAME}"
)

print(
    "Speakers in this split:",
    len(split_speakers)
)


# ============================================================
# LOAD mHuBERT
# ============================================================

print("\n" + "=" * 70)
print("LOADING mHuBERT")
print("=" * 70)

feature_extractor = (
    AutoFeatureExtractor.from_pretrained(
        MODEL_NAME
    )
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(device)

model.eval()

print("mHuBERT loaded.")


# ============================================================
# LOAD INDICVOICES
# ============================================================

print("\n" + "=" * 70)
print("LOADING INDICVOICES")
print("=" * 70)

ds = load_dataset(
    "ai4bharat/IndicVoices",
    "tamil",
    split="train",
    streaming=True
)

ds = ds.cast_column(
    "audio_filepath",
    Audio(decode=False)
)

print("IndicVoices stream ready.")


# ============================================================
# AUDIO DECODING
# ============================================================

def decode_audio(audio_bytes):

    audio, sr = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    # Stereo -> mono
    if audio.ndim > 1:

        audio = audio.mean(
            axis=1
        )

    # Resample if required
    if sr != TARGET_SR:

        g = gcd(
            int(sr),
            TARGET_SR
        )

        audio = resample_poly(
            audio,
            TARGET_SR // g,
            int(sr) // g
        )

    return np.asarray(
        audio,
        dtype=np.float32
    )


# ============================================================
# FEATURE LENGTH
# ============================================================

def get_feature_length(
    input_length
):

    if hasattr(
        model,
        "_get_feat_extract_output_lengths"
    ):

        length = (
            model
            ._get_feat_extract_output_lengths(
                input_length
            )
        )

        if torch.is_tensor(length):
            length = length.item()

        return int(length)

    # Fallback for HuBERT-style encoders
    return max(
        1,
        input_length // 320
    )


# ============================================================
# NORMAL BATCH PROCESSING
# ============================================================

def process_batch(
    audio_batch
):

    raw_lengths = [
        len(audio)
        for audio in audio_batch
    ]

    inputs = feature_extractor(
        audio_batch,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True
    )

    input_values = (
        inputs["input_values"]
        .to(device)
    )

    attention_mask = (
        inputs.get(
            "attention_mask"
        )
    )

    if attention_mask is not None:

        attention_mask = (
            attention_mask.to(device)
        )

    # Reset peak measurement for this batch
    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats()


    with torch.inference_mode():

        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

        else:

            outputs = model(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True
            )


    hidden = (
        outputs
        .hidden_states[HIDDEN_LAYER]
    )


    result = []


    for i, input_length in enumerate(
        raw_lengths
    ):

        valid_frames = (
            get_feature_length(
                input_length
            )
        )

        valid_frames = min(
            valid_frames,
            hidden.shape[1]
        )

        if valid_frames <= 0:
            continue


        take = min(
            FRAMES_PER_UTTERANCE,
            valid_frames
        )


        indices = np.random.choice(
            valid_frames,
            size=take,
            replace=False
        )


        indices = torch.tensor(
            indices,
            device=hidden.device,
            dtype=torch.long
        )


        sampled = hidden[
            i,
            indices
        ]


        sampled = (
            sampled
            .float()
            .cpu()
            .numpy()
        )


        result.append(
            sampled
        )


    if device.type == "cuda":

        peak = (
            torch.cuda
            .max_memory_allocated()
            / (1024 ** 3)
        )

        print(
            f"Peak GPU memory: "
            f"{peak:.2f} GB"
        )


    # Explicitly release large tensors
    del outputs
    del hidden
    del input_values

    if attention_mask is not None:
        del attention_mask


    return result


# ============================================================
# SAFE OOM PROCESSING
# ============================================================

def process_batch_safe(
    audio_batch
):

    """
    Try processing the complete batch.

    If CUDA runs out of memory:

        8
       / \
      4   4
         / \
        2   2

    until the batch fits.

    Only an utterance that cannot fit by itself
    is skipped.
    """

    try:

        return (
            process_batch(
                audio_batch
            ),
            0
        )


    except (
        torch.cuda.OutOfMemoryError,
        RuntimeError
    ) as e:

        message = str(e).lower()

        if (
            "out of memory"
            not in message
            and
            "cudaerrormemoryallocation"
            not in message
        ):

            raise


        print(
            "\n" + "!" * 70
        )

        print(
            f"CUDA OOM with batch of "
            f"{len(audio_batch)} utterances."
        )


        # Release any cached memory
        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


        # ====================================================
        # SINGLE UTTERANCE STILL TOO LARGE
        # ====================================================

        if len(audio_batch) == 1:

            duration = (
                len(audio_batch[0])
                / TARGET_SR
            )

            print(
                f"Single utterance could not "
                f"fit on GPU."
            )

            print(
                f"Duration: "
                f"{duration:.2f} seconds"
            )

            print(
                "Skipping this utterance."
            )

            print(
                "!" * 70 + "\n"
            )

            return [], 1


        # ====================================================
        # SPLIT BATCH
        # ====================================================

        mid = (
            len(audio_batch) // 2
        )


        left_batch = (
            audio_batch[:mid]
        )

        right_batch = (
            audio_batch[mid:]
        )


        print(
            f"Splitting "
            f"{len(audio_batch)} -> "
            f"{len(left_batch)} + "
            f"{len(right_batch)}"
        )

        print(
            "!" * 70 + "\n"
        )


        left_features, left_skipped = (
            process_batch_safe(
                left_batch
            )
        )


        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


        right_features, right_skipped = (
            process_batch_safe(
                right_batch
            )
        )


        return (
            left_features
            + right_features,
            left_skipped
            + right_skipped
        )


# ============================================================
# SAVE FEATURE CHUNK
# ============================================================

def save_chunk(
    features,
    chunk_number
):

    if not features:

        return None


    matrix = np.concatenate(
        features,
        axis=0
    )


    filename = (
        f"features_chunk_"
        f"{chunk_number:05d}.npy"
    )


    path = os.path.join(
        CHUNK_DIR,
        filename
    )


    # Important:
    # Never overwrite an existing chunk.

    if os.path.exists(path):

        raise FileExistsError(
            f"Refusing to overwrite "
            f"existing chunk: {path}"
        )


    np.save(
        path,
        matrix
    )


    print(
        "\n" + "=" * 70
    )

    print(
        f"SAVED CHUNK {chunk_number}"
    )

    print(
        "File:",
        path
    )

    print(
        "Shape:",
        matrix.shape
    )

    print(
        "Size:",
        f"{os.path.getsize(path) / (1024 ** 2):.2f} MB"
    )

    print(
        "=" * 70 + "\n"
    )


    del matrix

    gc.collect()


    return path


# ============================================================
# EXTRACTION
# ============================================================

print("\n" + "=" * 70)
print("STARTING RESUMED FEATURE EXTRACTION")
print("=" * 70)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Frames per utterance:",
    FRAMES_PER_UTTERANCE
)

print(
    "Resume after:",
    f"{RESUME_FROM_UTTERANCE:,}"
)

print(
    "New chunk interval:",
    CHUNK_UTTERANCES
)

print(
    "Next chunk:",
    START_CHUNK_NUMBER
)

print(
    "Maximum utterances:",
    MAX_UTTERANCES
)


# Number of qualifying train utterances encountered
eligible_seen = 0

# Number processed during THIS resumed run
processed_this_run = 0

# Number skipped because of errors/OOM
skipped = 0

# Number accumulated in current chunk
chunk_utterances = 0

chunk_number = (
    START_CHUNK_NUMBER
)

sampled_features = []

audio_batch = []

start_time = time.time()


# ============================================================
# STREAM DATASET
# ============================================================

for row in ds:

    speaker_id = row.get(
        "speaker_id"
    )


    # Ignore speakers that belong to other splits
    if speaker_id not in split_speakers:
        continue


    # ========================================================
    # RESUME LOGIC
    # ========================================================

    if (
        eligible_seen
        < RESUME_FROM_UTTERANCE
    ):

        eligible_seen += 1

        if (
            eligible_seen % 1000
            == 0
        ):

            print(
                f"Skipping already "
                f"processed data: "
                f"{eligible_seen:,} / "
                f"{RESUME_FROM_UTTERANCE:,}"
            )

        continue


    eligible_seen += 1


    # ========================================================
    # DECODE AUDIO
    # ========================================================

    try:

        audio_info = row[
            "audio_filepath"
        ]

        audio_bytes = (
            audio_info["bytes"]
        )


        audio = decode_audio(
            audio_bytes
        )


        if len(audio) == 0:

            skipped += 1
            continue


        if not np.all(
            np.isfinite(audio)
        ):

            skipped += 1
            continue


    except Exception as e:

        skipped += 1

        print(
            "Audio decoding error:",
            e
        )

        continue


    # ========================================================
    # ADD TO BATCH
    # ========================================================

    audio_batch.append(
        audio
    )


    # ========================================================
    # PROCESS COMPLETE BATCH
    # ========================================================

    if len(audio_batch) >= BATCH_SIZE:

        new_features, batch_skipped = (
            process_batch_safe(
                audio_batch
            )
        )


        sampled_features.extend(
            new_features
        )


        successful = (
            len(audio_batch)
            - batch_skipped
        )


        processed_this_run += (
            successful
        )

        chunk_utterances += (
            successful
        )

        skipped += (
            batch_skipped
        )


        audio_batch = []


        # ====================================================
        # PROGRESS
        # ====================================================

        total_processed = (
            RESUME_FROM_UTTERANCE
            + processed_this_run
        )


        if (
            processed_this_run
            % 100
            < BATCH_SIZE
        ):

            elapsed = (
                time.time()
                - start_time
            )


            speed = (
                processed_this_run
                / elapsed
                if elapsed > 0
                else 0
            )


            print(
                f"Processed total: "
                f"{total_processed:,}"
                f" | This run: "
                f"{processed_this_run:,}"
                f" | Current chunk: "
                f"{chunk_utterances:,}"
                f" | Frames buffered: "
                f"{sum(len(x) for x in sampled_features):,}"
                f" | Skipped: "
                f"{skipped:,}"
                f" | Speed: "
                f"{speed:.2f} utt/s"
            )


        # ====================================================
        # SAVE CHUNK
        # ====================================================

        if (
            chunk_utterances
            >= CHUNK_UTTERANCES
        ):

            save_chunk(
                sampled_features,
                chunk_number
            )


            chunk_number += 1

            sampled_features = []

            chunk_utterances = 0


            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()


    # ========================================================
    # OPTIONAL LIMIT
    # ========================================================

    if (
        MAX_UTTERANCES
        is not None
        and
        processed_this_run
        >= MAX_UTTERANCES
    ):

        break


# ============================================================
# PROCESS FINAL PARTIAL BATCH
# ============================================================

if audio_batch:

    new_features, batch_skipped = (
        process_batch_safe(
            audio_batch
        )
    )


    sampled_features.extend(
        new_features
    )


    successful = (
        len(audio_batch)
        - batch_skipped
    )


    processed_this_run += (
        successful
    )

    chunk_utterances += (
        successful
    )

    skipped += (
        batch_skipped
    )


# ============================================================
# SAVE FINAL PARTIAL CHUNK
# ============================================================

if sampled_features:

    save_chunk(
        sampled_features,
        chunk_number
    )

    chunk_number += 1


# ============================================================
# EXTRACTION SUMMARY
# ============================================================

elapsed = (
    time.time()
    - start_time
)


print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    "Previously processed:",
    f"{RESUME_FROM_UTTERANCE:,}"
)

print(
    "Processed this run:",
    f"{processed_this_run:,}"
)

print(
    "Approx total processed:",
    f"{RESUME_FROM_UTTERANCE + processed_this_run:,}"
)

print(
    "Skipped this run:",
    f"{skipped:,}"
)

print(
    "Elapsed:",
    f"{elapsed / 60:.2f} minutes"
)

if elapsed > 0:

    print(
        "Average speed:",
        f"{processed_this_run / elapsed:.2f} utt/s"
    )


# ============================================================
# INSPECT ALL CHUNKS
# ============================================================

print("\n" + "=" * 70)
print("INSPECTING ALL FEATURE CHUNKS")
print("=" * 70)


chunk_files = sorted([
    os.path.join(
        CHUNK_DIR,
        filename
    )
    for filename in os.listdir(
        CHUNK_DIR
    )
    if (
        filename.startswith(
            "features_chunk_"
        )
        and
        filename.endswith(
            ".npy"
        )
    )
])


if not chunk_files:

    raise RuntimeError(
        "No feature chunks found."
    )


total_frames = 0

feature_dim = None


for i, path in enumerate(
    chunk_files,
    start=1
):

    arr = np.load(
        path,
        mmap_mode="r"
    )


    if feature_dim is None:

        feature_dim = (
            arr.shape[1]
        )


    if (
        arr.shape[1]
        != feature_dim
    ):

        raise ValueError(
            f"Feature dimension mismatch "
            f"in {path}"
        )


    total_frames += (
        arr.shape[0]
    )


    print(
        f"{i:03d} | "
        f"{os.path.basename(path)} | "
        f"{arr.shape}"
    )


print(
    "\nTotal chunks:",
    len(chunk_files)
)

print(
    "Total sampled frames:",
    f"{total_frames:,}"
)

print(
    "Feature dimension:",
    feature_dim
)


# ============================================================
# OPTIONAL: COMBINE CHUNKS INTO ONE FILE
# ============================================================

if not COMBINE_CHUNKS:

    print(
        "\nCOMBINE_CHUNKS is False: not creating a combined "
        "feature file. train_kmeans.py reads the chunks directly."
    )

    sys.exit(0)

print("\n" + "=" * 70)
print("CREATING FINAL FEATURE MATRIX")
print("=" * 70)


final_matrix = (
    np.lib.format.open_memmap(
        OUTPUT_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(
            total_frames,
            feature_dim
        )
    )
)


offset = 0


for i, path in enumerate(
    chunk_files,
    start=1
):

    arr = np.load(
        path,
        mmap_mode="r"
    )


    n = arr.shape[0]


    final_matrix[
        offset:
        offset + n
    ] = arr


    offset += n


    print(
        f"Combined chunk "
        f"{i}/{len(chunk_files)}"
        f" | Frames written: "
        f"{offset:,}"
    )


final_matrix.flush()

del final_matrix

gc.collect()


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)


features = np.load(
    OUTPUT_FILE,
    mmap_mode="r"
)


print(
    "Feature matrix:",
    features.shape
)

print(
    "dtype:",
    features.dtype
)


# Validate in chunks so RAM usage stays low

VALIDATION_BLOCK = 100_000

nan_count = 0

inf_count = 0

total_values = 0

sum_values = 0.0

sum_squares = 0.0


for start in range(
    0,
    features.shape[0],
    VALIDATION_BLOCK
):

    end = min(
        start + VALIDATION_BLOCK,
        features.shape[0]
    )


    block = np.asarray(
        features[start:end],
        dtype=np.float64
    )


    nan_count += int(
        np.isnan(block).sum()
    )

    inf_count += int(
        np.isinf(block).sum()
    )


    finite = np.isfinite(
        block
    )


    values = block[
        finite
    ]


    total_values += (
        values.size
    )

    sum_values += (
        values.sum()
    )

    sum_squares += (
        np.square(values).sum()
    )


mean = (
    sum_values
    / total_values
)


variance = (
    sum_squares
    / total_values
) - (
    mean ** 2
)


std = np.sqrt(
    max(
        variance,
        0.0
    )
)


print(
    "NaN:",
    nan_count
)

print(
    "Inf:",
    inf_count
)

print(
    "Mean:",
    mean
)

print(
    "Std:",
    std
)


file_size = (
    os.path.getsize(
        OUTPUT_FILE
    )
    / (1024 ** 3)
)


print(
    "Final file size:",
    f"{file_size:.2f} GB"
)

print(
    "Final file:",
    OUTPUT_FILE
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)