"""Extract full, ordered mHuBERT -> K-means unit sequences for IndicVoices Tamil.

Install: pip install torch transformers datasets soundfile scipy numpy joblib scikit-learn
Run: python generate encodings.py

Each JSONL row is one utterance. Outputs preserve repeated unit IDs and ordering.
The dataset may supply no stable filesystem path when streamed; utterance_index is
an index within the filtered speaker subset of the streamed dataset.
"""
import gc
import io
import json
import os
from math import gcd
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModel

# ------------------- CONFIG -------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_NAME = "utter-project/mHuBERT-147"

KMEANS_MODEL = BASE_DIR / "pretrained_models" / "tamil_kmeans.joblib"

SPLIT_FILE = BASE_DIR / "indicvoices_tamil_split_full.json"

SPLIT_KEY = "train"

OUTPUT_DIR = BASE_DIR / "tamil_full_units"

TARGET_SR = 16000
HIDDEN_LAYER = 11
BATCH_SIZE = 8
CHUNK_UTTERANCES = 1000
MAX_UTTERANCES = None # limit number of eligible utterances, or None
DATASET_NAME = "ai4bharat/IndicVoices"
DATASET_CONFIG = "tamil"
DATASET_SPLIT = "train"

# ------------------- HELPERS -------------------
def decode_audio(audio_info):
    if audio_info.get("bytes") is not None:
        with io.BytesIO(audio_info["bytes"]) as stream:
            audio, sr = sf.read(stream, dtype="float32")
    elif audio_info.get("path") and os.path.isfile(audio_info["path"]):
        audio, sr = sf.read(audio_info["path"], dtype="float32")
    else:
        raise ValueError("No decodable audio bytes or local path")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        d = gcd(int(sr), TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // d, int(sr) // d)
    audio = np.asarray(audio, dtype=np.float32)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("Empty or non-finite audio")
    return audio


def get_valid_frame_count(model, num_samples, max_frames):
    # mHuBERT / HuBERT convolutional output length; padding is excluded.
    if not hasattr(model, "_get_feat_extract_output_lengths"):
        raise RuntimeError("Model does not expose convolutional feature lengths; cannot safely trim padding")
    length = model._get_feat_extract_output_lengths(torch.tensor([num_samples]))
    return min(int(length.item()), max_frames)


def extract_batch(model, extractor, device, audios):
    inputs = extractor(audios, sampling_rate=TARGET_SR, padding=True,
                       return_attention_mask=True, return_tensors="pt")
    values = inputs["input_values"].to(device)
    mask = inputs.get("attention_mask")
    if mask is not None:
        mask = mask.to(device)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(values, attention_mask=mask, output_hidden_states=True)
        else:
            output = model(values, attention_mask=mask, output_hidden_states=True)
        hidden = output.hidden_states[HIDDEN_LAYER]
        # One full ordered array per input utterance, not random frame samples.
        arrays = [hidden[i, :get_valid_frame_count(model, len(a), hidden.shape[1])]
                  .float().cpu().numpy() for i, a in enumerate(audios)]
    del output, hidden, values, mask, inputs
    return arrays


def extract_safe(model, extractor, device, records):
    """Return (record, features) pairs; skip only a single audio that OOMs."""
    if not records:
        return []
    try:
        features = extract_batch(model, extractor, device, [r["audio"] for r in records])
        return list(zip(records, features))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" not in str(exc).lower() and "cudaerrormemoryallocation" not in str(exc).lower():
            raise
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if len(records) == 1:
            print(f"Skipping OOM utterance {records[0]['utterance_index']}")
            return []
        mid = len(records) // 2
        return (extract_safe(model, extractor, device, records[:mid]) +
                extract_safe(model, extractor, device, records[mid:]))


def atomic_write_jsonl(path, rows):
    # Create temp file and rename; never overwrite completed chunks.
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temp, path)
    print(f"Saved {path} ({len(rows)} utterances)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(BASE_DIR)
    # A checkpoint is the last successfully written utterance index, so failed
    # or interrupted runs can safely replay unsaved rows without duplication.
    chunks = sorted(OUTPUT_DIR.glob("units_chunk_*.jsonl"))
    last_saved = -1
    for path in chunks:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_saved = max(last_saved, int(json.loads(line)["utterance_index"]))
    next_chunk = max((int(p.stem.rsplit("_", 1)[1]) for p in chunks), default=-1) + 1
    print(f"Resuming after eligible utterance index {last_saved}; next chunk {next_chunk}")

    with open(SPLIT_FILE, encoding="utf-8") as f:
        splits = json.load(f)
    allowed_speakers = {str(x["speaker_id"]) for x in splits[SPLIT_KEY]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    kmeans = joblib.load(KMEANS_MODEL)
    if not hasattr(kmeans, "predict"):
        raise TypeError("Loaded K-means object has no predict method")
    if HIDDEN_LAYER >= model.config.num_hidden_layers + 1:
        raise ValueError("HIDDEN_LAYER out of range")
    print("K-means clusters:", getattr(kmeans, "n_clusters", "unknown"))

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True)
    ds = ds.cast_column("audio_filepath", Audio(decode=False))
    pending = []
    output_rows = []
    eligible_index = -1
    seen_this_run = 0
    success = 0
    errors = 0

    def process_pending():
        nonlocal pending, output_rows, next_chunk, success, errors
        results = extract_safe(model, extractor, device, pending)
        errors += len(pending) - len(results)
        for record, feats in results:
            if not len(feats):
                errors += 1
                continue
            # K-means requires the same mHuBERT layer/features as training.
            ids = kmeans.predict(feats).astype(np.int32).tolist()
            output_rows.append({
                "utterance_index": record["utterance_index"],
                "audio_path": record["audio_path"],
                "speaker_id": record["speaker_id"],
                "num_frames": len(ids),
                "units": ids,
            })
            success += 1
        pending = []
        while len(output_rows) >= CHUNK_UTTERANCES:
            atomic_write_jsonl(OUTPUT_DIR / f"units_chunk_{next_chunk:05d}.jsonl",
                               output_rows[:CHUNK_UTTERANCES])
            output_rows = output_rows[CHUNK_UTTERANCES:]
            next_chunk += 1

    for row in ds:
        if str(row.get("speaker_id")) not in allowed_speakers:
            continue
        eligible_index += 1
        if eligible_index <= last_saved:
            continue
        if MAX_UTTERANCES is not None and seen_this_run >= MAX_UTTERANCES:
            break
        seen_this_run += 1
        try:
            audio_info = row["audio_filepath"]
            audio = decode_audio(audio_info)
            # Streaming datasets may expose an archive-relative path or None,
            # not necessarily a path to a locally accessible file.
            audio_path = audio_info.get("path")
            pending.append({"audio": audio, "audio_path": audio_path,
                            "speaker_id": str(row.get("speaker_id")),
                            "utterance_index": eligible_index})
        except Exception as exc:
            errors += 1
            print(f"Audio error at eligible index {eligible_index}: {exc}")
            continue
        if len(pending) >= BATCH_SIZE:
            process_pending()
            if seen_this_run % 100 < BATCH_SIZE:
                print(f"Eligible this run: {seen_this_run}, successful: {success}, errors: {errors}")

    process_pending()
    if output_rows:
        atomic_write_jsonl(OUTPUT_DIR / f"units_chunk_{next_chunk:05d}.jsonl", output_rows)
    print(f"Finished. Successful: {success}; skipped/errors: {errors}; output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
