"""
Central configuration for the Tamil -> English S2ST project.

Every script imports its paths and shared constants from here, so nothing
depends on which directory you launch it from.

Scripts inside subfolders (dataset/, mHuBERT/) add the project root to
sys.path before importing this file:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config
"""

from pathlib import Path


# ============================================================
# PROJECT LAYOUT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "pretrained_models"


# ============================================================
# DATASET
# ============================================================

HF_DATASET = "ai4bharat/IndicVoices"
HF_LANGUAGE = "tamil"

# Speaker-disjoint 80/10/10 split metadata (speaker_id, duration,
# audio_path, text). Produced by dataset/data_split.py.
SPLIT_FILE = PROJECT_ROOT / "indicvoices_tamil_split_full.json"

TARGET_SR = 16000
SEED = 42


# ============================================================
# mHuBERT FEATURES + K-MEANS
# ============================================================

MHUBERT_MODEL = "utter-project/mHuBERT-147"

# Transformer layer whose output is quantized.
HIDDEN_LAYER = 11

N_CLUSTERS = 1000

# Frames randomly sampled from each utterance when collecting
# K-means training features.
FRAMES_PER_UTTERANCE = 50


def chunk_dir(split_name="train"):
    """
    Folder holding the sampled mHuBERT feature chunks for a split.

    train      -> tamil_kmeans_chunks
    validation -> tamil_kmeans_chunks_validation
    test       -> tamil_kmeans_chunks_test
    """
    if split_name == "train":
        return PROJECT_ROOT / "tamil_kmeans_chunks"

    return PROJECT_ROOT / f"tamil_kmeans_chunks_{split_name}"


# Final K-means model (output of mHuBERT/train_kmeans.py, input of
# mHuBERT/generate_kmeans_units.py).
KMEANS_MODEL = MODEL_DIR / "tamil_kmeans.joblib"

# Small K-means model checked into the repo for quick tests.
KMEANS_TEST_MODEL = MODEL_DIR / "tamil_kmeans_1000_test.joblib"

# Where generate_kmeans_units.py writes unit IDs.
UNITS_DIR = PROJECT_ROOT / "mHuBERT" / "tamil_kmeans_units"


# ============================================================
# SAMPLE FILES (used by the small test scripts)
# ============================================================

SAMPLE_FLAC = PROJECT_ROOT / "tamil_sample.flac"
SAMPLE_WAV_16K = PROJECT_ROOT / "tamil_sample_16k.wav"