import sys
from pathlib import Path

import joblib
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

INPUT_DIR = config.chunk_dir("train")

KMEANS_MODEL = config.KMEANS_MODEL

OUTPUT_DIR = config.UNITS_DIR

# Process this many feature vectors at a time.
# Useful if an individual .npy chunk is very large.
PREDICT_BATCH_SIZE = 100_000

# If True, files already generated will not be processed again.
RESUME = True


# ============================================================
# PREDICTION
# ============================================================

def predict_in_batches(kmeans, features):
    """
    Convert mHuBERT feature vectors into KMeans cluster IDs
    without passing the entire feature matrix through predict()
    at once.
    """

    num_frames = features.shape[0]

    units = np.empty(
        num_frames,
        dtype=np.int32
    )

    for start in range(
        0,
        num_frames,
        PREDICT_BATCH_SIZE
    ):
        end = min(
            start + PREDICT_BATCH_SIZE,
            num_frames
        )

        units[start:end] = kmeans.predict(
            features[start:end]
        )

    return units


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("mHuBERT FEATURES -> KMEANS UNITS")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found:\n{INPUT_DIR}"
        )

    if not KMEANS_MODEL.exists():
        raise FileNotFoundError(
            f"KMeans model not found:\n{KMEANS_MODEL}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nInput directory : {INPUT_DIR}")
    print(f"KMeans model    : {KMEANS_MODEL}")
    print(f"Output directory: {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Load KMeans model
    # --------------------------------------------------------

    print("\nLoading KMeans model...")

    kmeans = joblib.load(KMEANS_MODEL)

    print("KMeans loaded successfully.")

    if hasattr(kmeans, "cluster_centers_"):
        print(
            "Number of clusters:",
            kmeans.cluster_centers_.shape[0]
        )

        print(
            "Feature dimension:",
            kmeans.cluster_centers_.shape[1]
        )

    # --------------------------------------------------------
    # Find feature chunks
    # --------------------------------------------------------

    chunk_files = sorted(
        INPUT_DIR.glob("*.npy")
    )

    if not chunk_files:
        raise RuntimeError(
            f"No .npy files found in:\n{INPUT_DIR}"
        )

    print(
        f"\nFound {len(chunk_files)} feature chunks."
    )

    print("\n" + "=" * 70)
    print("STARTING KMEANS QUANTIZATION")
    print("=" * 70)

    total_frames = 0
    processed_files = 0

    # --------------------------------------------------------
    # Process each chunk independently
    # --------------------------------------------------------

    for index, chunk_path in enumerate(
        chunk_files,
        start=1
    ):

        output_path = (
            OUTPUT_DIR
            / f"{chunk_path.stem}_units.npy"
        )

        print(
            f"\n[{index}/{len(chunk_files)}] "
            f"{chunk_path.name}"
        )

        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if RESUME and output_path.exists():

            print(
                f"  Already exists -> skipping"
            )

            continue

        # ----------------------------------------------------
        # Load feature chunk
        # ----------------------------------------------------

        features = np.load(
            chunk_path,
            mmap_mode="r"
        )

        print(
            f"  Feature shape: {features.shape}"
        )

        print(
            f"  Feature dtype: {features.dtype}"
        )

        # ----------------------------------------------------
        # Validate feature matrix
        # ----------------------------------------------------

        if features.ndim != 2:

            raise ValueError(
                f"{chunk_path.name} has shape "
                f"{features.shape}.\n"
                "Expected shape: "
                "(num_frames, feature_dimension)"
            )

        # Check compatibility with KMeans model

        if hasattr(kmeans, "cluster_centers_"):

            expected_dim = (
                kmeans.cluster_centers_.shape[1]
            )

            if features.shape[1] != expected_dim:

                raise ValueError(
                    f"\nFeature dimension mismatch!\n"
                    f"File: {chunk_path.name}\n"
                    f"Feature dimension: "
                    f"{features.shape[1]}\n"
                    f"KMeans expects: "
                    f"{expected_dim}\n"
                )

        # ----------------------------------------------------
        # KMeans prediction
        # ----------------------------------------------------

        print("  Predicting cluster IDs...")

        units = predict_in_batches(
            kmeans,
            features
        )

        print(
            f"  Units shape: {units.shape}"
        )

        print(
            f"  Unit range: "
            f"{units.min()} -> {units.max()}"
        )

        # ----------------------------------------------------
        # Save discrete units
        # ----------------------------------------------------

        np.save(
            output_path,
            units
        )

        print(
            f"  Saved -> {output_path.name}"
        )

        total_frames += len(units)
        processed_files += 1

        # Explicitly release memory

        del features
        del units

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("QUANTIZATION COMPLETE")
    print("=" * 70)

    print(
        f"Files processed : {processed_files}"
    )

    print(
        f"Frames processed: {total_frames:,}"
    )

    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()