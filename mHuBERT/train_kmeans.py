import os
import glob
import json
import numpy as np
import joblib

from sklearn.cluster import MiniBatchKMeans


# ============================================================
# CONFIG
# ============================================================

CHUNK_DIR = "tamil_kmeans_chunks"

OUTPUT_MODEL = "tamil_kmeans.joblib"

CHECKPOINT_MODEL = "tamil_kmeans_checkpoint.joblib"
CHECKPOINT_STATE = "tamil_kmeans_checkpoint_state.json"

N_CLUSTERS = 1000
BATCH_SIZE = 4096
SEED = 42


# ============================================================
# FIND FEATURE CHUNKS
# ============================================================

chunk_files = sorted(
    glob.glob(
        os.path.join(
            CHUNK_DIR,
            "features_chunk_*.npy"
        )
    )
)

if not chunk_files:
    raise RuntimeError(
        f"No feature chunks found in: {CHUNK_DIR}"
    )

print("=" * 60)
print("TAMIL mHuBERT KMEANS TRAINING")
print("=" * 60)

print(f"Chunks found : {len(chunk_files)}")
print(f"Clusters     : {N_CLUSTERS}")
print(f"Batch size   : {BATCH_SIZE}")
print()


# ============================================================
# LOAD CHECKPOINT OR CREATE NEW MODEL
# ============================================================

start_chunk = 0
total_frames = 0


if (
    os.path.exists(CHECKPOINT_MODEL)
    and os.path.exists(CHECKPOINT_STATE)
):

    print("Checkpoint found.")
    print("Resuming training...\n")

    kmeans = joblib.load(CHECKPOINT_MODEL)

    with open(CHECKPOINT_STATE, "r") as f:
        state = json.load(f)

    start_chunk = state["next_chunk"]
    total_frames = state["total_frames"]

    print(
        f"Resuming from chunk "
        f"{start_chunk + 1}/{len(chunk_files)}"
    )

    print(
        f"Frames already trained: "
        f"{total_frames:,}"
    )

    print()

else:

    print("No checkpoint found.")
    print("Starting new KMeans model...\n")

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=BATCH_SIZE,
        random_state=SEED,
        n_init="auto",
        verbose=0
    )


# ============================================================
# TRAIN CHUNK BY CHUNK
# ============================================================

for chunk_index in range(
    start_chunk,
    len(chunk_files)
):

    chunk_file = chunk_files[chunk_index]

    print("=" * 60)

    print(
        f"Chunk {chunk_index + 1}/"
        f"{len(chunk_files)}"
    )

    print(
        f"File: {os.path.basename(chunk_file)}"
    )


    # --------------------------------------------------------
    # Memory-map feature file
    # --------------------------------------------------------

    features = np.load(
        chunk_file,
        mmap_mode="r"
    )

    print(
        f"Feature matrix: {features.shape}"
    )

    print(
        f"dtype: {features.dtype}"
    )


    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    if features.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix, "
            f"got {features.shape}"
        )

    num_frames = features.shape[0]


    # --------------------------------------------------------
    # Train this chunk in mini-batches
    # --------------------------------------------------------

    for start in range(
        0,
        num_frames,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            num_frames
        )

        batch = np.asarray(
            features[start:end],
            dtype=np.float32
        )


        # ----------------------------------------------------
        # Initial partial_fit needs >= N_CLUSTERS samples
        # ----------------------------------------------------

        if (
            not hasattr(kmeans, "cluster_centers_")
            and batch.shape[0] < N_CLUSTERS
        ):
            print(
                "Skipping initial batch because "
                "it contains fewer samples than clusters."
            )
            continue


        kmeans.partial_fit(batch)

        total_frames += batch.shape[0]


    print(
        f"Total frames trained: "
        f"{total_frames:,}"
    )


    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    print("Saving checkpoint...")

    joblib.dump(
        kmeans,
        CHECKPOINT_MODEL
    )

    state = {
        "next_chunk": chunk_index + 1,
        "total_frames": total_frames,
        "last_completed_chunk": os.path.basename(
            chunk_file
        )
    }

    with open(
        CHECKPOINT_STATE,
        "w"
    ) as f:
        json.dump(
            state,
            f,
            indent=4
        )

    print(
        f"Checkpoint saved after chunk "
        f"{chunk_index + 1}"
    )

    # Release mmap reference before next file
    del features


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Cluster centers:",
    kmeans.cluster_centers_.shape
)

print(
    "Total frames trained:",
    f"{total_frames:,}"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\nSaving final model...")

joblib.dump(
    kmeans,
    OUTPUT_MODEL
)

print(
    f"Saved final model: {OUTPUT_MODEL}"
)


# ============================================================
# REMOVE CHECKPOINT FILES
# ============================================================

if os.path.exists(CHECKPOINT_MODEL):
    os.remove(CHECKPOINT_MODEL)

if os.path.exists(CHECKPOINT_STATE):
    os.remove(CHECKPOINT_STATE)

print("Checkpoint files removed.")

print("\nDone!")