import numpy as np
import joblib

from sklearn.cluster import MiniBatchKMeans


FEATURE_FILE = "tamil_kmeans_features_test.npy"
OUTPUT_MODEL = "tamil_kmeans_1000_test.joblib"

N_CLUSTERS = 1000
BATCH_SIZE = 4096
SEED = 42


print("Loading features...")

features = np.load(FEATURE_FILE)

print("Feature matrix:", features.shape)
print("dtype:", features.dtype)


print("\nTraining MiniBatchKMeans...")

kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    batch_size=BATCH_SIZE,
    random_state=SEED,
    n_init="auto",
    verbose=1
)

kmeans.fit(features)


print("\nTraining complete!")

print("Cluster centers:", kmeans.cluster_centers_.shape)
print("Inertia:", kmeans.inertia_)


print("\nSaving model...")

joblib.dump(
    kmeans,
    OUTPUT_MODEL
)

print("Saved:", OUTPUT_MODEL)