import numpy as np

# Load the array from disk
data = np.load(r'tamil_mhubert_layer11_100.npy')

# View its properties
print(data.shape)
print(data.dtype)
print(data)
