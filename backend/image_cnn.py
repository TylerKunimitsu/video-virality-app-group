import pandas as pd
import numpy as np

print("Loading preprocessed data...")

# 1. Load the text and metadata
# df = pd.read_csv('processed_metadata.csv')

# 2. Load the image tensors
X_images = np.load('processed_images.npy')

# print(f"Successfully loaded {len(df)} rows of text data.")
print(f"Successfully loaded image tensor with shape: {X_images.shape}")

# Now you are ready to pass X_images into your model!