# This file assumes that the dataset used is already cleaned using the image.py file

import sys
from pathlib import Path
import requests
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import pandas as pd
import h5py # The new hero library

# 1. Setup paths
root_dir = Path(__file__).resolve().parent.parent.parent
csv_path = root_dir / 'USvideos.csv'
h5_path = root_dir / 'processed_images.h5'

sys.path.insert(0, str(root_dir)) 

def preprocess_thumbnail_pad(url, target_size=(224, 224), pad_color=(0, 0, 0)): 
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() 
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = ImageOps.pad(img, target_size, method=Image.Resampling.BICUBIC, color=pad_color)
        
        # Cast to float32 to save massive amounts of hard drive space!
        img_array = (np.array(img) / 255.0).astype(np.float32)
        return img_array
    except Exception as e:
        return None

# 2. Load your ALREADY CLEANED dataset
print(f"Loading dataset from: {csv_path}")
data = pd.read_csv(csv_path)
num_images = len(data)

print(f"Starting pipeline for {num_images} images...")

# 3. Open the HDF5 Vault directly on the hard drive
with h5py.File(h5_path, 'w') as hf:
    
    # Pre-allocate a massive empty array on the hard drive. (No RAM used!)
    images_db = hf.create_dataset("images", shape=(num_images, 224, 224, 3), dtype=np.float32)
    
    # 4. Process and stream directly to disk
    for i, (index, row) in enumerate(data.iterrows()):
        url = row['thumbnail_link'] 
        img_array = preprocess_thumbnail_pad(url)
        
        if img_array is not None:
            images_db[i] = img_array
        else:
            # Just in case a link died in the last 10 minutes, fill with black padding
            images_db[i] = np.zeros((224, 224, 3), dtype=np.float32)
            
        # Print a progress update every 500 images so you know it hasn't frozen
        if i % 500 == 0 and i > 0:
            print(f"Saved {i} / {num_images} images to disk...")

print(f"\n--- SUCCESS: Completely saved image tensor to {h5_path} ---")