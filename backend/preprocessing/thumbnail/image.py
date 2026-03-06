import sys
from pathlib import Path
import requests
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import pandas as pd

# 1. Dynamically locate the backend directory and the CSV file
root_dir = Path(__file__).resolve().parent.parent.parent
csv_path = root_dir / 'simplifiedUSvideos.csv'

# Add root_dir to sys.path if you still need to import other things from backend
sys.path.insert(0, str(root_dir)) 

def preprocess_thumbnail_pad(url, target_size=(224, 224), pad_color=(0, 0, 0)): 
    """
    Downloads an image from a URL, resizes it to fit within target_size 
    while maintaining aspect ratio, and pads the remaining space.
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() 
        
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        
        img = ImageOps.pad(img, target_size, method=Image.Resampling.BICUBIC, color=pad_color)
        
        img_array = np.array(img) / 255.0
        return img_array
        
    except Exception as e:
        # Keeping this print statement so you can see which URLs are failing
        print(f"Error loading {url}: {e}")
        return None

# 2. Load the CSV directly using Pandas
print(f"Loading dataset from: {csv_path}")
data = pd.read_csv(csv_path)

# --- HEADS UP! ---
# You have this set to 30 for testing. 
# Make sure to delete or comment out this line when you want to clean the whole dataset!
# data = data.head(30)

# Lists to hold data
valid_images = []
indices_to_drop = []

print("Starting thumbnail processing...")

# 3. Iterate and find the bad rows
for index, row in data.iterrows():
    url = row['thumbnail_link'] 
    
    img_array = preprocess_thumbnail_pad(url)
    
    if img_array is not None:
        valid_images.append(img_array)
    else:
        indices_to_drop.append(index)

# 4. Clean the DataFrame
data.drop(index=indices_to_drop, inplace=True)
data.reset_index(drop=True, inplace=True)

# 5. Overwrite the original CSV file!
data.to_csv(csv_path, index=False)
print(f"\n--- SUCCESS: Overwrote original CSV at {csv_path} ---")

# ==========================================
# 6. Save the Image Arrays locally to 'backend/'
# ==========================================
X_images = np.array(valid_images)

# Since root_dir points to your main 'backend' folder, 
# this saves the file directly into that folder.
npy_path = root_dir / 'processed_images.npy'

# Save the massive array locally
np.save(npy_path, X_images)
print(f"--- SUCCESS: Saved image tensor to {npy_path} ---")
print(f"Final Image Tensor Shape: {X_images.shape}")

