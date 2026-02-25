import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(root_dir))

from state import data # use data for csv file

import requests
from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import pandas as pd


def preprocess_thumbnail_pad(url, target_size=(224, 224), pad_color=(0, 0, 0)): 
    """
    Downloads an image from a URL, resizes it to fit within target_size 
    while maintaining aspect ratio, and pads the remaining space.
    """
    try:
        # 1. Fetch the image from the URL
        response = requests.get(url, timeout=5)
        response.raise_for_status() 
        
        # 2. Open the image and convert to RGB
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        
        # 3. Resize and Pad
        # ImageOps.pad scales the image so the longest side matches 224.
        # It then centers the image and fills the empty space with pad_color (black).
        img = ImageOps.pad(img, target_size, method=Image.Resampling.BICUBIC, color=pad_color)
        
        # 4. Convert to a numpy array and normalize pixel values to [0, 1]
        img_array = np.array(img) / 255.0
        
        return img_array
        
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None

# --- Example Usage ---
sample_url = "https://i.ytimg.com/vi/2kyS6SvSYSE/default.jpg" 
processed_image = preprocess_thumbnail_pad(sample_url)

if processed_image is not None:
    print(f"Success! Image shape: {processed_image.shape}") 
    # Output will be (224, 224, 3)

data=data.head(30)

# List to hold successful image arrays
valid_images = []
# List to hold the row numbers we need to delete
indices_to_drop = []

print("Starting thumbnail processing...")

for index, row in data.iterrows():
    url = row['thumbnail_link'] 
    
    # Call your padding function
    img_array = preprocess_thumbnail_pad(url)
    
    # Check if the download was successful
    if img_array is not None:
        valid_images.append(img_array)
    else:
        # If the image fails (404 error), mark this row's index for deletion
        indices_to_drop.append(index)

# 2. Convert the successful images into your final CNN input array
X_images = np.array(valid_images)

# 3. Drop the dead links from the original DataFrame in place
data.drop(index=indices_to_drop, inplace=True)

# 4. Reset the index so your rows are numbered cleanly (0, 1, 2...) again
data.reset_index(drop=True, inplace=True)

print("--- Processing Complete ---")
print(f"Total dead links dropped: {len(indices_to_drop)}")
print(f"Final dataset rows in 'df': {len(data)}")
print(f"Final Image Tensor Shape: {X_images.shape}")
print(data)
print(X_images[0][223])
print(X_images[0].shape)

# Steps below are to verify that not all values are 0 in X_images


# 1. Check the maximum value in the entire array
# Since we normalized by dividing by 255.0, pure white is 1.0. 
# This should print a number very close to 1.0!
print(f"Maximum pixel value: {X_images.max()}")

# 2. Check the average (mean) value
# If the array was entirely zeroes, this would be 0. 
# It should be a small decimal number greater than 0.
print(f"Average pixel value: {X_images.mean()}")

# 3. Look at a pixel right in the middle of the first image!
# Array slicing format: [image_index, y_coordinate, x_coordinate, rgb_channels]
# Since the image is 224x224, the exact center is at coordinates 112, 112.
center_pixel = X_images[0, 112, 112, :]
print(f"Center pixel RGB values: {center_pixel}")