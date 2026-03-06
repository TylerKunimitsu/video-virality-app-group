import pandas as pd
import numpy as np
import io
from google.cloud import storage # pip install google-cloud-storage
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def build_visual_branch_custom(input_shape=(224, 224, 3)):
    # 1. Define the input layer
    thumbnail_input = Input(shape=input_shape, name="thumbnail_image")
    
    # 2. First Convolutional Block
    x = Conv2D(32, (3, 3), activation='relu')(thumbnail_input)
    x = MaxPooling2D((2, 2))(x)
    
    # 3. Second Convolutional Block
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # 4. Third Convolutional Block
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # 5. Flatten the 3D feature maps into a 1D vector
    x = Flatten()(x)
    
    # 6. Compress it into a dense feature vector
    visual_features = Dense(128, activation='relu', name="visual_features_dense")(x)
    
    return thumbnail_input, visual_features


# Read CSV directly from the bucket
df = pd.read_csv('gs://virality/processed_metadata.csv')

# Reading a massive .npy file from a bucket takes a few extra lines 
# because NumPy needs to download the byte stream:
client = storage.Client()
bucket = client.get_bucket('virality')
blob = bucket.blob('processed_images.npy')

# Download as a byte string and load into NumPy
byte_stream = io.BytesIO(blob.download_as_bytes())
X_images = np.load(byte_stream)



# Create the visual branch
img_input, img_features = build_visual_branch_transfer()