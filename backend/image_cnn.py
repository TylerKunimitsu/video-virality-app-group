import pandas as pd
import numpy as np
import io
from google.cloud import storage # pip install google-cloud-storage

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