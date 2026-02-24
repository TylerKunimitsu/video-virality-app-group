import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(root_dir))

from state import data

from sentence_transformers import SentenceTransformer
import pandas as pd
import re

model = SentenceTransformer('all-MiniLM-L6-v2')

def text_cleaner(text):
    # 1. Lowercase for uniformity
    text = text.lower()
    
    # 2. Remove URLs (http/https/www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove Emojis and non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # 4. Remove Punctuation and Special Characters
    # [^a-z0-9\s] means: "Find anything that is NOT a letter, number, or space"
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # 5. Remove extra whitespace (turns "hello    world" into "hello world")
    text = " ".join(text.split())
    
    return text

def semantic_vectors(text):
    clean_text = text_cleaner(text)
    sem_vec = model.encode(clean_text)

    return sem_vec

data['description_semantics'] = data['description'].apply(semantic_vectors)

print(data['description_semantics'].head(1))