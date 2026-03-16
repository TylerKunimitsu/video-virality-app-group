import sys
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# ==========================================
# 1. REUSABLE FUNCTIONS
# ==========================================

def preprocess(text):
    if not isinstance(text, str): 
        return ""
    
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_sentiment(raw_text):
    # Safety check: If description is missing, default to 100% Neutral
    if not isinstance(raw_text, str) or raw_text.strip() == "":
        return {'Negative': 0.0, 'Neutral': 1.0, 'Positive': 0.0}

    text = preprocess(raw_text)

    encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded)

    # Use .logits to safely extract the model's math output
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)

    labels = ['Negative', 'Neutral', 'Positive']

    results = {labels[i]: np.round(float(scores[i]), 4) for i in range(len(labels))}
    return results

def count_tokens(text):
    # Safety check: return an empty list if there is no text
    if not isinstance(text, str) or text.strip() == "":
        return []
    
    # truncation=False allows it to count all tokens, even if it goes over 512!
    return tokenizer.encode(text, add_special_tokens=True, truncation=False)

# ==========================================
# 2. DATASET SCRIPT (Hidden from imports)
# ==========================================

if __name__ == "__main__":
    from state import data
    print("Calculating sentiment and token counts (this may take a minute)...")
    
    data['description_sentiment'] = data['description'].apply(analyze_sentiment)
    data['description_tokens'] = data['description'].apply(count_tokens)
    data['description_tokenCounts'] = data['description_tokens'].apply(len)
    
    print("Sentiment and tokens calculated successfully!")

# import sys
# from pathlib import Path

# root_dir = Path(__file__).resolve().parent.parent.parent

# sys.path.insert(0, str(root_dir))

# from state import data
# #print(data.head(1))

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from scipy.special import softmax
# import numpy as np

# model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# def preprocess(text):
#     new_text = []
#     for t in text.split(" "):
#         t = '@user' if t.startswith('@') and len(t) > 1 else t
#         t = 'http' if t.startswith('http') else t
#         new_text.append(t)
#     return " ".join(new_text)

# def analyze_sentiment(raw_text):
#     text = preprocess(raw_text)

#     encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

#     output = model(**encoded)

#     scores = output[0][0].detach().numpy()
#     scores = softmax(scores)

#     labels = ['Negative', 'Neutral', 'Positive']

#     results = {labels[i]: np.round(float(scores[i]), 4) for i in range(len(labels))}
#     return results

# def count_tokens(text):
#     return tokenizer.encode(text, add_special_tokens=True)

# data['description_sentiment'] = data['description'].apply(analyze_sentiment)
# data['description_tokens'] = data['description'].apply(count_tokens)
# data['description_tokenCounts'] = data['description_tokens'].apply(len)

# #print(data[['description_sentiment', 'description_tokens']].head(1))