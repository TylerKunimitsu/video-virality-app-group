# Character Length: Essential for predicting "truncation." On many mobile layouts, titles get cut off after ~50-60 characters. A title that hides the payoff behind an ellipsis (...) often sees lower performance.

# Has Number: This identifies "Listicles" (e.g., 7 Ways to...) or specific data (e.g., I made $5,000...). Numbers provide a cognitive "anchor" that makes the content feel more tangible.

# Caps Ratio: This measures "Intensity." A ratio of 0.1 is standard; a ratio of 0.8 suggests high-energy "clickbait." Your model will likely find a non-linear relationship here (too much is as bad as too little).

# Exclamation Count: This captures the "Hype Factor." It’s a great proxy for the creator's intended energy level.

# Is End in Question: This is the "Curiosity Gap" flag. Questions naturally prompt the human brain to seek an answer, which is the fundamental driver of a click.

# POS (Part of Speech) Diversity
# Highly informative titles usually have a specific "shape" in their grammar. You can use a library like spaCy to tag the words and look for:
# Concrete Nouns: "iPhone," "Python," "Pizza."
# Proper Nouns: Brand names or specific locations.
# Action Verbs: "Exploded," "Crushed," "Solved."

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(root_dir))

from state import data

import pandas as pd
import re
import spacy

def capsRatio(text):
    sum = 0
    for c in text:
        if c.isupper(): sum+=1

    return sum/len(text) if len(text) > 0 else 0

data['title_cLength'] = data['title'].str.len()
data['title_hasNumber'] = data['title'].str.contains(r'\d').astype(int)
data['title_capsRatio'] = data['title'].apply(lambda x: capsRatio(x))
data['title_exCount'] = data['title'].str.count('!')
data['title_endInQ'] = data['title'].apply(lambda x: 1 if len(x) > 0 and x.strip()[-1]=='?' else 0)

# Load small english model
nlp = spacy.load("en_core_web_sm")

def infoDensity(text):
    if pd.isnull(text) or text.strip() == "":
        return 0.0
    
    doc = nlp(text)

    content_tags = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

    total_words = [token for token in doc if not token.is_punct]
    content_words = [token for token in total_words if token.pos_ in content_tags]

    return len(content_words)/len(total_words)

data['title_infoDensity'] = data['title'].apply(infoDensity).astype(float)

#print(data[['title_cLength', 'title_hasNumber', 'title_capsRatio', 'title_exCount', 'title_endInQ', 'title_infoDensity']].head(1))