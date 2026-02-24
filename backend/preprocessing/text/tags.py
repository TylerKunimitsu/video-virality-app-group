import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(root_dir))

from state import data

from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import re

def get_tags(text):
    if not text: return []

    raw_tags = text.strip("[]").split('|')
    simplified_tags = []
    for tag in raw_tags:
        clean_tag = tag.lower().strip().replace('"', '')
        if clean_tag:
            simplified_tags.append(clean_tag)

    return simplified_tags

data['tags_list'] = data['tags'].apply(get_tags)

# Don't know if main_tag is effective because a lot of videos only contain one tag that is irrelevant to the video title (like channel name), can be deleted if the accuracy is low.
def get_tag_features(row):
    tags = [tag for tag in row['tags_list'] if tag.strip()] # the if statement makes the array only store non-empty string
    if not tags: return 'none', 0.0

    clean_title = row['title'].encode("ascii", "ignore").decode() # This method can only be used for English videos. Also, the overlap count will have this English Trap, but I think it's ok since most English videos are targeted to English users.
    title_words = set(clean_title.lower().split())

    main_tag = tags[0] # Default to first tag since already detected if the tags are junks
    all_tag_words = set() # Flatten all tags into one set of unique words
    max_overlap = 0

    # main_tag is determined by how much it overlaps with the title. If this sounds redundant, we can try the popular tag main_tag approach.
    for tag in tags:
        # Split multi-word tags like "apple watch" into ["apple", "watch"]
        tag_words = set(tag.split())
        if not tag_words: continue

        all_tag_words.update(tag_words)

        overlap = len(tag_words.intersection(title_words))
        if overlap > max_overlap:
            max_overlap = overlap
            main_tag = tag

    if not all_tag_words: return 'none', 0.0

    overlap_words = all_tag_words.intersection(title_words)
    ratio = len(overlap_words)/len(all_tag_words)
    
    return main_tag, ratio # Using this method to calculate overlap ratio detects redundancy of the tags

data['tags_count'] = data['tags_list'].apply(len)
data['main_tag'], data['tags_title_overlapRatio'] = zip(*data.apply(get_tag_features, axis=1)) # Unpacks two returns into two features

# Getting the binarizer for the tags that are in top 500 tags_list

all_tags = [tag for tags in data['tags_list'] for tag in tags] # Nested looping for tags_list
tag_counts = Counter(all_tags)
top500tags = [tag for tag, count in tag_counts.most_common(500)]
top500set = set(top500tags)

data['tags_filtered'] = data['tags_list'].apply(lambda x: [tag for tag in x if tag in top500set])

mlb = MultiLabelBinarizer(classes=top500tags)
tag_binarized = mlb.fit_transform(data['tags_filtered']) # Returns a matrix (2D NumPy array)

data['topTagsBinarized'] = list(tag_binarized)

print(data['tags_list'].head(5))
print(data[['tags_count', 'main_tag', 'tags_title_overlapRatio', 'topTagsBinarized']].head(5))