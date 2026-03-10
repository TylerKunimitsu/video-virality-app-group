from state import data

import runpy
import pandas as pd
import numpy as np

#runpy.run_path('preprocessing/thumbnail/image.py')

data['title'] = data['title'].fillna('')
data['description'] = data['description'].fillna('')
data['tags'] = data['tags'].fillna('')

data.dropna(subset=['views', 'likes'], inplace=True)
# Also, good practice to reset the index after dropping rows
data.reset_index(drop=True, inplace=True)

runpy.run_path('preprocessing/text/sentiment_tokens.py')
runpy.run_path('preprocessing/text/semantics.py')
runpy.run_path('preprocessing/text/title.py')
runpy.run_path('preprocessing/text/tags.py')

# 1. Identify the columns that are safe (Numerical/String)
cols_to_save = [
    'video_id', 'trending_date', 'title', 'channel_title', 'category_id', 
    'publish_time', 'tags', 'views', 'likes', 'dislikes', 'comment_count', 
    'thumbnail_link', 'comments_disabled', 'ratings_disabled', 
    'video_error_or_removed', 'description',
    'title_cLength', 'title_hasNumber', 'title_capsRatio', 'title_exCount', 
    'title_endInQ', 'title_infoDensity', 'tags_count', 
    'tags_title_overlapRatio', 'main_tag', 'description_tokenCounts'
]

# 2. Add the sentiment columns (assuming you expanded them already)
# If you haven't expanded them yet, do:
sentiment_cols = data['description_sentiment'].apply(pd.Series)
data = pd.concat([data, sentiment_cols], axis=1)
cols_to_save.extend(['Negative', 'Neutral', 'Positive'])

# 3. Save only the safe columns
data[cols_to_save].to_csv('processedUSvideos.csv', index=False)

print("Arrays saved successfully to processedUSvideos.csv!")

# --- Saving Semantic Vectors (Group 2) ---
# We stack the list of 384-D arrays into one large (N, 384) matrix
semantics_matrix = np.stack(data['description_semantics'].values)
np.save('description_semantics.npy', semantics_matrix)

# --- Saving Binarized Tags (Group 4) ---
# We convert the list of 500-D arrays into one large (N, 500) matrix
tags_matrix = np.array(data['topTagsBinarized'].tolist())
np.save('top_tags_binarized.npy', tags_matrix)

print("Arrays saved successfully as .npy files!")

#runpy.run_path('train.py')