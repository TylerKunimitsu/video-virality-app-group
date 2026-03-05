from state import data

import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np

# Setting up tf inputs

#x1 -> image
x1_input = layers.Input(shape=..., name='Group1_Input')
x1 = layers.Dense(...)(x1_input)

#x2 -> semantics
x2_input = layers.Input(shape=(384,), name='Group2_Input')
x2 = layers.Dense(128, activation='relu')(x2_input)
x2 = layers.Dropout(0.2)(x2)
x2 = layers.Dense(64, activation='relu', name='Group2_Final')(x2)

#x3 -> 1d features
x3_input = layers.Input(shape=(12,), name='Group3_Input')
x3 = layers.Dense(64, activation='relu')(x3_input)
x3 = layers.BatchNormalization()(x3)
x3 = layers.Dropout(0.2)(x3)
x3 = layers.Dense(32, activation='relu')(x3)
x3 = layers.BatchNormalization()(x3)

#x4 -> high d & categorical


# Merging nodes (concatenating)

merged = layers.Concatenate()([x1, x2, x3, x4])

x = layers.Dense(32, activation='relu')(merged)
output = layers.Dense(2, activation='linear', name='Results')(x)

# Defining model

model = models.Model(inputs=[x1_input, x2_input, x3_input, x4_input], outputs=output)
model.compile(optimizer='adam', loss='mse')

# Fitting data into the model
# Turning data into correct forms (whatever input shape defined above)

#x1 operations
x_group1 = ...

#x2
x_group2 = np.stack(data['description_semantics']) # Turns 1d array into (#rows, 384)

#x3
x_group3 = pd.concat([data[['title_cLength', 'title_hasNumber', 'title_capsRatio', 'title_exCount', 'title_endInQ', 'title_infoDensity', 'tags_count', 'tags_title_overlapRatio', 'description_tokens']], data['description_sentiment'][['Negative', 'Neutral', 'Positive']]], axis=1).values

#y
y = data[['views', 'likes']]

# Training the model
model.fit(
    x={'Group1_Input': x_group1, 'Group2_Input': x_group2, 'Group3_Input': x_group3, 'Group4_Input': x_group4},
    y=y,
    epochs=100
)