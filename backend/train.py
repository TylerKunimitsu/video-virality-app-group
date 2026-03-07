from state import data

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0 # for cnn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import h5py # for reading h5 file


# Setting up tf inputs

#x1 -> image
x1_input = layers.Input(shape=(224, 224, 3), name='Group1_Input')
base_model = EfficientNetB0(weights='imagenet', include_top=False)
base_model.trainable = False # Freeze pre-trained weights
x1 = base_model(x1_input)
x1 = layers.GlobalAveragePooling2D()(x1)
x1 = layers.Dense(128, activation='relu', name='Group1_Final')(x1)

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

#x4 -> binarizer
x4_input = layers.Input(shape=(500,), name='Group4_Input')
x4 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.01))(x4_input)

#x5 -> main_tag
x5_vocab = data['main_tag'].unique() # the "vocab list" for encoding the string
x5_input = layers.Input(shape=(1,), dtype='string', name='Group5_Input')
x5_encoded = layers.StringLookup(vocabulary=x5_vocab)(x5_input) # encode the string
x5_embedding = layers.Embedding(input_dim=len(x5_vocab)+1, output_dim=4)(x5_encoded) # pass into an embedding layer to output a 4d vector (but in [[]] form) (like semantics for the correlations of main tags)
x5_flat = layers.Flatten()(x5_embedding) # make matrix outputted by the embedding layer to 1d

# Merging nodes (concatenating)

merged = layers.Concatenate()([x1, x2, x3, x4, x5_flat])

x = layers.Dense(32, activation='relu')(merged)
output = layers.Dense(2, activation='linear', name='Results')(x)

# Defining model

model = models.Model(inputs=[x1_input, x2_input, x3_input, x4_input, x5_input], outputs=output)
model.compile(optimizer='adam', loss='mse')

# Fitting data into the model
# Turning data into correct forms (whatever input shape defined above)

# Open the HDF5 vault in read mode
hf = h5py.File('processed_images.h5', 'r')

#x1 operations
# Notice we DO NOT use [:] here. We just pass the dataset reference.
x_group1 = hf['images']

#x2
x_group2 = np.stack(data['description_semantics']) # Turns 1d array into (#rows, 384)

#x3
x3_sentiment = data['description_sentiment'].apply(pd.Series)
x3_df = pd.concat([data[['title_cLength', 'title_hasNumber', 'title_capsRatio', 'title_exCount', 'title_endInQ', 'title_infoDensity', 'tags_count', 'tags_title_overlapRatio', 'description_tokenCounts']], x3_sentiment[['Negative', 'Neutral', 'Positive']]], axis=1)
scaler = StandardScaler()
x_group3 = scaler.fit_transform(x3_df.values)

#x4
x_group4 = np.array(data['topTagsBinarized'].tolist())

#x5
x_group5 = data['main_tag'].values

#y
y = np.log1p(data[['views', 'likes']].values)

# Training the model
model.fit(
    x={'Group1_Input': x_group1, 'Group2_Input': x_group2, 'Group3_Input': x_group3, 'Group4_Input': x_group4, 'Group5_Input': x_group5},
    y=y,
    epochs=100,
    batch_size=32,
    validation_split=0.2 # monitoring overfitting
)

# Close the images file when training finishes
hf.close()