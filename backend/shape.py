import numpy as np
import pandas as pd
binary = np.load('top_tags_binarized.npy')
sem = np.load('description_semantics.npy')
print(binary.dtype)
print(sem.dtype)