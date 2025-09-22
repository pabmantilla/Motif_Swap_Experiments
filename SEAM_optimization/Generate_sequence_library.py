from numpy.ctypeslib import as_array
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import time
import random
import h5py
import logging
import warnings

# Suppress TensorFlow warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '3'  # Suppress oneDNN messages
tf.get_logger().setLevel('ERROR')

# Redirect stderr to suppress C++ backend messages
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Load in keras EvoAug student model
model = tf.keras.models.load_model("../EvoAug_keras_model")

# Check is assets_deepstarr folder exists with deepstarr_data.h5 file
if not os.path.exists("assets_deepstarr"):
    os.makedirs("assets_deepstarr")

# Download deepstarr_data.h5 file if it doesn't exist
if not os.path.exists("assets_deepstarr/deepstarr_data.h5"):
    os.system("wget 'https://www.dropbox.com/scl/fi/cya4ntqk2o8yftxql52lu/deepstarr_data.h5?rlkey=5ly363vqjb3vaw2euw2dhsjo3&st=6eod6fg8&dl=1' -O assets_deepstarr/deepstarr_data.h5")

# load test DeepsTARR test data from .h5 files 
deepstarr_data = h5py.File("assets_deepstarr/deepstarr_data.h5", "r")


# Test model on test data
X_test = deepstarr_data["x_test"][:]
y_test = deepstarr_data["y_test"][:]

print(X_test.shape)
print(y_test.shape)


X_test = np.transpose(X_test, (0, 2, 1))
print(f"Transposed X_test shape: {X_test.shape}")

print("Making predictions in batches...")
batch_size = 1000  # Process 1000 sequences at a time
predictions_list = []

for i in range(0, len(X_test), batch_size):
    batch_end = min(i + batch_size, len(X_test))
    batch_X = X_test[i:batch_end]
    if i % 10000 == 0:
        print(f"Processing batch {i//batch_size + 1}/{(len(X_test) + batch_size - 1)//batch_size} ({batch_end - i} sequences)")
    
    batch_predictions = model(batch_X)
    predictions_list.append(batch_predictions.numpy())

# Concatenate all predictions
predictions = np.concatenate(predictions_list, axis=0)
print(f"Total predictions shape: {predictions.shape}")

# Calculate metrics manually
from scipy.stats import pearsonr

# MSE for Dev and Hk tasks
dev_mse = np.mean((predictions[:, 0] - y_test[0, :])**2)
hk_mse = np.mean((predictions[:, 1] - y_test[1, :])**2)

# Pearson correlation for Dev and Hk tasks
dev_pearson, _ = pearsonr(predictions[:, 0], y_test[0, :])
hk_pearson, _ = pearsonr(predictions[:, 1], y_test[1, :])

print(f"Dev MSE: {dev_mse:.6f}")
print(f"Hk MSE: {hk_mse:.6f}")
print(f"Dev Pearson r: {dev_pearson:.6f}")
print(f"Hk Pearson r: {hk_pearson:.6f}")

## filter data on actual labels
high_cut = 2
low_cut = -1
mid_range = [0,0.5]

# Initialize counters and lists
target_per_bin = 5
high_count = 0
low_count = 0
mid_count = 0

indices = []
sequences_one_hot = []
bins = []
dev_labels = []
dev_predictions = []

print(y_test.shape)
print(f"Target: {target_per_bin} samples per bin")

# Keep sampling until each bin has 5 samples
while high_count < target_per_bin or low_count < target_per_bin or mid_count < target_per_bin:
    # Pick a random index
    random_idx = np.random.randint(0, len(predictions))
    hk_value = y_test[1, random_idx]
    
    if hk_value > high_cut and high_count < target_per_bin:
        indices.append(random_idx)
        sequences_one_hot.append(X_test[random_idx])
        bins.append("high")
        dev_labels.append(hk_value)
        dev_predictions.append(predictions[random_idx, 1])
        high_count += 1
        
    elif hk_value < low_cut and low_count < target_per_bin:
        indices.append(random_idx)
        sequences_one_hot.append(X_test[random_idx])
        bins.append("low")
        dev_labels.append(hk_value)
        dev_predictions.append(predictions[random_idx, 1])
        low_count += 1
        
    elif 0 < hk_value < 0.5 and mid_count < target_per_bin:
        indices.append(random_idx)
        sequences_one_hot.append(X_test[random_idx])
        bins.append("mid")
        dev_labels.append(hk_value)
        dev_predictions.append(predictions[random_idx, 1])
        mid_count += 1

from tangermeme.utils import characters
import torch

seq_tensors = [torch.from_numpy(as_array(seq)) for seq in sequences_one_hot]
sequences = [characters(seq) for seq in seq_tensors]
print(sequences[0])

test_library_df = pd.DataFrame({
    "test_index": indices, 
    "sequence": sequences,
    "bin": bins, 
    "hk_label": dev_labels, 
    "hk_prediction": dev_predictions,
    "seq_one_hot": sequences_one_hot
})

print(test_library_df.keys())
print(test_library_df[["test_index", "hk_label", "hk_prediction", "bin"]].head())

print(test_library_df["seq_one_hot"][0].shape)
print(type(test_library_df["sequence"][0]))
print(test_library_df[["bin", "hk_label", "hk_prediction"]].groupby("bin").mean())
print(f"High count: {test_library_df[test_library_df['bin'] == 'high'].shape[0]}")
print(f"Low count: {test_library_df[test_library_df['bin'] == 'low'].shape[0]}")
print(f"Mid count: {test_library_df[test_library_df['bin'] == 'mid'].shape[0]}")

# save test_library_df to csv
test_library_df.to_csv("test_library_df.csv", index=False)


