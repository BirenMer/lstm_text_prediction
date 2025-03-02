import os
import pickle
import numpy as np

def save_model(lstm, dense_layers, char_to_idx, idx_to_char, model_path):
    with open(os.path.join(model_path, "model.pkl"), 'wb') as f:
        pickle.dump({'lstm': lstm, 'dense_layers': dense_layers, 'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f)

def load_model(model_path):
    with open(os.path.join(model_path, "model.pkl"), 'rb') as f:
        data = pickle.load(f)
    return data['lstm'], data['dense_layers'], data['char_to_idx'], data['idx_to_char']

def one_hot_encode_batch(X_batch, vocab_size):
    """
    Instead of full one-hot encoding, use integer encoding to reduce memory usage.
    """
    if len(X_batch.shape) != 2:
        raise ValueError(f"Expected 2D input array, got shape {X_batch.shape}")
    
    if np.any((X_batch < 0) | (X_batch >= vocab_size)):
        raise ValueError("Values in X_batch must be in the range [0, vocab_size-1]")
    
    return np.array(X_batch, dtype=np.int32)  # Store indices instead of one-hot
