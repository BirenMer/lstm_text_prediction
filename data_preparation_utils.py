import numpy as np


def prepare_text_data(text, seq_length=100, step=1):
    """
    Prepares text data for LSTM training with proper shapes and types.
    """
    if len(text) < seq_length:
        raise ValueError(f"Text length ({len(text)}) is shorter than sequence length ({seq_length}).")
    
    # Create character mappings
    chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    # vocab_size = len(char_to_idx)
    
    # Convert text to integers
    text_as_int = np.array([char_to_idx[c] for c in text])
    
    # Create sequences
    X = []
    Y = []
    for i in range(0, len(text_as_int) - seq_length, step):
        sequence = text_as_int[i:i + seq_length]
        target = text_as_int[i + 1:i + seq_length + 1]  # Next character for each position
        X.append(sequence)
        Y.append(target)
    
    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y, char_to_idx, idx_to_char
