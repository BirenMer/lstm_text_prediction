import numpy as np


def softmax(logits, axis=-1):
    """
    Compute the softmax activation function in a numerically stable way.
    
    Parameters:
    logits (numpy.ndarray): Input array (logits) before applying softmax.
    axis (int): Axis along which to compute softmax (default: last axis).

    Returns:
    numpy.ndarray: Softmax probabilities.
    """
    exp_logits = np.exp(logits - np.max(logits, axis=axis, keepdims=True))  # Subtract max for numerical stability
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)  # Normalize across the specified axis
