import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from LSTM import LSTM
from activation_function.softmax import softmax
from optimizers.optimizerSGD import OptimizerSGD
from optimizers.optimizerSGDLSTM import OptimizerSGDLSTM
from layers.dense_layer import DenseLayer

logging.basicConfig(filename="epoch_update.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_LSTM(X, Y, vocab_size, char_to_idx, idx_to_char, n_epoch=500, n_neurons=500, learning_rate=1e-5, 
              decay=0, momentum=0, batch_size=1024):
    # Initialize models
    lstm = LSTM(n_neurons=n_neurons, n_features=vocab_size)
    dense = DenseLayer(n_neurons, vocab_size)
    optimizer_lstm = OptimizerSGDLSTM(learning_rate=learning_rate, decay=decay, momentum=momentum)
    optimizer_dense = OptimizerSGD(learning_rate=learning_rate, decay=decay, momentum=momentum)
    
    X = np.array(X)
    Y = np.array(Y)
    n_samples, seq_length = X.shape
    
    losses = []
    print(f"Starting training with {n_samples} samples...")
    logging.info(f"Number of samples : {n_samples}")

    for epoch in range(n_epoch):
        print(f"Currently at epoch {epoch}")

        start_time = datetime.datetime.now()
        loss_total = 0
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        
        for i in range(0, n_samples, batch_size):
            print(f"\rProcessing  {i}/{n_samples}",end="",flush=True)
            end_idx = min(i + batch_size, n_samples)
            X_batch = X_shuffled[i:end_idx]
            Y_batch = Y_shuffled[i:end_idx]
            current_batch_size = end_idx - i
            
            # One-hot encode batches on the fly
            X_batch_one_hot = np.eye(vocab_size, dtype=np.float32)[X_batch]
            Y_batch_one_hot = np.eye(vocab_size, dtype=np.float32)[Y_batch]
            
            # Forward pass
            lstm_out = lstm.forward(X_batch_one_hot)
            dense_input = lstm_out.reshape(-1, lstm.n_neurons)
            dense_out = dense.forward(dense_input)
            probs = softmax(dense_out.reshape(current_batch_size, seq_length, vocab_size), axis=-1)
            
            # Compute loss
            log_probs = np.log(probs + 1e-10)
            loss = -np.mean(np.sum(Y_batch_one_hot * log_probs, axis=-1))
            loss_total += loss * current_batch_size  # Weighted by batch size
            
            # Backward pass
            dlogits = probs - Y_batch_one_hot
            dense.backward(dlogits.reshape(-1, vocab_size))
            dlstm_out = dense.dinputs.reshape(current_batch_size, seq_length, lstm.n_neurons)
            lstm.backward(dlstm_out)
            
            # Update parameters
            optimizer_dense.update_params(dense)
            optimizer_lstm.update_params(lstm)
        
        epoch_loss = loss_total / n_samples
        losses.append(epoch_loss)
        
        print(f"Epoch {epoch+1}/{n_epoch}, Loss: {epoch_loss:.4f}")
        end_time = datetime.datetime.now()
        print(rf"Total time for epoch {epoch}: {end_time - start_time}")
        logging.info(f"Done for epoch {epoch} in {end_time - start_time}")

    # Plot training loss
    plt.plot(losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    
    return lstm, [dense], char_to_idx, idx_to_char