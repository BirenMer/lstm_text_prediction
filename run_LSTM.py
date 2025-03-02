import datetime
import numpy as np
import matplotlib.pyplot as plt
from LSTM import LSTM
from LSTM_text_prediction.activation_function import softmax
from LSTM_text_prediction.model_utils import one_hot_encode_batch
from activation_function.Sigmoid import Sigmoid
from activation_function.Tanh import Tanh
from optimizers.optimizerSGD import OptimizerSGD
from optimizers.optimizerSGDLSTM import OptimizerSGDLSTM
from layers.dense_layer import DenseLayer


def RunMyLSTM(X, Y, vocab_size, char_to_idx, idx_to_char, n_epoch=500, n_neurons=500, learning_rate=1e-5, 
              decay=0, momentum=0, batch_size=1024):
    # Initialize models
    lstm = LSTM(n_neurons=n_neurons, n_features=vocab_size)
    dense = DenseLayer(n_neurons, vocab_size)
    optimizer_lstm = OptimizerSGDLSTM(learning_rate=learning_rate, decay=decay, momentum=momentum)
    optimizer_dense = OptimizerSGD(learning_rate=learning_rate, decay=decay, momentum=momentum)
    
    X, Y = np.array(X), np.array(Y)
    n_samples, seq_length = X.shape
    # print(seq_length)
    n_samples=n_samples-247
    X_one_hot = one_hot_encode_batch(X, vocab_size)
    
    Y_one_hot = one_hot_encode_batch(Y, vocab_size)
    
    
    losses = []
    for epoch in range(n_epoch):
        start_time = datetime.datetime.now()
        print(f"\rEpoch {epoch}/{n_epoch}",end='\n',flush=True)
        loss = 0
        indices = np.random.permutation(n_samples)
        X_shuffled = X_one_hot[indices]
        Y_shuffled = Y_one_hot[indices]
        print(n_samples)
        
        for i in range(0, n_samples, batch_size):
            # print(n_samples)
            print(f"\r Processing sample {i}",end='',flush=True)
            end_idx = min(i + batch_size, n_samples)
            X_batch = X_shuffled[i:end_idx]
            Y_batch = Y_shuffled[i:end_idx]
            X_batch_one_hot = np.eye(vocab_size, dtype=np.float32)[X_batch]
            Y_batch_one_hot = np.eye(vocab_size, dtype=np.float32)[Y_batch]

            # Forward pass
            lstm_out = lstm.forward(X_batch_one_hot)
            # dense_out = dense.forward(lstm_out.reshape(end_idx - i, seq_length * lstm.n_neurons))
            dense_input = lstm_out[:, -1, :]  # Take only the last time step of LSTM output
            dense_out = dense.forward(dense_input)  # Shape should match Dense layer input
            # dense_out_3d = dense_out.reshape(end_idx - i, seq_length, vocab_size)
            
            # Compute softmax and loss
            # probs = softmax(dense_out_3d, axis=-1)
            probs = softmax(dense_out, axis=-1)
            log_probs = np.log(probs + 1e-10)
            # loss += -np.mean(Y_batch * log_probs)
            loss += -np.mean(Y_batch_one_hot[:, -1, :] * log_probs)

            
            # Backward pass
            # dlogits = probs - Y_batch_one_hot
            dlogits = probs - Y_batch_one_hot[:, -1, :]
            dlogits_flat = dlogits.reshape(-1, vocab_size)
            dense.backward(dlogits_flat)
            dlstm_out = np.zeros((batch_size, seq_length, lstm.n_neurons))
            # dlstm_out[:, -1, :] = dense.dinputs  # Assign only the last timestep
            dlstm_out[:dense.dinputs.shape[0], -1, :] = dense.dinputs

            # dlstm_out = dense.dinputs.reshape(-1, seq_length, lstm.n_neurons)
            lstm.backward(dlstm_out)
            
            # Update parameters
            optimizer_dense.update_params(dense)
            optimizer_lstm.update_params(lstm)
        
        epoch_loss = loss / (n_samples // batch_size)
        losses.append(epoch_loss)
        
        print(f"Epoch {epoch+1}/{n_epoch}, Loss: {epoch_loss:.4f}")

        end_time = datetime.datetime.now()
        print(
            rf"total_time for {epoch} files is : {end_time-start_time}"
        )
    # Plot training loss
    plt.plot(losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    
    return lstm, [dense], char_to_idx, idx_to_char
