import numpy as np
from activation_function.Tanh import Tanh
from activation_function.Sigmoid import Sigmoid

class LSTM:
    def __init__(self, n_neurons, n_features):
        self.n_neurons = n_neurons
        self.n_features = n_features

        # Initialize weights with Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (n_features + n_neurons))
        
        # Forget gate parameters
        self.Uf = np.random.randn(n_neurons, n_features) * scale
        self.Wf = np.random.randn(n_neurons, n_neurons) * scale
        self.bf = np.zeros((n_neurons, 1))

        # Input gate parameters
        self.Ui = np.random.randn(n_neurons, n_features) * scale
        self.Wi = np.random.randn(n_neurons, n_neurons) * scale
        self.bi = np.zeros((n_neurons, 1))

        # Output gate parameters
        self.Uo = np.random.randn(n_neurons, n_features) * scale
        self.Wo = np.random.randn(n_neurons, n_neurons) * scale
        self.bo = np.zeros((n_neurons, 1))

        # Cell candidate parameters
        self.Ug = np.random.randn(n_neurons, n_features) * scale
        self.Wg = np.random.randn(n_neurons, n_neurons) * scale
        self.bg = np.zeros((n_neurons, 1))

    def lstm_cell(self, xt, ht_prev, ct_prev):
        # Initialize activation functions
        sigmoid = Sigmoid()
        tanh = Tanh()
        
        # Compute gates
        ft = sigmoid.forward(np.dot(self.Uf, xt) + np.dot(self.Wf, ht_prev) + self.bf)
        it = sigmoid.forward(np.dot(self.Ui, xt) + np.dot(self.Wi, ht_prev) + self.bi)
        ot = sigmoid.forward(np.dot(self.Uo, xt) + np.dot(self.Wo, ht_prev) + self.bo)
        
        # Compute cell candidate
        c_tilde = tanh.forward(np.dot(self.Ug, xt) + np.dot(self.Wg, ht_prev) + self.bg)
        
        # Update cell state
        # print(f"ft: {ft}, ct_prev: {ct_prev}, c_tilde: {c_tilde}")
        ct = ft * ct_prev + it * c_tilde
        
        # Compute hidden state
        ht = ot * tanh.forward(ct)
        
        return ht, ct, c_tilde, ft, it, ot

    def forward(self, X):
        batch_size, seq_length, n_features = X.shape
        
        if n_features != self.n_features:
            raise ValueError(f"Input feature size {n_features} does not match expected size {self.n_features}")

        # Initialize states
        self.H = np.zeros((batch_size, seq_length + 1, self.n_neurons))
        self.C = np.zeros((batch_size, seq_length + 1, self.n_neurons))
        self.gates = {
            'C_tilde': np.zeros((batch_size, seq_length, self.n_neurons)),
            'F': np.zeros((batch_size, seq_length, self.n_neurons)),
            'I': np.zeros((batch_size, seq_length, self.n_neurons)),
            'O': np.zeros((batch_size, seq_length, self.n_neurons))
        }
        
        # Store input for backprop
        self.X = X
        
        # Process each timestep
        for t in range(seq_length):
            for b in range(batch_size):
                xt = X[b, t].reshape(-1, 1)
                ht_prev = self.H[b, t].reshape(-1, 1)
                ct_prev = self.C[b, t].reshape(-1, 1)
                
                ht, ct, c_tilde, ft, it, ot = self.lstm_cell(xt, ht_prev, ct_prev)
                
                self.H[b, t + 1] = ht.reshape(-1)
                self.C[b, t + 1] = ct.reshape(-1)
                self.gates['C_tilde'][b, t] = c_tilde.reshape(-1)
                self.gates['F'][b, t] = ft.reshape(-1)
                self.gates['I'][b, t] = it.reshape(-1)
                self.gates['O'][b, t] = ot.reshape(-1)
        
        return self.H[:, 1:]  # Return all hidden states except initial state

    def backward(self, dH):
        batch_size, seq_length, _ = dH.shape
        dUf, dWf, dbf = np.zeros_like(self.Uf), np.zeros_like(self.Wf), np.zeros_like(self.bf)
        dUi, dWi, dbi = np.zeros_like(self.Ui), np.zeros_like(self.Wi), np.zeros_like(self.bi)
        dUo, dWo, dbo = np.zeros_like(self.Uo), np.zeros_like(self.Wo), np.zeros_like(self.bo)
        dUg, dWg, dbg = np.zeros_like(self.Ug), np.zeros_like(self.Wg), np.zeros_like(self.bg)
        
        for b in range(batch_size):
            delta_h = np.zeros((self.n_neurons, 1))
            delta_c = np.zeros((self.n_neurons, 1))
            for t in reversed(range(seq_length)):
                xt = self.X[b, t].reshape(-1, 1)
                ht_prev = self.H[b, t].reshape(-1, 1)
                ft = self.gates['F'][b, t].reshape(-1, 1)
                it = self.gates['I'][b, t].reshape(-1, 1)
                ot = self.gates['O'][b, t].reshape(-1, 1)
                c_tilde = self.gates['C_tilde'][b, t].reshape(-1, 1)
                ct = self.C[b, t+1].reshape(-1, 1)
                ct_prev = self.C[b, t].reshape(-1, 1)

                dht = dH[b, t].reshape(-1, 1) + delta_h
                tanh = Tanh()
                dct = delta_c + dht * ot * tanh.backward(tanh.forward(ct))
                
                dot = (dht * tanh.forward(ct) + dct * c_tilde) * (ot * (1 - ot))
                dit = dct * c_tilde * (it * (1 - it))
                dft = dct * ct_prev * (ft * (1 - ft))
                dc_tilde = dct * it * (1 - c_tilde**2)
                
                # Compute gradients and accumulate
                dUf += np.dot(dft, xt.T)
                dWf += np.dot(dft, ht_prev.T)
                dbf += dft.sum(axis=0).reshape(-1,1)
                
                delta_h = (np.dot(self.Wf.T, dft) 
                        + np.dot(self.Wi.T, dit) 
                        + np.dot(self.Wo.T, dot)
                        + np.dot(self.Wg.T, dc_tilde))
                
                delta_c = dct * ft
            
        self.gradients = {
            'Uf': dUf, 'Wf': dWf, 'bf': dbf,
            'Ui': dUi, 'Wi': dWi, 'bi': dbi,
            'Uo': dUo, 'Wo': dWo, 'bo': dbo,
            'Ug': dUg, 'Wg': dWg, 'bg': dbg
        }