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
        
        # Initialize gradients
        dUf = np.zeros_like(self.Uf)
        dWf = np.zeros_like(self.Wf)
        dbf = np.zeros_like(self.bf)
        dUi = np.zeros_like(self.Ui)
        dWi = np.zeros_like(self.Wi)
        dbi = np.zeros_like(self.bi)
        dUo = np.zeros_like(self.Uo)
        dWo = np.zeros_like(self.Wo)
        dbo = np.zeros_like(self.bo)
        dUg = np.zeros_like(self.Ug)
        dWg = np.zeros_like(self.Wg)
        dbg = np.zeros_like(self.bg)
        
        # Initialize previous deltas
        delta_h_prev = np.zeros((self.n_neurons, 1))
        delta_c_prev = np.zeros((self.n_neurons, 1))
        
        # Loop over each batch
        for b in range(batch_size):
            delta_h = np.zeros((self.n_neurons, 1))
            delta_c = np.zeros((self.n_neurons, 1))
            # Process each timestep in reverse
            for t in reversed(range(seq_length)):

                # Retrieve inputs and states
                xt = self.X[b, t].reshape(-1, 1)
                ft = self.gates['F'][b, t].reshape(-1, 1)
                it = self.gates['I'][b, t].reshape(-1, 1)
                ot = self.gates['O'][b, t].reshape(-1, 1)
                c_tilde = self.gates['C_tilde'][b, t].reshape(-1, 1)
                ct_prev = self.C[b, t].reshape(-1, 1)
                ht_prev = self.H[b, t].reshape(-1, 1)
                ct = self.C[b, t + 1].reshape(-1, 1)
                
                # Current hidden state gradient
                current_dh = dH[b, t].reshape(-1, 1)
                delta_h = current_dh + delta_h_prev
                    
                # Compute cell state gradient
                tanh_ct = np.tanh(ct)
                grad_tanh_ct = 1 - tanh_ct ** 2
                delta_c = delta_c_prev + delta_h * ot * grad_tanh_ct
                
                # Compute gate gradients
                dft = delta_c * ct_prev * ft * (1 - ft)
                dit = delta_c * c_tilde * it * (1 - it)
                dot = delta_h * tanh_ct * ot * (1 - ot)
                dc_tilde = delta_c * it * (1 - c_tilde ** 2)
                
                # Update parameter gradients
                dUf += np.dot(dft, xt.T)
                dWf += np.dot(dft, ht_prev.T)
                dbf += dft.sum(axis=0)
                
                dUi += np.dot(dit, xt.T)
                dWi += np.dot(dit, ht_prev.T)
                dbi += dit.sum(axis=0)
                
                dUo += np.dot(dot, xt.T)
                dWo += np.dot(dot, ht_prev.T)
                dbo += dot.sum(axis=0)
                
                dUg += np.dot(dc_tilde, xt.T)
                dWg += np.dot(dc_tilde, ht_prev.T)
                dbg += dc_tilde.sum(axis=0)
                
                # Update previous deltas
                delta_h_prev = np.dot(self.Wf.T, dft) + np.dot(self.Wi.T, dit) + \
                            np.dot(self.Wo.T, dot) + np.dot(self.Wg.T, dc_tilde)
                delta_c_prev = delta_c * ft
        
            # Average gradients across batch
            n_samples = batch_size
            self.dUf = dUf / n_samples
            self.dWf = dWf / n_samples
            self.dbf = dbf / n_samples
            self.dUi = dUi / n_samples
            self.dWi = dWi / n_samples
            self.dbi = dbi / n_samples
            self.dUo = dUo / n_samples
            self.dWo = dWo / n_samples
            self.dbo = dbo / n_samples
            self.dUg = dUg / n_samples
            self.dWg = dWg / n_samples
            self.dbg = dbg / n_samples