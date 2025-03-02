import numpy as np

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        """
        Initializes a dense layer with specified input and output dimensions.
        :param n_inputs: Number of input features per data point.
        :param n_neurons: Number of neurons in the layer.
        """

        # Weight initialization: small random values
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # Bias initialized to zero
       

    def forward(self, inputs):
        """
        Forward pass: Computes weighted sum of inputs plus biases.
        :param inputs: Input data of shape (batch_size, n_inputs)
        """
       
        if inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(f"Input features {inputs.shape[1]} do not match expected size {self.weights.shape[0]}")

        self.inputs = inputs  # Store inputs for use in backpropagation
        self.output = np.dot(inputs, self.weights) + self.biases
        
        return self.output

    def backward(self, dvalues):
        """
        Backward pass: Computes gradients of weights, biases, and inputs.
        :param dvalues: Gradient of the loss w.r.t. the layer's output.
        """
        if dvalues.shape[1] != self.weights.shape[1]:
            raise ValueError(f"Output gradients {dvalues.shape[1]} do not match expected size {self.weights.shape[1]}")

        # Gradient of weights: d(Loss)/d(Weights)
        self.dweights = np.dot(self.inputs.T, dvalues)

        # Gradient of biases: d(Loss)/d(Biases)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient of inputs: d(Loss)/d(Inputs)
        self.dinputs = np.dot(dvalues, self.weights.T)
