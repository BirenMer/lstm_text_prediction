import numpy as np
class Sigmoid:
    def forward(self,M):
        sigm=np.clip(1/(1+np.exp(-M)),1e-7,1-1e-7)
        self.output=sigm
        self.inputs=sigm #needed for back prop
        return self.output
    def backward(self,dvalues):
        sigm=self.inputs
        deriv=np.multiply(sigm,(1-sigm))
        self.dinputs=np.multiply(deriv,dvalues)
