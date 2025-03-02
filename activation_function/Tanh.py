import  numpy as np
class Tanh:
    # defining tanh activiation function
    def forward(self,inputs):
        self.output=np.tanh(inputs)
        self.inputs=inputs
        return self.output
        
    def backward(self,dvalues):
        deriv=1-self.output**2
        self.dinputs=np.multiply(deriv,dvalues)
        return self.dinputs 