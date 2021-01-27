import numpy as np

class Neuron:
    def __init__(self, input_size=1, learning_rate=0.01):
        self.weights = np.random.random((input_size))
        self.bias = np.random.random()
        self.last_z = 0.0
        self.last_input = numpy.array([0,0,0])
        self.last_a = 0.0
        self.learning_rate=learning_rate
    
    def fire(self, inp):
        self.last_input = inp
        self.last_z = sum(self.weights * self.last_input) + self.bias
        #Put last Output through activation function 
        self.last_a = self.last_z
        return self.last_output

    def learn(self, error):
        for i, activation in enumerate(self.last_input):
            self.weights[i] += self.learning_rate * error * activation * self.f_prime(self.last_z)
            