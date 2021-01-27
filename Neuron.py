import numpy as np

class Neuron:
    def __init__(self, input_size=1, learning_rate=0.01, activation='sig'):
        self.weights = np.random.random(input_size)
        self.activation=activation.lower()
        self.bias = np.random.random()
        self.last_z = 0.0
        self.last_input = np.array([0,0,0])
        self.last_a = 0.0
        self.learning_rate=learning_rate

    def attach(self, amount=1):
        np.append(self.weights, np.random.random(amount))
    
    def f_prime(self, z):
        if self.activation == 'sig':
            a = self.f(z) * (1 - self.f(z))
        elif self.activation == 'tanh':
            a = 1 - (self.f(z))**2
        elif self.activation == 'bin':
            a = 0
        elif self.activation == 'relu':
            a = 1
        else:
            a = z
        return a

    def f(self, z):
        if self.activation == 'sig':
            a = 1 / (1 + np.exp(-1 * z))
        elif self.activation == 'tanh':
            a = np.tanh(z)
        elif self.activation == 'bin':
            a = int(z >= 0.5) * 1.0
        elif self.activation == 'relu':
            a = max(0, z)
        else:
            a = z
        return a

    def fire(self, inp):
        """
        Make sure inp is a numpy array the same size (size, ) as the one given
        """
        self.last_input = inp
        self.last_z = sum(self.weights * self.last_input) + self.bias
        #Put last Output through activation function 
        self.last_a = self.f(self.last_z)
        return self.last_z

    def learn(self, error):
        for i, activation in enumerate(self.last_input):
            self.weights[i] += self.learning_rate * error * self.last_input[i] * self.f_prime(self.last_z)
        self.bias += self.learning_rate * error