import numpy
from Neuron import Neuron

class NeuralNetwork:
    def __init__(self, input_size, learning_rate=0.01):
        """
        input_size is a single int to tell the size of the input space given
        """
        self.input_shape = input_shape
        self.network = [[]]
        self.learning_rate = learning_rate
        for i in range(input_size):
            self.network[0].append(Neuron(input_size=1, learning_rate=self.learning_rate, activation='linear'))

    def Dense(self, units=1, activation='linear'):
        """
        Add a dense Layer of Neurons of size units to the end of the neural network
        """
        new_layer = []
        for i in range(units):
            new_layer.append(Neuron(input_size=len(self.network[-1])))