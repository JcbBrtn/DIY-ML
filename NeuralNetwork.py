import numpy as np
from Neuron import Neuron

class NeuralNetwork:
    def __init__(self, input_size, learning_rate=0.01):
        """
        input_size is a single int to tell the size of the input space given
        """
        self.input_size = input_size
        self.network = [[]]
        self.learning_rate = learning_rate
        self.last_size = input_size

    def Dense(self, units=1, learning_rate=0.01, activation='linear'):
        """
        Add a dense Layer of Neurons of size units to the end of the neural network
        """
        new_layer = []
        for i in range(units):
            new_layer.append(Neuron(input_size=(self.last_size,),learning_rate=learning_rate, activation=activation))
        self.network.append(new_layer)
        self.last_size = len(new_layer)

    def fit(self, X, Y, epochs=1):
        for epoch in range(epochs):
            for i, x in enumerate(X):
                pred = self.fire(x)
                total_error = 0
                for i,y in enumerate(Y):
                    total_error += (pred[i] - y)**2
                self.back_prop(total_error)

    def back_prop(self, error):
        for layer in range(1, 1 + len(self.network)):
            i = -1 * layer
            next_error = 0.0
            for n in range(len(self.network[i])):
                self.network[i][n].learn(error)
                next_error += self.network[i][n].last_error()
            error = next_error

    def fire(self, x):
        """
        Ad-hoc fire, given input x and current weights, get outputs.
        """
        for layer in self.network:
            x = x
            next_x = []
            for n in layer:
                x = x
                print(x)
                next_x.append(n.fire(x))
            x = next_x
        return x

def main():
    model = NeuralNetwork(4)
    model.Dense(4, activation='sig')
    model.Dense(4, activation='sig')
    model.Dense(4, activation='sig')
    model.Dense(4, activation='sig')

    X = np.array([
        [0,0,0,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,0,1,1],
        [0,1,0,0],
        [0,1,0,1],
        [0,1,1,0],
        [0,1,1,1],
        [1,0,0,0],
        [1,0,0,1],
        [1,0,1,0],
        [1,0,1,1],
        [1,1,0,0],
        [1,1,0,1],
        [1,1,1,0],
        [1,1,1,1]
    ])

    Y = np.array([
        [0,0,0,1],
        [0,0,1,0],
        [0,0,1,1],
        [0,1,0,0],
        [0,1,0,1],
        [0,1,1,0],
        [0,1,1,1],
        [1,0,0,0],
        [1,0,0,1],
        [1,0,1,0],
        [1,0,1,1],
        [1,1,0,0],
        [1,1,0,1],
        [1,1,1,0],
        [1,1,1,1],
        [0,0,0,0]
    ])

    model.fit(X, Y, epochs=100)

    for x in X:
        print(f'Input : {x}\nOutput : {model.fire(x)}')
if __name__ == '__main__':
    main()
