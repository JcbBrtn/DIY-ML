import numpy as np
from Neuron import Neuron

class NeuralNetwork:
    def __init__(self, input_size, learning_rate=0.05):
        """
        input_size is a single int to tell the size of the input space given
        """
        self.input_size = input_size
        self.network = []
        self.learning_rate = learning_rate
        self.last_size = input_size
        self.total_loss = 0.0

    def Dense(self, units=1, learning_rate=0.01, activation='linear'):
        """
        Add a dense Layer of Neurons of size units to the end of the neural network
        """
        new_layer = []
        for i in range(units):
            new_layer.append(Neuron(input_size=(self.last_size,),learning_rate=learning_rate, activation=activation))
        self.network.append(new_layer)
        self.last_size = len(new_layer)

    def back_prop(self, error):
        for layer in range(1, 1 + len(self.network)):
            i = -1 * layer
            next_error = []
            for n in range(len(self.network[i])):
                self.network[i][n].learn(error[n])
                next_error.append(self.network[i][n].get_error())
            error = next_error

    def forward_prop(self, inp):
        """
        Ad-hoc fire, given input x and current weights, get outputs.
        """
        for layer in self.network:
            next_act = []
            for neu in layer:
                act = neu.fire(inp)
                next_act.append(act)
            inp = next_act
        return inp

    def fit(self, X, Y, epochs=1):
        for epoch in range(epochs):
            print(f'Epoch {epoch} / {epochs}', end=' ')
            for i, x in enumerate(X):
                pred = self.forward_prop(x)
                #print(f'predictions = {pred}')
                total_error = 0
                error = []
                for j,y in enumerate(Y[i]):
                    error.append((pred[j] - y))
                self.total_loss = sum(error)
                self.back_prop(error)
            print(f'Total Loss : {self.total_loss}')

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

    model.fit(X, Y, epochs=300)

    for x in X:
        print(f'Input : {x} -> Output : {model.forward_prop(x)}')
if __name__ == '__main__':
    main()
