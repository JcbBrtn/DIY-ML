import numpy as np
from Neuron import Neuron

class NeuralNetwork:
    def __init__(self, input_size):
        """
        input_size is a single int to tell the size of the input space given
        """
        self.input_size = input_size
        self.network = [[]]
        for i in range(input_size):
            self.network[0].append(Neuron(True))

    def toString(self):
        output = f''
        for layer in range(len(self.network)):
            output += f'#####################\nLayer {layer}\n####################\n'
            for n in self.network[layer]:
                output += n.toString()
                output+='\n'
        return output

    def Dense(self, units, learning_rate=0.25, activation='relu'):
        """
        Adds a Dense layer of units neurons to the end of the neural network.
        """
        new_layer = []
        for i in range(units):
            new_layer.append(Neuron(False, input_arr=self.network[-1], learning_rate=learning_rate,activation_type=activation))
        self.network.append(new_layer)

    def connect(self, l1, l2):
        """
        This function connects 2 layers such that l1 is outputing to l2.
        l1 and l2 are both the integer describing which number layer in self.network
        being described.
        """
        for inp_i in range(len(self.network[l1])):
            for i in range(len(self.network[l2])):
                self.network[l2][i].front_attach(self.network[l1][inp_i])

    def new_prop(self):
        for l in self.network:
            for i in range(len(l)):
                l[i].fired = False

    def feed_forward(self, x):
        #Ad-hoc forward propagation given input array x
        self.new_prop()
        #Set the input layer activations
        for j, feat in enumerate(x):
            #print(f'Setting Activation to {feat}')
            self.network[0][j].set_activation(feat)
        
        #Fire by calling each end neuron
        pred = []
        for out in self.network[-1]:
            #print(f'Len {len(self.network)} | Neuron {out.toString()}')
            pred.append(out.activate())

        return pred

    def reset_error(self):
        for l in self.network:
            for n in l:
                n.reset_error()

    def update_lr(self, new_lr):
        for l in self.network:
            for n in l:
                n.learning_rate = new_lr

    def fit(self, X, Y, epochs=1, batch_size=1, optimizer="adam"):
        for epoch in range(epochs):
            total_cost = 0.0
            self.reset_error()
            for i in range(len(X)):
                #print(self.toString())
                x = X[i]
                pred = self.feed_forward(x)
                #Update each end neurons error with each prediction made

                for j,p in enumerate(pred):
                    total_cost += (p - Y[i][j])**2
                    self.network[-1][j].update_error((p - Y[i][j]))

                if i%batch_size == 0:
                    #If It is the end of a batch
                    #Call backprop going backwards through the Network
                    for layer in range(1, len(self.network)):
                        for n in self.network[-1 * layer]:
                            n.backprop(optimizer)
                    self.reset_error()
                

            print(f'Epoch {epoch} / {epochs} | Avg Network Cost : {total_cost / len(X)}', end='\r')

def main():
    model = NeuralNetwork(4)
    model.Dense(8, learning_rate=0.02, activation='tanh')
    model.Dense(4, learning_rate=0.02, activation='tanh')
    for layer1 in range(len(model.network)):
        for layer2 in range(1, len(model.network)):
            model.connect(layer1, layer2)

    X = np.array([
        [0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],
        [0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],
        [1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],
        [1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]
    ])

    Y = np.array([
        [0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],
        [0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],
        [1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],
        [1,1,0,1],[1,1,1,0],[1,1,1,1],[0,0,0,0]
    ])

    model.fit(X, Y, epochs=500, batch_size=1, optimizer="adam")

    print(model.toString())

    for y,x in zip(Y,X):
        print(f'Expected Output : {y}\nOutput : {model.feed_forward(x)}')


if __name__ == '__main__':
    main()
