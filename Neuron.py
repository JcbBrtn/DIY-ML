import numpy as np
from Linear import Linear

class Neuron:
    def __init__(self, is_input=False, input_arr=[], learning_rate=0.01, activation_type='sig'):
        """
        The Neuron Class:

        The main unit of a neural network. Each neuron is created induvidually and is refrenced
        through eachother to form a proper neural network.

        neuron_type ->  The 'type' of neuron. Only options are normal or input.
                        Input neurons have no inputs other than the specific data
                        refrence they are meant to hold
        input_arr -> The array of direct neuron refrence.
                     The forward attached neruons are called to attach current neuron to the back.
        """
        self.is_input = is_input
        self.back_connections = input_arr
        self.front_connections = []
        self.curr_activation = 0.0 
        self.fired = False
        self.total_error = 0.0
        self.m = [0] * len(self.back_connections)
        self.v = [0] * len(self.back_connections)

        if not self.is_input:
            for n in self.back_connections:
                n.front_attach(self)
            size = len(input_arr)
            self.weights = np.random.random(size)
            self.activation_type=activation_type.lower()
            self.bias = np.random.random()
            self.curr_z = 0.0
            self.last_input = []
            self.learning_rate=learning_rate
            self.smart_agent=Linear(input_shape=len(self.back_connections))
        else:
            self.fired = True
           
    def toString(self):
        if not self.is_input:
            return f'Neuron With : \nWeights : {self.weights}\nAct : {self.activation_type}\nLast Input : {self.last_input}\nZ value {self.curr_z}\nActivation {self.curr_activation}'
        else:
            return f'Neuron of Type : Input\nCurrent Activation : {self.curr_activation}'

    def reset_fired(self):
        self.fired = False

    def reset_error(self):
        self.total_error = 0.0

    def update_error(self, error):
        self.total_error += error

    def front_attach(self, n):
        self.front_connections.append(n)

    def reset_m_and_v(self):
        self.m = [0] * len(self.back_connections)
        self.v = [0] * len(self.back_connections)

    def back_attach(self, n):
        self.back_connections.append(n)
        self.reset_m_and_v()
        n.front_attach(self)
        np.append(self.weights, np.random.random())
        self.smart_agent=Linear(input_shape=len(self.back_connections))

    def activation_der(self, z):
        a = z
        if self.activation_type == 'sig':
            a = self.activation(z) * (1 - self.activation(z))
        elif self.activation_type == 'tanh':
            a = 1 - (self.activation(z))**2
        elif self.activation_type == 'bin':
            a = 0
        elif self.activation_type == 'relu':
            if z > 0:
                a = 1
            else:
                a = 0
        else:
            a = z
        return a

    def set_activation(self, a):
        self.curr_activation = a

    def activation(self, z):
        if self.activation_type == 'sig':
            a = 1 / (1 + np.exp(-1 * z))
        elif self.activation_type == 'tanh':
            a = np.tanh(z)
        elif self.activation_type == 'bin':
            a = int(z >= 0.5) * 1.0
        elif self.activation_type == 'relu':
            a = max(0, z)
        else:
            a = z
        return a

    def activate(self):
        if self.is_input or self.fired:
            return self.curr_activation
        else:
            self.fired = True
            inputs = []
            for n in self.back_connections:
                n.toString()
                inputs.append(n.activate())

            self.last_input = np.array(inputs)
            #print(f'last_input : {self.last_input}\nWeights : {self.weights}')
            self.curr_z = sum(self.last_input * self.weights) + self.bias

            self.curr_activation = self.activation(self.curr_z)

            return self.curr_activation

    def backprop(self, optimizer):
        n = 1
        if len(self.front_connections) > 0:
            n = len(self.front_connections)
        
        error = self.total_error / n
        if optimizer=='sgd':
            #Update the Weights
            for i in range(len(self.weights)):
                weight_error = self.last_input[i] * self.activation_der(self.curr_z) * error

                self.weights[i] -= self.learning_rate * self.last_input[i] * self.activation_der(self.curr_z) * error
                self.back_connections[i].update_error(weight_error)

            #Update the Bias
            self.bias -= self.activation_der(self.curr_z) * error * self.learning_rate
        
        elif optimizer=='adam':
            beta_1 = 0.9
            beta_2 = 0.999
            epsilon = 1.0e-8
            for i in range(len(self.weights)):
                weight_error = self.last_input[i] * self.activation_der(self.curr_z) * error
                self.m[i] = beta_1 * self.m[i] + ((1-beta_1) * weight_error)
                self.v[i] = beta_2 * self.v[i] + ((1-beta_2) * weight_error**2)
                m_hat = self.m[i] / (1 - beta_1)
                v_hat = self.v[i] / (1 - beta_2)
                self.weights[i] -= (self.learning_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
                self.back_connections[i].update_error(weight_error)
            
            #Update the Bias
            self.bias -= self.activation_der(self.curr_z) * error * self.learning_rate


def main():
    model = []
    model.append(Neuron(True))
    model.append(Neuron(False, [model[0]], learning_rate=0.5))
    model.append(Neuron(False, [model[1]], learning_rate=0.5))

    for i in range(10000):
        for n in model:
            n.reset_error()
            n.reset_fired()

        model[0].set_activation(np.array([1]))
        pred = model[-1].activate()
        print(f'Model\'s prediction : {pred}')
        model[-1].update_error(pred)
        for j in range(1, len(model)):
            model[-1 * j].backprop('sgd')

if __name__ == '__main__':
    main()