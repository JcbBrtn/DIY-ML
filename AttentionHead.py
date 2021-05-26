import numpy as np
from NeuralNetwork import NeuralNetwork

class AttentionHead:
    def __init__(self, input_shape, learning_rate = 0.015):
        self.q_model = NeuralNetwork(input_shape)
        self.q_model.Dense(input_shape, learning_rate)
        self.k_model = NeuralNetwork(input_shape)
        self.k_model.Dense(input_shape, learning_rate)
        self.v_model = NeuralNetwork(input_shape)
        self.v_model.Dense(input_shape, learning_rate)
        self.input_shape = input_shape

    def dot_product(self, q, k):
        """
        Dot product to Create the scale matrix. 
        Q and K are np arrays of the two different heads.
        """
        return np.dot(q, k)

    def scale(self, scale_matrix):
        return scale_matrix / np.sqrt(self.input_shape)

    def softmax(self, scale_matrix):
        e = np.exp(scale_matrix)
        return e / e.sum()

    def run(self, q, k, v):
        q = self.q_model.feed_forward(q)
        k = self.k_model.feed_forward(k)
        scale_matrix = self.dot_product(q, k)
        scale_matrix = self.scale(scale_matrix)
        scale_matrix = self.softmax(scale_matrix)
        v = self.v_model.feed_forward(v)
        return self.dot_product(scale_matrix, v)

    def fit(self, X, epochs=1, batch_size=1, optimizer="adam"):
        print('Training the q Encoder...')
        self.q_model.fit(X, X, epochs, batch_size, optimizer)
        print('Training the k Encoder...')
        self.k_model.fit(X, X, epochs, batch_size, optimizer)
        print('Training the v Encoder...')
        self.v_model.fit(X, X, epochs, batch_size, optimizer)