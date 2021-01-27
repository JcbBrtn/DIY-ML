import numpy as np

class Linear:
    def __init__(self, input_shape, learning_rate=0.015):
        self.learning_rate = learning_rate
        self.weights = np.random.random(input_shape)
        print(f'weight shape : {self.weights.shape}')
        self.last_output = 0.0

    def run(self, X):
        if X.shape == self.weights.shape:
            self.last_output = sum(X * self.weights)
        else:
            print('X is not same shape as weights!')

        return self.last_output

    def fit(self, X, Y, epochs=1):
        for epoch in range(epochs):
            print(f'Epoch {epoch} / {epochs}')
            for t, x in enumerate(X):
                print(f'\tt = {t} / {len(X)}', end='\r')
                self.run(x)
                self.weights += self.learning_rate * (Y[t] - self.last_output) * x
            print(self.weights)
        
    def predict(self, X):
        output = []
        for x in X:
            output.append(self.run(x))

        return output

def test():
    X = np.array([[1,3,2],
         [1,5,4],
         [2,6,1],
         [8,3,0],
         [5,5,5]])
        
    y = np.array([3,5,6,3,5])

    model = Linear(len(X[0]))
    model.fit(X, y, epochs=50)

    print(model.run(np.array([[4,7,1]])))

if __name__ == '__main__':
    test()