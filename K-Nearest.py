import numpy as np

class KNN:
    def __init__(self):
        self.Memory = {} #Keys are the variable Y's or sets that the points belong.
        #The Values are 2d numpy arrays of the seen vectors.
        self.Errors = {}
        self.y_mapping={}
    def fit(self, X, Y):
        for x,y in zip(X,Y):
            self.y_mapping.setdefault(str(y), y)
            self.Memory.setdefault(str(y), [])
            self.Memory[str(y)].append(np.asarray(x))
            self.Errors.setdefault(str(y), 0.1)

    def predict(self, X):
        #for numpy array of X
        submission = []
        for x in X:
            min_error = 9999999999999999
            set_to_return = 0
            for y in self.Memory.keys():
                error = 1.0
                for x_hat in self.Memory[y]:
                    #print(f'x = {x}\nx_hat = {x_hat}')
                    error *= sum((x - x_hat)**2)
                #print(f'curr_error : {error}\nmin_error : {min_error}')
                self.Errors[y] = error
                if error < min_error:
                    min_error = error
                    set_to_return = y
            submission.append(self.y_mapping[set_to_return])
        return submission

def main():
    model = KNN()

    X = np.array([
        [1,1,1,1],[1,1,1,2],[1,1,2,1],[1,1,2,2],
        [1,2,1,1],[1,2,1,2],[1,2,2,1],[1,2,2,2],
        [2,1,1,1],[2,1,1,2],[2,1,2,1],[2,1,2,2],
        [2,2,1,1],[2,2,1,2],[2,2,2,1],[2,2,2,2]
    ])

    Y = np.array([
        [1,1,1,2],[1,1,2,1],[1,1,2,2],[1,2,1,1],
        [1,2,1,2],[1,2,2,1],[1,2,2,2],[2,1,1,1],
        [2,1,1,2],[2,1,2,1],[2,1,2,2],[2,2,1,1],
        [2,2,1,2],[2,2,2,1],[2,2,2,2],[1,1,1,1]
    ])

    model.fit(X, Y)

    for y,x in zip(Y,X):
        print(f'Expected Output : {y}\nOutput : {model.predict([x])}')


if __name__ == '__main__':
    main()