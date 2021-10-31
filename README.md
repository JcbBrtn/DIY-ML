# Do it Yourself - Machine Learning!

Included in this Repository are a couple machine learning models I created from the Python Library Numpy. Included in here are:

* Attention Heads: Other wise known as a Transformer. This is the basis for OpenAI's GPT Natural Language Processing model.
* K-Nearest Neighbors: This is a classifier that uses all of the points it knows the lables of to decided what the current point should be labled as. It does this by calculateing the distance between all of the N neighbors and decides which classification is closest.
* Linear Model: A simple regression model that calculates the linear relationship between features X and points of interest y by using calculus and a loss function to set w such that Xw = y
* Neuron: Otherwise known as a perceptron, this is the base layer of the Neural network class. Similar to the Linear model, except instead of a matrix of weights, its only one weight with one bias such that sum(X)w + b = y.
* Neural Network: An artifical neural netowrk that behaves much like that of Tensorflow. Currently, the only optimizers that can be used on this is Adam, and SGD.
*Recursive model: This is an experimental proof of work where I want to see if there is a f and x0 such that f(x0) ran in a fractal fashion where the output becomes the new input can produce any ordered list of numbers. so, say the wieghts of a neural netowrk could be compressed down to a single function and starting point.
