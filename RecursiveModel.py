import numpy as np
import random

class FunctionGenerator:
    """
    Basis of our Evolutionary Agent. This collects and evolves numpy functions by and array and executes these
    Functions in the order that they are placed within the array.
    """

    def __init__(self):
        self.func = []
        self.varBank = []
        self.toChooseFrom = [
            np.sin, np.cos, np.tan, np.arcsin, np.arccos,
            np.arctan, np.sinh, np.cosh, np.tanh, np.arcsinh,
            np.archcosh, np.arctanh, np.add, np.multiply,
            np.divide, np.power, np.subtract, np.sqrt, np.square
        ]
        #initilize with a random function to start
