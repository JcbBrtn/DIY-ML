from black import out
import numpy as np

class FunctionGenerator:
    """
    Basis of our Evolutionary Agent. This collects and evolves numpy functions by and array and executes these
    Functions in the order that they are placed within the array.
    """

    def __init__(self, dimensions=1):
        self.dimensions = [str(i) for i in range(dimensions)]
        self.func = [[] for i in range(dimensions)]
        self.varBank = [[] for i in range(dimensions)]
        self.toChooseFrom = [
            np.sin, np.cos, np.tan, np.arcsin, np.arccos,
            np.arctan, np.sinh, np.cosh, np.tanh, np.arcsinh,
            np.arccosh, np.arctanh, np.add, np.multiply,
            np.divide, np.power, np.subtract, np.sqrt, np.square
        ]
        #initilize with a random function to start
        self.create_func()

    def create_func(self):
        for i, _ in enumerate(self.func):
            top = 1.0
            while np.random.random() <= top:
                #add to the function
                self.func[i].append(np.random.choice(self.toChooseFrom))

                #always add to the var bank
                #50% chance to add one of the variables to the var bank
                #The rest of the time, add a random number between -10 and 10
                if np.random.random() < .5:
                    self.varBank[i].append(np.random.choice(self.dimensions))
                else:
                    self.varBank[i].append(np.random.uniform(-10, 10))
                
                #Divide the top by 2
                top /= 2
    
    def get_var(self, vars, v):
        if type(v) == str:
            return vars[int(v)]
        else:
            return v


    def run(self, vars):
        #for every input dimension in var, which should be as long as the func array
        #run through the func and get the new total
        output = []
        for f_set, v_set in zip(self.func, self.varBank):
            total = float(self.get_var(vars, v_set[0]))

            for  f, v in (zip(f_set, v_set)):
                if f in [np.add, np.multiply, np.divide, np.power, np.subtract]:
                    #These Functions require 2 variables, so pull from varBank
                    #and use that var as the second parameter
                    x = float(self.get_var(vars, v))
                    f(total, x)
                else:
                    total = f(total)

            output.append(total)
        return output

    def mutate(self):
        #set up the child to be the same as the parent
        child = FunctionGenerator(dimensions=len(self.dimensions))
        child.func = self.func
        child.varBank = self.varBank

        #make slight changes to the child

        return child

    def score(self, output):
        #This needs to be tweaked to include points outside of the set
        #take the magnitude of the output array that is given by the run function.
        total = 0
        for i in output:
            total += (i * i)
        
        return np.sqrt(i)