
import random

class FunctionGenerator:
    def __init__(self, n, funcs=False):
        self.operations = ['+', '-', '*', '**', '/', '%', '//']
        self.constants = ['args[N]','args[N]','args[N]', 'random.random()', 'random.randint(-5, 5)']

        if not funcs:
            self.funcs = self.GetNewFunc(n)
        else:
            self.funcs = funcs
        
        self.n = n

    def get_new_const(self, n):
        con = random.choice(self.constants)
        if 'args' in con:
            return con.replace('N', str(random.randint(0, n-1)))
        else:
            return str(eval(con))

    def GetNewFunc(self, n):
        funcs= []
        #Create string of constant operator constant at random
        for i in range(n):
            func = ''
            #Create a new function
            while random.random() < .5:
                func += self.get_new_const(n)
                func += ' '
                func += random.choice(self.operations)
                func += ' '

            #To end the generation, we add in a constant to be sure we close off any operations
            func += self.get_new_const(n)

            funcs.append(func)
            
        return funcs

    def run(self, args):
        if len(args) != self.n:
            raise Exception('Length of Args does not match the given length at initialization.')
            
        out = []
        try:
            for f in self.funcs:
                out.append(eval(f))
        except:
            out = [99999999 for i in range(self.n)]
        
        return out
    
    def mutate(self):
        #mutate slightly on all functions
        for i in range(self.n):
            func = self.funcs[i]
            f = func.split(' ')
            for p in range(len(f)):
                if random.random() <= .3:
                    #Now we can mutate
                    if f[p] in self.operations:
                        f[p] = random.choice(self.operations)
                    else:
                        f[p] = self.get_new_const(self.n)
            self.funcs[i] = ' '.join(f)
    
    def get_child(self):
        child = FunctionGenerator(self.n, self.funcs)
        child.mutate()
        return child

