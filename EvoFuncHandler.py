from ast import excepthandler
from FunctionGenerator import FunctionGenerator
from sklearn.metrics import log_loss
import random
import math

#An agent that attempts to fit the data using an envolutionary algorithm and the Function Generator

class EvoBinaryHandler:
    def __init__(self,shape,pop=100,iterations=20,escape_dist=10,threshold=0.5):
        self.pop = [FunctionGenerator(shape) for _ in range(pop)]
        print(random.choice(self.pop).funcs)
        self.pop_len= pop
        self.iterations = iterations
        self.escape_dist = escape_dist
        self.shape = shape
        self.threshold = threshold

    def run_single_agent(self, popIndex, inp):
        for _ in range(self.iterations):
            for i in range(len(inp)):
                inp[i] = self.pop[popIndex].run(inp[i])
        
        y_pred = []
        for x in inp:
            y_pred.append(self.get_bin_from_dist(x))

        return y_pred

    def repopulate(self):
        while len(self.pop) <= self.pop_len:
            if random.random() <= 0.5 and len(self.pop) > 1:
                p = random.choice(self.pop)
                child = p.get_child()
                self.pop.append(child)
            else:
                self.pop.append(FunctionGenerator(self.shape))
        
    def get_bin_from_dist(self, inp):
        total = 0
        try:
            for i in inp:
                total += i**2
            return int(math.sqrt(total) <= self.escape_dist)
        except:
            print(f'Error processing {inp}')
            return 1000

    def to_string(self):
        for agent in self.pop:
            print(agent.funcs)

    def score(self, y_true, y_pred):
        return log_loss(y_true, y_pred)
    
    def fit(self, X, y, iterations=100):
        for iter in range(iterations):
            #Get a sorted Dictionary of all agents in the pop and their scores
            agent_score = {}

            for i in range(len(self.pop)):
                y_pred = self.run_single_agent(i, X)
                score = self.score(y, y_pred)
                agent_score[self.pop[i]] = score
            
            #sort the agent_score dict
            dictionary_keys = list(agent_score.keys())
            sorted_dict = {dictionary_keys[i]: sorted(
                agent_score.values())[i] for i in range(len(dictionary_keys))}
            
            ordered_agents = list(sorted_dict.keys())
            best_agent_score = min(agent_score.values())
            
            #Take the better half of the dictionary and repopulate based on it
            half_way_index = len(ordered_agents)//2
            
            self.pop = ordered_agents[half_way_index:]
            
            print(f'Iteration {iter} / {iterations} | best agent score {best_agent_score}')

            self.repopulate()
            