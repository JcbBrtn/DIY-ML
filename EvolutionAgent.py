

class EvoAgent:
    #An evolutionary algorithm. The population class must have 3 functions to be callable:
    #score() where the judgement is made against the population member.
    #mutate() where a new child member is returned with sligth alterations from the parent.
    #run() where the agent is set off to do whatever it needs to do, a scoreable output is returned.
    def __init__(self, population):
        self.population = population

    