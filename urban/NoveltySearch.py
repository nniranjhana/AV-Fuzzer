import numpy as np
np.random.seed(0)
import env, os, sys, pickle, random, yaml, copy
from Chromosome import Chromosome
from datetime import datetime
import utils, tools
import generateRestart

class NoveltySearch:
    def __init__(self, bounds, pm, pc, pop_size, NPC_size, time_size, max_gen):

        self.bounds = bounds                # The value ranges of the inner most elements
        self.pm = pm
        self.pc = pc
        self.pop_size = pop_size            # Number of scenarios in the population
        self.NPC_size = NPC_size            # Number of NPC in each scenario
        self.time_size = time_size          # Number of time slides in each NPC
        self.max_gen = max_gen
        self.pop = []
        self.bests = [0] * max_gen
        self.bestIndex = 0
        self.g_best = None
        self.ck_path = None                 # Checkpoint path, if set, GE will start from the checkpoint (population object)
        self.touched_chs = []               # Record which chromosomes have been touched in each generation

        self.isInLis = False                # Set flag for local iterative search (LIS)
        self.minLisGen = 2                  # Min gen to start LIS
        self.numOfGenInLis = 5              # Number of gens in LIS
        self.hasRestarted = False
        self.lastRestartGen = 0
        self.bestYAfterRestart = 0

    def set_checkpoint(self, ck_path):
        self.ck_path = ck_path

    def take_checkpoint(self, obj, ck_name):
        if os.path.exists('GaCheckpoints') == False:
            os.mkdir('GaCheckpoints')
        ck_f = open('GaCheckpoints/' + ck_name, 'wb')
        pickle.dump(obj, ck_f)
        ck_f.truncate() 
        ck_f.close()
   
    def setLisFlag(self):
        self.isInLis = True

    def setLisPop(self, singleChs):
        for i in range(self.pop_size):
            self.pop.append(copy.deepcopy(singleChs))

        # Add some entropy
        tempPm = self.pm
        self.pm = 1
        self.mutation(0)
        self.pm = tempPm
        self.g_best, bestIndex = self.find_best()
    
    def get_action(self, obs):
        return obs.copy()
    
    def evaluate(self):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = get_action(ns, obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        return total_reward
    
    def behaviour(self, candidate):
        obs = env.reset()
        done = False
        while not done:
            action = get_action(ns, obs)
            obs, reward, done, _ = env.step(action)
        return obs
    
    def similarity(self, candidate1, candidate2):
        b1, b2 = behaviour(candidate1), behaviour(candidate2)
        return np.sum((b1 - b2)**2)
    
    def compute_novelty(self, pop, k=3):
        distances = []
        n = len(pop)
        for i in range(n):
            distance_i = sorted([similarity(pop[i], pop[j]) for j in range(n) if i != j])[:k]
            distances.append(np.mean(distance_i))
        return distances
        
    def get_novel_subpop(self, pop, novelty_scores):
        return pop[novelty_scores.argmax()]
    
    def select_most_novel(self, pop, novelty_scores, k=0.5):
        return pop[np.argsort(novelty_scores)[-int(len(pop) * k):]]
    
    def perform_reproduction(self, subpop):
        num_children = pop_size - len(subpop)
        parents = np.random.choice(subpop, num_children)
        return np.append(subpop, [copy.deepcopy(p) for p in parents], axis=0)

    def init_pop(self):
        for i in range(self.pop_size):
            # A chromosome is a scenario
            chromosome = Chromosome(self.bounds, self.NPC_size, self.time_size)
            chromosome.rand_init()
            chromosome.func()
            self.pop.append(chromosome)

    def cross(self):
        # Implementation of random crossover

        for i in range(int(self.pop_size / 2.0)):
            # Check crossover probability
            if self.pc > random.random():
            # randomly select 2 chromosomes(scenarios) in pops
                i = 0
                j = 0
                while i == j:
                    i = random.randint(0, self.pop_size-1)
                    j = random.randint(0, self.pop_size-1)
                pop_i = self.pop[i]
                pop_j = self.pop[j]

                # Record which chromosomes have been touched
                self.touched_chs.append(self.pop[i])
                self.touched_chs.append(self.pop[j])

                # Every time we only switch one NPC between scenarios
                # select cross index
                swap_index = random.randint(0, pop_i.code_x1_length - 1)

                temp = copy.deepcopy(pop_j.scenario[swap_index])
                pop_j.scenario[swap_index] = copy.deepcopy(pop_i.scenario[swap_index])
                pop_i.scenario[swap_index] = temp

    def mutation(self, gen):
        i = 0
        while(i<len(self.pop)) :
            eachChs = self.pop[i]
            i += 1
            if self.pm >= random.random():

                beforeMutation = copy.deepcopy(eachChs)
                # select mutation index
                npc_index = random.randint(0, eachChs.code_x1_length-1)
                time_index = random.randint(0, eachChs.code_x2_length-1)

                # Record which chromosomes have been touched
                self.touched_chs.append(eachChs)
                actionIndex = random.randint(0, 1)
                
                if actionIndex == 0:
                    # Change Speed
                    eachChs.scenario[npc_index][time_index][0] = random.uniform(self.bounds[0][0], self.bounds[0][1])
                elif actionIndex == 1:
                    # Change direction
                    eachChs.scenario[npc_index][time_index][1] = random.randrange(self.bounds[1][0], self.bounds[1][1])

            # Only run simulation for the chromosomes that are touched in this generation
            if eachChs in self.touched_chs:
                eachChs.func(gen, self.isInLis)
            else:
                util.print_debug(" --- The chromosome has not been touched in this generation, skip simulation. ---")


            util.print_debug(" --- In mutation: Current scenario has y = " + str(eachChs.y))

    def select_top2(self):

        util.print_debug(" +++ Before select() +++ ")
        for i in range(0, self.pop_size):
            util.print_debug(" === Most novel of the scenario is " + str(self.pop[i].y) + " === ")
    
        maxNovel = 0
        v = []
        for i in range(0, self.pop_size):
            if self.pop[i].y > maxNovel:
                maxNovel = self.pop[i].y

        for i in range(0, self.pop_size):
            if self.pop[i].y == maxNovel:
                for j in range(int(self.pop_size / 2.0)):
                    selectedChromosome = Chromosome(self.bounds, self.NPC_size, self.time_size)
                    selectedChromosome.scenario = self.pop[i].scenario
                    selectedChromosome.y = self.pop[i].y
                    v.append(selectedChromosome)
                break

        max2Novel = 0
        for i in range(0, self.pop_size):
            if self.pop[i].y > max2Novel and self.pop[i].y != maxNovel:
                max2Novel = self.pop[i].y

        for i in range(0, self.pop_size):
            if self.pop[i].y == max2Novel:
                for j in range(int(self.pop_size / 2.0)):
                    selectedChromosome = Chromosome(self.bounds, self.NPC_size, self.time_size)
                    selectedChromosome.scenario = self.pop[i].scenario
                    selectedChromosome.y = self.pop[i].y
                    v.append(selectedChromosome)
                break

        self.pop = copy.deepcopy(v)
        util.print_debug(" +++ After select() +++ ")
        for i in range(0, self.pop_size):
            util.print_debug(" === Novel result of the scenario is " + str(self.pop[i].y) + " === ")


    def select_roulette(self):

        sum_f = 0

        util.print_debug(" +++ Before select() +++ ")
        for i in range(0, self.pop_size):
            if self.pop[i].y == 0:
                self.pop[i].y = 0.001
            util.print_debug(" === Novel result of the scenario is " + str(self.pop[i].y) + " === ")

        ############################################################
        min = self.pop[0].y
        for k in range(0, self.pop_size):
            if self.pop[k].y < min:
                min = self.pop[k].y
        if min < 0:
            for l in range(0, self.pop_size):
                self.pop[l].y = self.pop[l].y + (-1) * min

        # roulette
        for i in range(0, self.pop_size):
            sum_f += self.pop[i].y
        p = [0] * self.pop_size
        for i in range(0, self.pop_size):
            if sum_f == 0:
                sum_f = 1
            p[i] = self.pop[i].y / sum_f
        q = [0] * self.pop_size
        q[0] = 0
        for i in range(0, self.pop_size):
            s = 0
            for j in range(0, i+1):
                s += p[j]
            q[i] = s

        # start roulette
        v = []
        for i in range(0, self.pop_size):
            r = random.random()
            if r < q[0]:
                selectedChromosome = Chromosome(self.bounds, self.NPC_size, self.time_size)
                selectedChromosome.scenario = self.pop[0].scenario
                selectedChromosome.y = self.pop[0].y
                v.append(selectedChromosome)
            for j in range(1, self.pop_size):
                if q[j - 1] < r <= q[j]:
                    selectedChromosome = Chromosome(self.bounds, self.NPC_size, self.time_size)
                    selectedChromosome.scenario = self.pop[j].scenario
                    selectedChromosome.y = self.pop[j].y
                    v.append(selectedChromosome)
        self.pop = copy.deepcopy(v)
        ############################################################

        util.print_debug(" +++ After select() +++ ")
        for i in range(0, self.pop_size):
            util.print_debug(" === Novel result of the scenario is " + str(self.pop[i].y) + " === ")


# NS parameters
pop_size = 20
num_generations = 30
top_k = 0.2
mutation_sigma = 0.1
k_nearest = 3

if __name__ == '__main__':
    bounds = [[0, 70], [0, 3]]
    algorithm = NoveltySearch(bounds,0.4, 0.8, 4, 4, 5, 30)
    for i in range(num_generations):
        novelty_scores = compute_novelty(pop, k=k_nearest)
        novel_subpop = select_most_novel(pop, novelty_scores, k=top_k)
        algorithm()
    pass