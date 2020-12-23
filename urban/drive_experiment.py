import NoveltySearch

####################################################################################
bounds = [[0, 67], [0, 3]]                      #[[speed range], [actions]]
mutationProb = 0.4                             # mutation rate
crossoverProb = 0.4                             # crossover rate
popSize = 4
numOfNpc = 1
numOfTimeSlice = 5
maxGen = 100
####################################################################################

ns = NoveltySearch.NoveltySearch(bounds, mutationProb, crossoverProb, popSize, numOfNpc, numOfTimeSlice, maxGen)
ns.set_checkpoint('GaCheckpoints/last_gen.obj')
ns()
