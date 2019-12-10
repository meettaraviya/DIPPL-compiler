from compile import *
from generators.gamblersrun_gen import *

for nsteps, walklength, pforward in [(10, 5, 0.4), (10, 10, 0.4), (10, 20, 0.4), (10, 40, 0.4)]:

	generate(nsteps, walklength, pforward)
	print("\nnsteps={}, walklength={}, pforward={}:".format(nsteps, walklength, pforward))
	compile(program='gambler_{}_{}_{}'.format(nsteps, walklength, pforward), queries=['reached'], algo='approximate')

# start = time.time()
# total_weight = wmc(w, phi, bdd, init=True)
# reached_weight = wmc(w, bdd.apply('and', phi, bdd.var('reached')), bdd, init=True)
# end = time.time()
# print("Query time:",end-start)

# print("Probability of reaching x10:", reached_weight/total_weight)
