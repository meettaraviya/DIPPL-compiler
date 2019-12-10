from compile import *
from generators.gamblersrun_gen import *

for n, p in [(10, 0.5), (20, 0.5), (30, 0.5), (40, 0.5)]:

	generate(n, p)
	print("\nnsteps={}, walklength={}, pforward={}:".format(n, p))
	compile(program='binomial_{}_{}'.format(n, p), queries=['reached'], algo='approximate')

# start = time.time()
# total_weight = wmc(w, phi, bdd, init=True)
# reached_weight = wmc(w, bdd.apply('and', phi, bdd.var('reached')), bdd, init=True)
# end = time.time()
# print("Query time:",end-start)

# print("Probability of reaching x10:", reached_weight/total_weight)
