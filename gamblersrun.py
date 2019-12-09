import sys
sys.argv = ["", "gambler_2_2_0.4"]
from dippl_yacc import *

start = time.time()
total_weight = wmc(w, phi, bdd, init=True)
reached_weight = wmc(w, bdd.apply('and', phi, bdd.var('reached')), bdd, init=True)
end = time.time()
print("Query time:",end-start)

print("Probability of reaching x10:", reached_weight/total_weight)
