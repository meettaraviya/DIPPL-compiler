def get_wmc_bruteforce(w, phi, bdd):
	
	retval = 0.0

	for model in bdd.pick_iter(phi):

		model_wt = 1.0
		# print(*[str(k)+"="+str(int(v))+"," for k,v in model.items()])

		for var, val in model.items():
			model_wt *= w["~"*(1-val) + var]

		retval += model_wt

	return retval


def print_marginal_distribution(wbdd, knowns, bdd):

	phi, w = wbdd
	new_phi = bdd.apply('and', phi, bdd.cube(knowns))

	unknowns = (all_vars | all_flips) - knowns.keys()

	knowns_wt_prod = 1.0

	for var, val in knowns.items():
		if val:
			knowns_wt_prod *= w[var]
		else:
			knowns_wt_prod *= w['~'+var]

	unknowns_wt_prod_sum = 0.0

	i = 1

	for model in bdd.pick_iter(new_phi, care_vars=unknowns):

		if bdd.let(model, new_phi) == bdd.true:
			unknowns_wt_prod = 1.0
			for var in unknowns:
				if model[var]:
					unknowns_wt_prod *= w[var]
				else:
					unknowns_wt_prod *= w['~'+var]
			# print(model, ":", "{:.4f}".format(unknowns_wt_prod*knowns_wt_prod))
			unknowns_wt_prod_sum += unknowns_wt_prod


		if i % 1000 == 0:
			print("{:10d}".format(i))
		i += 1

	# print("Weight sum:", "{:.4f}".format(unknowns_wt_prod_sum*knowns_wt_prod))
	# print("\n")
	return unknowns_wt_prod_sum*knowns_wt_prod


# def normalize_weights(w):
# 	nw = {}
# 	for k, v in w.items():
# 		if not k.startswith('~'):
# 			w1 = w[k]
# 			w2 = w['~'+k]
# 			nw[k] = w1/(w1+w2)
# 			nw['~'+k] = w2/(w1+w2)

# 	return nw

wmc_dict = {1:1.0}

def get_wmc(w, phi, bdd):
	# if not wmc_dict:
	# 	wmc_dict = {1:1.0}
	# 	w = normalize_weights(w)
	if abs(phi) not in wmc_dict:
		level, low, high = bdd.succ(phi)

		v = bdd.var_at_level(level)

		wmc_low = get_wmc(w, low, bdd)
		wmc_high = get_wmc(w, high, bdd)
		
		wmc_dict[abs(phi)] = w[v]*wmc_high+ w["~"+v]*wmc_low

		# print("{} & {} & {} & {} & {} & {} & {} & {} & {:.4f} \\\\".format(abs(phi), low, wmc_low, high, wmc_high, v, w[v], w["~"+v], wmc_dict[abs(phi)]))

	if phi > 0:
		return wmc_dict[phi]

	else:
		return 1.0 - wmc_dict[-phi]