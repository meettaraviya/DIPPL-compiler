from dippl_parser import *
import argparse
from inference import *
import time
import os

from itertools import repeat
from collections import Counter

try:
	import dd.cudd as _bdd
except ImportError:
    # import dd.autoref as _bdd
    import dd.bdd as _bdd

# from numpy.random import binomial
from random import random as uniform

argparser = argparse.ArgumentParser(description='Compiles probabilistic programs to BDDs.')
argparser.add_argument('program', help='File to be compiled.', type=str)
argparser.add_argument('--pdf', help='Dump output BDD to a pdf.', action='store_const', const=True, default=False)
# argparser.add_argument('--tcompile', help='Time compilation?', action='store_const', const=True, default=False)
argparser.add_argument('--time', dest='tqueries', help='Time queries?', action='store_const', const=True, default=False)
argparser.add_argument('--queries', help='Inference queries for the BDD.', type=str, default=[], nargs='+')
argparser.add_argument('--algo', help='Algorithm to use.', type=str, required=True)
argparser.add_argument('--maxbddsize', help='Max BDD size for approximate compilation.', type=int, default=1000)
argparser.add_argument('--nsamples', help='Number of samples to take.', type=int, default=1000)


def gamma_without(x):

	if x not in gw_nodes:
		node = env.true
		for v in variable_list:
			if v != x:
				node = env.apply('and', node, equiv_nodes[v])
		gw_nodes[x] = node

	return gw_nodes[x]


def shadow_union(w1, w2):

	w1.update(w2)
	return w1

def to_wbdd_exact(ast, env, sampling_data=[{}, 1.0]):

	if ast.op == VAR:
		env.declare(ast.lchild)
		retval = env.add_expr(ast.lchild), None

	elif ast.op == TRUE:

		retval = env.true, None

	elif ast.op == FALSE:

		retval = env.false, None

	elif ast.op == NOT:

		b, w = to_wbdd(ast.lchild, env, sampling_data)
		retval = env.apply('not', b), None

	elif ast.op == AND:
		b1, w1 = to_wbdd(ast.lchild, env, sampling_data)
		b2, w2 = to_wbdd(ast.rchild, env, sampling_data)
		retval = env.apply('and', b1, b2), None 

	elif ast.op == OR:
		b1, w1 = to_wbdd(ast.lchild, env, sampling_data)
		b2, w2 = to_wbdd(ast.rchild, env, sampling_data)
		retval = env.apply('or', b1, b2), None

	elif ast.op == IFELSE:
		b, w = to_wbdd(ast.cchild, env, sampling_data)
		b_then, w_then = to_wbdd(ast.lchild, env, sampling_data)
		b_else, w_else = to_wbdd(ast.rchild, env, sampling_data)
		retval = env.ite(b, b_then, b_else), shadow_union(w_then, w_else)

	elif ast.op == FLIP:
		env.declare('f'+str(ast.cchild))
		v = env.add_expr('f'+str(ast.cchild))
		w = delta_v.copy()
		w['f'+str(ast.cchild)] = ast.rchild
		w['~f'+str(ast.cchild)] = 1.0-ast.rchild
		draw_equiv = env.apply('equiv', env.var(ast.lchild+"_"), env.var('f'+str(ast.cchild)))
		phi = env.apply('and', draw_equiv, gamma_without(ast.lchild))
		retval = phi, w

	elif ast.op == OBSERVE:
		b, w = to_wbdd(ast.lchild, env, sampling_data)
		retval = env.apply('and', b, gamma_v), delta_v


	elif ast.op == ASSIGN:
		b, w = to_wbdd(ast.rchild, env, sampling_data)
		draw_equiv = env.apply('equiv', env.var(ast.lchild+"_"), b)
		phi = env.apply('and', draw_equiv, gamma_without(ast.lchild))
		w = delta_v
		retval = phi, w

	elif ast.op == BLOCK:
		if len(ast.lchild)>0:
			b1, w1 = to_wbdd(ast.lchild[0], env, sampling_data)
			for child in ast.lchild[1:]:
				b2, w2 = to_wbdd(child, env, sampling_data)
				b2_ = env.let(let_right, b2)
				
				new_b1 = env.exist([v+'_' for v in variable_list], env.apply('and', b1, b2_))
				new_b1 = env.let(let_half_left, new_b1)
				new_w = shadow_union(w1, w2)
				
				b1 = new_b1
				w1 = new_w

				# print()
				# print(b1, ":", child)
				# print()
				# env.dump('roenvs/'+sys.argv[1]+"_" + str(b1)+'.pdf', roots=[b1])
			retval = b1, w1

		else:
			retval = gamma_v, delta_v

	# env.incref(retval[0])
	if sampling_data[0]:
		phi, w = retval
		phi = env.let(sampling_data[0], phi)
		retval = phi, w

	return retval


def to_wbdd_approximate(ast, env, sampling_data=[{}, 1.0]):

	phi, w = to_wbdd_exact(ast, env, sampling_data)
	all_nodes = env.descendants([phi])

	while len(all_nodes) > MAXBDDSIZE:
	# while len(phi) > MAXBDDSIZE:

		counts = Counter()

		for node in all_nodes:
			if node > 1:
				counts[env.var_at_level(env.succ(node)[0])] += 1

		var = counts.most_common()[0][0]
		# val = w[var] >= w['~'+var]

		marginal = get_wmc(w, env.apply('and', phi, env.var(var)), env)/get_wmc(w, phi, env)

		if uniform() < marginal:
			sampling_data[1] *= marginal
			val = True
		else:
			sampling_data[1] *= (1.0 - marginal)
			val = False

		# print("{} set to {}".format(var, val))
		# env.decref(phi)
		sampling_data[0][var] = val
		phi = env.let({var: val}, phi)
		# env.incref(phi)

		all_nodes = env.descendants([phi])

		# env.collect_garbage()

		# print(len(all_nodes))

	return phi, w


def setup_env():

	global env, gw_nodes, equiv_nodes, delta_v, gamma_v, let_right, let_left, let_half_left, exist_

	env = _bdd.BDD()
	gw_nodes = {}
	equiv_nodes = {}

	for var in env_var_order:

		env.declare(var)

		if var in variable_list:

			env.declare(var+'_')
			env.declare(var+'__')
			equiv_nodes[var] = env.apply('equiv', env.var(var), env.var(var+'_'))

	delta_v = dict(zip(
		[v for v in variable_list]+
		['~'+v for v in variable_list]+
		[v+'_' for v in variable_list]+
		['~'+v+'_' for v in variable_list]
		, repeat(0.5)))

	gamma_v = env.true

	for v in variable_list:
		gamma_v = env.apply('and', gamma_v, equiv_nodes[v])

	let_right = {v:v+'_' for v in variable_list}
	let_right.update({v+'_':v+'__' for v in variable_list})
	let_left = {v+'_':v for v in variable_list}
	let_left.update({v+'__':v+'_' for v in variable_list})
	let_half_left = {v+'__':v+'_' for v in variable_list}
	exist_ = '\E '+' '.join([v+'_' for v in variable_list]) + ' :'


def compile(program, pdf=False, queries=[], algo='exact', maxbddsize=1000, tqueries=False, nsamples=1000):

	# args = argparser.parse_args(*args, **kwargs)

	global variable_list, flip_list, env_var_order, to_wbdd, MAXBDDSIZE

	MAXBDDSIZE = maxbddsize

	program_infile = 'programs/{}.dippl'.format(program)
	ast, variable_list, flip_list, env_var_order = parse_file(program_infile)

	if algo == 'exact':
		to_wbdd = to_wbdd_exact
		setup_env()
		phi, w = to_wbdd(ast, env)
		phi = env.let(let_left, phi)

		if pdf:
			# os.remove('robdds/{}.pdf'.format(program))
			env.dump('robdds/{}.pdf'.format(program), roots=[phi])

		# for var in variable_list:
		# 	assert var not in env.support(phi)


		for query in queries:

			if tqueries:
				start = time.time()

			wmc = get_wmc(w, phi, env)
			wmc_num = get_wmc(w, env.apply('and', phi, env.add_expr(query)), env)
			# print(wmc_num)
			# print(wmc_dict)
			# for k, v in wmc_dict.items():
				# print(env.to_expr(k), v)
			print("Probability for '{}': {}".format(query, wmc_num/ wmc))

			if tqueries:
				print("Time to answer query (after compilation):", time.time()-start)

		

	elif algo == 'approximate':
		to_wbdd = to_wbdd_approximate
		setup_env()

		for query in queries:

			if tqueries:
				start = time.time()

			ans_num = 0.0
			ans_den = 0.0

			for i in range(nsamples):
				sampling_data = [{}, 1.0]
				phi, w = to_wbdd(ast, env, sampling_data)
				phi = env.let(let_left, phi)

				wmc = get_wmc(w, phi, env)
				wmc_num = get_wmc(w, env.apply('and', phi, env.add_expr(query)), env)
				p_hat = wmc_num/wmc
				q_hat = sampling_data[1]
			# print("Probability for '{}': {}".format(query, wmc_num/ wmc))

			if tqueries:
				print("Time to answer query:", time.time()-start)



if __name__ == '__main__':
	compile(**argparser.parse_args().__dict__)
