from dippl_parser import *
import argparse
from inference import *

from itertools import repeat

try:
    import dd.cudd as _bdd
except ImportError:
    import dd.bdd as _bdd

argparser = argparse.ArgumentParser(description='Compiles probabilistic programs to BDDs.')
argparser.add_argument('program', help='File to be compiled.', type=str)
argparser.add_argument('--pdf', help='Dump output BDD to a pdf.', action='store_const', const=True, default=False)
argparser.add_argument('--queries', help='Inference queries for the BDD.', type=str, default=[], nargs='+')


def gamma_without(x):

	if x not in gw_nodes:
		node = env.true
		for v in variable_list:
			if v != x:
				node = env.apply('and', node, equiv_nodes[v])
		gw_nodes[x] = node

	return gw_nodes[x]


def shadow_union(w1, w2):

	w3 = w1.copy()
	for x,y in w2.items():
		w3[x] = y

	return w3


def to_wbdd(ast, env):

	if ast.op == VAR:
		env.declare(ast.lchild)
		return env.add_expr(ast.lchild), None

	elif ast.op == TRUE:

		return env.true, None

	elif ast.op == FALSE:

		return env.false, None

	elif ast.op == NOT:

		b, w = to_wbdd(ast.lchild, env)
		return env.apply('not', b), None

	elif ast.op == AND:
		b1, w1 = to_wbdd(ast.lchild, env)
		b2, w2 = to_wbdd(ast.rchild, env)
		return env.apply('and', b1, b2), None 

	elif ast.op == OR:
		b1, w1 = to_wbdd(ast.lchild, env)
		b2, w2 = to_wbdd(ast.rchild, env)
		return env.apply('or', b1, b2), None

	elif ast.op == IFELSE:
		b, w = to_wbdd(ast.cchild, env)
		b_then, w_then = to_wbdd(ast.lchild, env)
		b_else, w_else = to_wbdd(ast.rchild, env)
		return env.ite(b, b_then, b_else), shadow_union(w_then, w_else)

	elif ast.op == FLIP:
		env.declare('f'+str(ast.cchild))
		v = env.add_expr('f'+str(ast.cchild))
		w = delta_v.copy()
		w['f'+str(ast.cchild)] = ast.rchild
		w['~f'+str(ast.cchild)] = 1.0-ast.rchild
		draw_equiv = env.apply('equiv', env.var(ast.lchild+"_"), env.var('f'+str(ast.cchild)))
		phi = env.apply('and', draw_equiv, gamma_without(ast.lchild))
		return phi, w

	elif ast.op == OBSERVE:
		b, w = to_wbdd(ast.lchild, env)
		return env.apply('and', b, gamma_v), delta_v


	elif ast.op == ASSIGN:
		b, w = to_wbdd(ast.rchild, env)
		draw_equiv = env.apply('equiv', env.var(ast.lchild+"_"), b)
		phi = env.apply('and', draw_equiv, gamma_without(ast.lchild))
		w = delta_v
		return phi, w

	elif ast.op == BLOCK:
		if len(ast.lchild)>0:
			b1, w1 = to_wbdd(ast.lchild[0], env)
			for child in ast.lchild[1:]:
				b2, w2 = to_wbdd(child, env)
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
			return b1, w1

		else:
			return gamma_v, delta_v

	else:
		print("Error:",ast.op)
		exit(0)

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
		, repeat(1.0)))

	gamma_v = env.true

	for v in variable_list:
		gamma_v = env.apply('and', gamma_v, equiv_nodes[v])

	let_right = {v:v+'_' for v in variable_list}
	let_right.update({v+'_':v+'__' for v in variable_list})
	let_left = {v+'_':v for v in variable_list}
	let_left.update({v+'__':v+'_' for v in variable_list})
	let_half_left = {v+'__':v+'_' for v in variable_list}
	exist_ = '\E '+' '.join([v+'_' for v in variable_list]) + ' :'


def main(args):

	global variable_list, flip_list, env_var_order

	program_infile = 'programs/{}.dippl'.format(args.program)
	ast, variable_list, flip_list, env_var_order = parse_file(program_infile)

	setup_env()

	phi, w = to_wbdd(ast, env)

	if args.pdf:
		env.dump('robdds/{}.pdf'.format(args.program), roots=[phi])

	wmc = get_wmc(w, phi, env)

	for var in variable_list:
		assert var not in env.support(phi)

	phi = env.let(let_left, phi)

	for query in args.queries:
		wmc_num = get_wmc(w, env.apply('and', phi, env.add_expr(query)), env)
		print("Probability for '{}': {}".format(query, wmc_num/ wmc))



if __name__ == '__main__':
	args = argparser.parse_args()
	main(args)
