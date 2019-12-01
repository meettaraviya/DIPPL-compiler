import ply.yacc as yacc
from dippl_lex import tokens
import sys
from itertools import repeat, product
import timeit
import time

try:
    import dd.cudd as _bdd
except ImportError:
    import dd.bdd as _bdd

bddvars = {}

VAR, NOT, AND, OR, IFELSE, FLIP, ASSIGN, OBSERVE, BLOCK, TRUE, FALSE = range(10, 21)

precedence = (
    ('left', 'AND', 'OR'),
    ('left', 'NOT'),
)

all_vars = set()
all_flips = set()

equiv_nodes = {}
gw_nodes = {}

def shadow_union(w1, w2):

	w3 = w1.copy()
	for x,y in w2.items():
		w3[x] = y

	return w3


def gamma_without(x):

	if x not in gw_nodes:
		node = bdd.true
		for v in all_vars:
			if v != x:
				node = bdd.apply('and', node, equiv_nodes[v])
		gw_nodes[x] = node

	return gw_nodes[x]


class AST:

	def __init__(self, op, lchild=None, rchild=None, cchild=None):

		self.op = op
		self.lchild = lchild
		self.rchild = rchild
		self.cchild = cchild

	def __repr__(self):

		if self.op == VAR:

			return str(self.lchild)

		elif self.op >= TRUE:

			return 'T' if self.op == TRUE else 'F'

		elif self.op == NOT:

			return '!' + str(self.lchild)

		elif self.op == AND:
			return '(' + str(self.lchild) + ' & ' + str(self.rchild) + ')'

		elif self.op == OR:
			return '(' + str(self.lchild) + ' | ' + str(self.rchild) + ')'

		elif self.op == IFELSE:
			return 'if({}) {{\n{}\n}} else {{\n{}\n}})'.format(self.cchild, self.lchild, self.rchild)

		elif self.op == FLIP:
			return '{} ~ flip<{}>({})'.format(self.lchild, self.cchild, self.rchild)

		elif self.op == ASSIGN:
			return '{} := {}'.format(self.lchild, self.rchild)

		elif self.op == OBSERVE:
			return 'observe({})'.format(self.lchild)

		elif self.op == BLOCK:
			return ';\n'.join([repr(c) for c in self.lchild])


	def to_wbdd(self, bdd):

		if self.op == VAR:
			assert not self.lchild.endswith("_")
			return bdd.add_expr(self.lchild), None

		elif self.op == TRUE:

			return bdd.true, None

		elif self.op == FALSE:

			return bdd.false, None

		elif self.op == NOT:

			b, w = self.lchild.to_wbdd(bdd)
			return bdd.apply('not', b), None

		elif self.op == AND:
			b1, w1 = self.lchild.to_wbdd(bdd)
			b2, w2 = self.rchild.to_wbdd(bdd)
			return bdd.apply('and', b1, b2), None 

		elif self.op == OR:
			b1, w1 = self.lchild.to_wbdd(bdd)
			b2, w2 = self.rchild.to_wbdd(bdd)
			return bdd.apply('or', b1, b2), None

		elif self.op == IFELSE:
			b, w = self.cchild.to_wbdd(bdd)
			b_then, w_then = self.lchild.to_wbdd(bdd)
			b_else, w_else = self.rchild.to_wbdd(bdd)
			return bdd.ite(b, b_then, b_else), shadow_union(w_then, w_else)

		elif self.op == FLIP:
			v = bdd.add_expr('f'+str(self.cchild))
			w = delta_v.copy()
			w['f'+str(self.cchild)] = self.rchild
			w['~f'+str(self.cchild)] = 1.0-self.rchild
			draw_equiv = bdd.apply('equiv', bdd.var(self.lchild+"_"), bdd.var('f'+str(self.cchild)))
			phi = bdd.apply('and', draw_equiv, gamma_without(self.lchild))
			return phi, w

		elif self.op == OBSERVE:
			b, w = self.lchild.to_wbdd(bdd)
			return bdd.apply('and', b, gamma_v), delta_v


		elif self.op == ASSIGN:
			b, w = self.rchild.to_wbdd(bdd)
			draw_equiv = bdd.apply('equiv', bdd.var(self.lchild+"_"), b)
			phi = bdd.apply('and', draw_equiv, gamma_without(self.lchild))
			w = delta_v
			return phi, w

		elif self.op == BLOCK:
			if len(self.lchild)>0:
				b1, w1 = self.lchild[0].to_wbdd(bdd)
				for child in self.lchild[1:]:
					b2, w2 = child.to_wbdd(bdd)
					b2_ = bdd.let(let_right, b2)
					
					new_b1 = bdd.exist([v+'_' for v in all_vars], bdd.apply('and', b1, b2_))
					new_b1 = bdd.let(let_half_left, new_b1)
					new_w = shadow_union(w1, w2)
					
					b1 = new_b1
					w1 = new_w

					# print()
					# print(b1, ":", child)
					# print()
					# bdd.dump('robdds/'+sys.argv[1]+"_" + str(b1)+'.pdf', roots=[b1])
				return b1, w1

			else:
				return gamma_v, delta_v

		else:
			print("Error:",self.op)
			exit(0)


def p_program_base(p):
	'program : statement'
	p[0] = [p[1]]


def p_program_recurse(p):
	'program : statement SCOLON program'
	p[0] = [p[1]]+p[3]


def p_expression_variable(p):
	'expression : VAR'
	bdd.declare(p[1])
	bdd.declare(p[1]+'_')
	bdd.declare(p[1]+'__')
	all_vars.add(p[1])
	if p[1] not in equiv_nodes:
		equiv_nodes[p[1]] = bdd.apply('equiv', bdd.var(p[1]), bdd.var(p[1]+'_'))
	p[0] = AST(VAR, p[1])


def p_expression_true(p):
	'expression : TRUE'
	p[0] = AST(TRUE)

def p_expression_false(p):
	'expression : FALSE'
	p[0] = AST(FALSE)


def p_expression_and(p):
	'expression : expression AND expression'
	p[0] = AST(AND, p[1], p[3])


def p_expression_or(p):
	'expression : expression OR expression'
	p[0] = AST(OR, p[1], p[3])


def p_expression_not(p):
	'expression : NOT expression'
	p[0] = AST(NOT, p[2])


def p_statement_assign(p):
	'statement : VAR EQUALS expression'
	# 'statement : VAR COLONEQ expression'
	bdd.declare(p[1])
	bdd.declare(p[1]+'_')
	bdd.declare(p[1]+'__')
	all_vars.add(p[1])
	if p[1] not in equiv_nodes:
		equiv_nodes[p[1]] = bdd.apply('equiv', bdd.var(p[1]), bdd.var(p[1]+'_'))
	p[0] = AST(ASSIGN, p[1], p[3])


def p_expression_paren(p):
	'expression : LPAREN expression RPAREN'
	p[0] = p[2]


def p_statement_ifelse(p):
	'statement : IF LPAREN expression RPAREN LBRACE program RBRACE ELSE LBRACE program RBRACE'
	p[0] = AST(IFELSE, AST(BLOCK, p[6]), AST(BLOCK, p[10]), p[3])


# def p_statement_repeat(p):
# 	'statement : REPEAT LPAREN INTEGER RPAREN LBRACE program RBRACE'

# 	p[0] = AST(BLOCK, p[6]*p[3])


def p_statement_if(p):
	'statement : IF LPAREN expression RPAREN LBRACE program RBRACE'
	p[0] = AST(IFELSE, AST(BLOCK, p[6]), AST(BLOCK, []), p[3])


flip_id = 0

def p_statement_draw(p):
	'statement : VAR EQUALS FLIP LPAREN FLOAT RPAREN'
	global flip_id
	flip_id += 1
	bdd.declare('f'+str(flip_id))
	all_flips.add('f'+str(flip_id))
	bdd.declare(p[1])
	bdd.declare(p[1]+'_')
	bdd.declare(p[1]+'__')
	all_vars.add(p[1])
	if p[1] not in equiv_nodes:
		equiv_nodes[p[1]] = bdd.apply('equiv', bdd.var(p[1]), bdd.var(p[1]+'_'))
	p[0] = AST(FLIP, p[1], p[5], flip_id)


def p_statement_observe(p):
	'statement : OBSERVE LPAREN expression RPAREN'
	p[0] = AST(OBSERVE, p[3])


def p_program_skip(p):
	'program : SKIP'
	p[0] = []


def p_program_blank(p):
	'program : '
	p[0] = []


gp = None
# Error rule for syntax errors
def p_error(p):
	print("Error:", p)
	# print("Syntax error at line:", p.lexer.lineno)
	# p[0] = 123
 
bdd = _bdd.BDD()
# Build the parser
parser = yacc.yacc()

if len(sys.argv) < 2:
	print("Missing filename")
	exit(0)

def wmc_bruteforce(w, phi, bdd):
	
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


def normalize_weights(w):
	nw = {}
	for k, v in w.items():
		if not k.startswith('~'):
			w1 = w[k]
			w2 = w['~'+k]
			nw[k] = w1/(w1+w2)
			nw['~'+k] = w2/(w1+w2)

	return nw

wmc_dict = {}

def wmc(w, phi, bdd, init=False):
	global wmc_dict
	if init:
		wmc_dict = {1:1.0}
	if abs(phi) not in wmc_dict:
		level, low, high = bdd.succ(phi)

		v = bdd.var_at_level(level)

		wmc_low = wmc(w, low, bdd)
		wmc_high = wmc(w, high, bdd)
		
		wmc_dict[abs(phi)] = w[v]*wmc_high + w["~"+v]*wmc_low

	if phi > 0:
		return wmc_dict[phi]

	else:
		return 1.0 - wmc_dict[-phi]



start = time.time()
s = open('programs/'+sys.argv[1]+'.dippl').read()
end = time.time()
print("Time to read and parse to AST:",end-start)
# ast = AST(BLOCK, parser.parse(s, debug=True))
ast = AST(BLOCK, parser.parse(s))

gamma_v = bdd.true

for v in all_vars:
	gamma_v = bdd.apply('and', gamma_v, equiv_nodes[v])

delta_v = dict(zip(
	[v for v in all_vars]+
	['~'+v for v in all_vars]+
	[v+'_' for v in all_vars]+
	['~'+v+'_' for v in all_vars]
	, repeat(1.0)))

let_right = {v:v+'_' for v in all_vars}
let_right.update({v+'_':v+'__' for v in all_vars})
let_left = {v+'_':v for v in all_vars}
let_left.update({v+'__':v+'_' for v in all_vars})
let_half_left = {v+'__':v+'_' for v in all_vars}
exist_ = '\E '+' '.join([v+'_' for v in all_vars]) + ' :'


start = time.time()
phi, w = ast.to_wbdd(bdd)
end = time.time()
print("Time to convert AST to ROBDD:",end-start)
phi = bdd.let(let_left, phi)

w = normalize_weights(w)

print("Result node:", phi)
print("WMC linear:", wmc(w, phi, bdd, init=True))
print("Time taken for WMC linear:",timeit.timeit('wmc(w, phi, bdd)', globals=globals(), number=1))
# print("WMC brute force:", wmc_bruteforce(w, phi, bdd))
# print("Time taken for WMC brute force:",timeit.timeit('wmc_bruteforce(w, phi, bdd)', globals=globals(), number=1))


bdd.dump('robdds/'+sys.argv[1]+'.pdf', roots=[phi])
