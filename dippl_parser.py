import ply.lex as lex
import ply.yacc as yacc

reserved = {
	'skip': 'SKIP',
	'observe': 'OBSERVE',
	'flip': 'FLIP',
	'if': 'IF',
	'else': 'ELSE',
	# 'repeat': 'REPEAT',
	'T' : 'TRUE',
	'F' : 'FALSE',
}


tokens = [
		'LPAREN',
		'RPAREN',
		'LBRACE',
		'RBRACE',
		# 'LBOX',
		# 'RBOX',
		# 'COLONEQ',
		'EQUALS',
		# 'SIM',
		'OR',
		'AND',
		'NOT',
		# 'TRUE',
		# 'FALSE',
		'VAR',
		'SCOLON',
		'FLOAT',
		# 'INTEGER',
		# 'INDEX',
	] + list(reserved.values())

t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
# t_LBOX = r'\['
# t_RBOX = r'\]'
# t_COLONEQ = r':='
t_EQUALS = r'='
# t_SIM = r'~'
t_OR = r'\|\|'
t_AND = r'&&'
t_NOT = r'\!'
# t_TRUE = r'T'
# t_FALSE = r'F'
t_SCOLON = r';'

def t_VAR(t):
	r'[a-zA-Z][a-zA-Z0-9_]*'
	t.type = reserved.get(t.value,'VAR')
	return t

def t_FLOAT(t):
	r'\d+\.\d+'
	t.value = float(t.value)
	return t

# def t_INTEGER(t):
# 	r'\d+'
# 	t.value = int(t.value)
# 	return t

# def t_INDEX(y):
# 	r'\d+'
# 	t.value = float(t.value)
# 	return t

t_ignore  = ' \t'

def t_newline(t):
	r'\n+'
	t.lexer.lineno += len(t.value)

def t_error(t):
	print("Illegal character:", t.value[0])
	t.lexer.skip(1)


VAR, NOT, AND, OR, IFELSE, FLIP, ASSIGN, OBSERVE, BLOCK, TRUE, FALSE = range(10, 21)

precedence = (
    ('left', 'AND', 'OR'),
    ('left', 'NOT'),
)


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


def p_program_base(p):
	'program : statement'
	p[0] = [p[1]]


def p_program_recurse(p):
	'program : statement SCOLON program'
	p[0] = [p[1]]+p[3]


def p_expression_variable(p):
	'expression : VAR'
	if p[1] not in variable_list:
		variable_list.append(p[1])
		bdd_var_order.append(p[1])
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
	if p[1] not in variable_list:
		variable_list.append(p[1])
		bdd_var_order.append(p[1])
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


def p_statement_draw(p):
	'statement : VAR EQUALS FLIP LPAREN FLOAT RPAREN'
	global flip_id
	flip_id += 1
	flip_list.append('f'+str(flip_id))
	bdd_var_order.append('f'+str(flip_id))
	if p[1] not in variable_list:
		variable_list.append(p[1])
		bdd_var_order.append(p[1])
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
 
# Build the parser

def build_parser():
	global dippl_lexer, dippl_parser
	lexer = lex.lex()
	dippl_parser = yacc.yacc()


def parse_string(s):
	global flip_id, dippl_parser, dippl_lexer, variable_list, flip_list, bdd_var_order
	build_parser()
	flip_id = 0
	variable_list = []
	flip_list = []
	bdd_var_order = []
	ast = AST(BLOCK, dippl_parser.parse(s))
	return ast, variable_list, flip_list, bdd_var_order


def parse_file(f):
	return parse_string(open(f).read())


# build_parser()


if __name__ == '__main__':

	parse_file('programs/gambler_2_1_0.4.dippl')
	parse_file('programs/gambler_2_2_0.4.dippl')