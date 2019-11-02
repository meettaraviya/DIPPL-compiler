import ply.lex as lex

reserved = {
	'skip': 'SKIP',
	'observe': 'OBSERVE',
	'flip': 'FLIP',
	'if': 'IF',
	'else': 'ELSE',
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
		'TRUE',
		'FALSE',
		'VAR',
		'SCOLON',
		'NUMBER',
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
t_TRUE = r'T'
t_FALSE = r'F'
t_SCOLON = r';'

def t_VAR(t):
	r'[a-zA-Z][a-zA-Z0-9]*'
	t.type = reserved.get(t.value,'VAR')
	return t

def t_NUMBER(t):
	r'\d+(\.\d+)?'
	t.value = float(t.value)
	return t

# def t_INDEX(y):
# 	r'\d+'
# 	t.value = float(t.value)
# 	return t

t_ignore  = ' \t'

def t_newline(t):
	r'\n+'
	t.lexer.lineno += len(t.value)

def t_error(t):
	print("Illegal character '%s'" % t.value[0])
	t.lexer.skip(1)

lexer = lex.lex()