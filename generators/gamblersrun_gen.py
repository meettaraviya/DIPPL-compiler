import argparse

argparser_gen = argparse.ArgumentParser(description="Generates DIPPL programs for gamblers's run.")
argparser_gen.add_argument('nplaces', help='Total number of places.', type=int)
argparser_gen.add_argument('walklength', help='Number of steps.', type=int)
argparser_gen.add_argument('pforward', help='Probability of going forward.', type=float)

def generate(nplaces, walklength, pforward):
	# args = argparse.Namespace(*inargs, **inkwargs)
	# locals().update(kwargs)
	# nplaces = args.nplaces
	# walklength = args.walklength
	# pforward = args.pforward
	
	fname = "programs/gambler_{}_{}_{}.dippl".format(nplaces, walklength, pforward)

	prog = "x0 = T;\n"

	for i in range(1, nplaces+1):
		prog += "x{} = F;\n".format(i)

	prog += "\n"
	prog += "reached = F;\n\n"

	commonpart = ""

	commonpart += """
z = flip({});

done = F;

if(z){{{{
{{}}
}}}}
else{{{{
{{}}
}}}};

reached = reached || x{};

""".format(pforward, nplaces)

	innerpartthan = ""

	for i in range(nplaces):
		innerpartthan += """
	if(x{} && !done){{
		x{} = T;
		x{} = F;
		done = T
	}};
""".format(i,i+1,i)

	innerpartthan += """
	if(x{} && !done){{
		x{} = T;
		x{} = F;
		done = T
	}}
""".format(nplaces, nplaces-1, nplaces)

	innerpartelse = ""

	innerpartelse += """
	if(x0 && !done){
		x1 = T;
		x0 = F;
		done = T
	};
	"""

	for i in range(nplaces):
		innerpartelse += """
	if(x{} && !done){{
		x{} = T;
		x{} = F;
		done = T
	}}{}
""".format(i+1,i,i+1, ";" if i < nplaces-1 else "")




	# print(innerpartthan)

	commonpart = commonpart.format(innerpartthan, innerpartelse)


	for j in range(walklength):
		prog += commonpart

	# print(prog)
	open(fname, 'w').write(prog)


if __name__ == '__main__':
	args = argparser_gen.parse_args()
	generate(**argparser_gen.parse_args().__dict__)
