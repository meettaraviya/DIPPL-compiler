nplaces = int(input("Total number of places: "))
walklength = int(input("Number of steps: "))
pforward = float(input("Probability of going forward: "))
fname = "programs/gambler_{}_{}_{}.dippl".format(nplaces, walklength, pforward)

prog = ""

for i in range(nplaces+1):
	prog += "x{} = T;\n".format(i)

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