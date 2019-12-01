n = int(input("n = "))
p = float(input("p = "))
# n = 5
# p = 0.4

prog = "x_0_0 = T;\n"

for i in range(n):
	for j in range(i+2):
		prog += "x_{}_{} = F;\n".format(i+1, j)

prog += "\n"

for i in range(n):

	prog += "z{} = flip({});\n\nif(z{}){{\n".format(i+1, p, i+1)

	for j in range(i+1):

		prog += "\tif(x_{}_{}) {{ x_{}_{} = T }}{}\n".format(i, j, i+1, j, ";"*(j!=i))

	prog += "} else {\n"

	for j in range(i+1):

		prog += "\tif(x_{}_{}) {{ x_{}_{} = T }}{}\n".format(i, j, i+1, j+1, ";"*(j!=i))

	prog += "};\n\n"

open("programs/binomial_{}_{}.dippl".format(n,p), 'w').write(prog)