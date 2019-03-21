import math
import numpy as np
import itertools

def actionQuant(n, m, z):
	k_prime = []
	n_prime = 0
	y = []
	for i in range(0,m):
		val = math.floor(n*z[i] + 0.5)
		k_prime.append(val)
		n_prime = n_prime + val
	if n_prime == sum(z):
		print('n_prime')
		y = [x / n for x in k_prime]
		return y
	else:
		delta = []
		for j in range(0,m):
			val = k_prime[j] - n*z[j]
			delta.append(val)

		delta.sort()
		cap_delta = n_prime - n
		for j in range(m):
			if cap_delta > 0:
				if j <= (m - cap_delta - 1):
					y.append(k_prime[j])
				else:
					y.append(k_prime[j] - 1)
			else:
				if j <= abs(cap_delta):
					y.append(k_prime[j])
				else:
					y.append(k_prime[j])

		return [x/n for x in y]

def action_space(n_stocks, options):
	actions = []
	for i in itertools.product(options, repeat=n_stocks):
		if sum(i) == 1:
			actions.append(i)
	return actions

if __name__ =="__main__":
	n_stocks = 4
	options = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	actions = action_space(n_stocks, options)
	print(actions)
	print(len(actions))
