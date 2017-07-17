# coding: utf-8

import numpy as np


n=16
m=4
d=20
s=16

beta = 0.5
theta = 3

pPos = 0.1
pNeg = 0.08
pDecay = 0.008


# ---- Step 1: Initialisation ---- #
A = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])
P = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])
D = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])

for i, j, k in np.ndindex(m, n, d):
	for l in np.ndindex(s):
		x = np.random.random_integers(0, m-1)
		y = np.random.random_integers(0, n-1)
		cw = np.random.random()
		while x == i and y==j:
			x = np.random.random_integers(0, m-1)
			y = np.random.random_integers(0, n-1)
		while cw == 0:
			cw = np.random.random()
		D[i][j][k][l]["x"] = x
		D[i][j][k][l]["y"] = y
		D[i][j][k][l]["cw"] = cw




#S = {   "A" : np.array([1,0,0,1]),
 #       "B" : np.array([0,0,1,0]),
  #      "C" : np.array([0,1,0,1])}
S = { "A" : np.array([1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]),
      "B" : np.array([0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1]),
      "C" : np.array([0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0]) }

W = np.zeros((n), dtype = int)

#print "P", P['t']
#print "A", A['t']

seq = ["A", "B", "C"]

def reinforce(x, y, z, D):
    delta = 0
    for s in D[i][j][k]:
    	if s["cw"] > 0:
			delta = pPos * s["cw"] * A['t-1'][s["x"]][s["y"]] - pNeg * s["cw"]
			s["cw"] = s["cw"] + delta
			    #delta = pPos * Dpos[i][j][k] * A['t-1'] - pNeg * Dpos[i][j][k]


def decay(x, y, z, D):
    delta = 0

    for s in D[i][j][k]:
		if s["cw"] > 0:
			delta = pDecay * s["cw"]
			s["cw"] = s["cw"] - delta
			    #delta = pDecay * Dpos[i][j][k]


def count_ones_in_product_Dcon(G, H):
	count = 0
	for s in H:
		if s["cw"] > beta and G[s["x"], s["y"]] == 1:
			count = count + 1


	return count

def count_ones_in_product_Dpos(G, H):
	count = 0
	for s in H:
		if s["cw"] > 0 and G[s["x"], s["y"]] == 1:
			count = count + 1
	return count


for ntrials in range(30):
	for sPos in range(len(seq)):
		# ---- Step 2: Computing cell states ---- #
		W = S[seq[sPos]]
		for i, j in np.ndindex(m, n):
		    if W[j] == 1:
		        if P['t-1'][i][j] == 1:
		            A['t'][i][j] = 1
		        else:
		            tempSum = 0
		            for l in np.ndindex(m):
		                tempSum += P['t-1'][l][j]
		            if tempSum == 0:
		                A['t'][i][j] = 1
		            else:
		        		A['t'][i][j] = 0
		    else:
		        A['t'][i][j] = 0
		
		for i, j in np.ndindex(m, n):
		    P['t'][i][j] = 0
		    for q in np.ndindex(d):
		    	if count_ones_in_product_Dcon(A['t'], D[i][j][q]) > theta:
		        #if sum(sum(A['t']*Dcon[i][j][q])) > theta:
		            P['t'][i][j] = 1
		
		# ---- Step 3: Learning ---- #
		#Dpos = D > 0

		correctly_predicted = 'false'
		for i, j, k in np.ndindex(m, n, d):
		    if W[j] == 1:
		    	if P['t-1'][i][j] > 0 and count_ones_in_product_Dcon(A['t-1'], D[i][j][k]) > theta:
		        #if P['t-1'][i][j] > 0 and sum(sum(Dcon[i][j][k] * A['t-1'])) > theta:
		            reinforce(i, j, k, D)
		            correctly_predicted = 'true'

		if correctly_predicted == 'false':
			for j in np.ndindex(n):
			    if W[j] == 1:
			    	tempSum = 0
			    	for i in np.ndindex(m):
			    		tempSum += P['t-1'][i][j]

			    	if tempSum == 0:
			    		maxSum = 0
				    	for i, k in np.ndindex(m, d):
				    		currSum = count_ones_in_product_Dpos(A['t-1'], D[i][j][k])
				    	#	currSum = sum(sum(Dpos[i][j][k] * A['t-1']))
				    		maxSum = max(maxSum, currSum)
				        for i, k in np.ndindex(m, d):
				        	if count_ones_in_product_Dpos(A['t-1'], D[i][j][k]) == maxSum:
					    #    if sum(sum(Dpos[i][j][k] * A['t-1'])) == maxSum:
					            reinforce(i, j, k, D)
					            break

		for i, j, k in np.ndindex(m, n, d):
			if A['t'][i][j] == 0 and count_ones_in_product_Dcon(A['t-1'], D[i][j][k]) > theta:
		#	if A['t'][i][j] == 0 and sum(sum(Dcon[i][j][k] * A['t-1'])) > theta:
				decay(i, j, k, D) 
		
		print "W"
		print W
		print "P"
		print P['t']
		print "A"
		print A['t']
		#print "D"
		#print D

		P['t-1'] = P['t']
		A['t-1'] = A['t']
	P['t-1'] = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])
	A['t-1'] = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])

print "ERR"
