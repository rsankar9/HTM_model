# coding: utf-8

import numpy as np


n=16
m=4
d=20
s=16

beta = 0.5
theta = 5

pPos = 0.1
pNeg = 0.05
pDecay = 0.001

# ---- Step 1: Initialisation ---- #
A = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])
P = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])

D = np.zeros((m, n, d, m*n), dtype=int)
D[:,:,:,:s] = 1

for i,j,k in np.ndindex(m, n, d):
    np.random.shuffle(D[i,j,k])
    # No self connections
    while D[i,j,k,i*n+j] == 1:				# Doubt 1: i*n+j or i*j
         np.random.shuffle(D[i,j,k])

D = D.reshape(m, n, d, m, n)
D = D * np.random.uniform(0, 1, (m, n, d, m, n))

Dcon = D > beta

#S = {   "A" : np.array([1,0,0,1]),
 #       "B" : np.array([0,0,1,0]),
  #      "C" : np.array([0,1,0,1])}
S = { "A" : np.array([1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]),
      "B" : np.array([0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1]),
      "C" : np.array([0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0]) }

W = np.zeros((n), dtype = [("t", int), ("t-1", int)])

print "P", P['t']
print "A", A['t']

seq = ["A", "B", "C"]


for ntrials in range(50):
	# ---- Step 2: Computing cell states ---- #
	W['t'] = S[seq[ntrials % len(seq)]]
	for i, j in np.ndindex(m, n):
	    if W['t'][j] == 1:
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
	        if sum(sum(A['t']*Dcon[i][j][q])) > theta:
	            P['t'][i][j] = 1
	
	# ---- Step 3: Learning ---- #
	Dpos = D > 0

	def reinforce(i, j, k, D):
	    delta = pPos * Dpos[i][j][k] * A['t-1'] - pNeg * Dpos[i][j][k]
	    D = D + delta

	def decay(i, j, k, D):
	        delta = pDecay * Dpos[i][j][k]
	        D = D + delta

#	if correctly_predicted
	for i, j, k in np.ndindex(m, n, d):
	    if W['t'][j] == 1:
	        if P['t-1'][i][j] > 0 and sum(sum(Dcon[i][j][k] * A['t-1'])) > theta:
	            reinforce(i, j, k, D)

#	else
	for j in np.ndindex(n):
	    if W['t'][j] == 1:
	    	tempSum = 0
	    	for i in np.ndindex(m):
	    		tempSum += P['t-1'][i][j]

	    	if tempSum == 0:
	    		maxSum = 0
		    	for i, k in np.ndindex(m, d):
		    		currSum = sum(sum(Dpos[i][j][k] * A['t-1']))
		    		maxSum = max(maxSum, currSum)
		        for i, k in np.ndindex(m, d):
			        if sum(sum(Dpos[i][j][k] * A['t-1'])) == maxSum:
			            reinforce(i, j, k, D)
			            break

	for i, j, k in np.ndindex(m, n, d):
		if A['t'][i][j] == 0 and sum(sum(Dcon[i][j][k] * A['t-1'])) > theta:
			decay(i, j, k, D) 
	
	print "W"
	print W['t']
	print "P"
	print P['t']
	print "A"
	print A['t']
	#print "D"
	#print D

	P['t-1'] = P['t']
	W['t-1'] = W['t']
	A['t-1'] = A['t']




print "ERR"