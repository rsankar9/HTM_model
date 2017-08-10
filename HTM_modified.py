# coding: utf-8

import numpy as np


n=21	# no. of columns in a layer
m=6		# no. of cells per column
d=20	# no. of distal segments per cell
s=16	# no. of potential synapses per segment

beta = 0.5
theta = 3
synPerm = 0.21

pPos = 0.8
pNeg = 0.2
pDecay = 0.008


# ---- Step 1: Initialisation ---- #
A = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])
P = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])
D = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])

for i, j, k in np.ndindex(m, n, d):								# initialising each distal segment with synapses to random cells with random strength
	for l in np.ndindex(s):
		x = np.random.random_integers(0, m-1)
		y = np.random.random_integers(0, n-1)
	#	cw = np.random.random()
		while x == i and y==j:
			x = np.random.random_integers(0, m-1)
			y = np.random.random_integers(0, n-1)
	#	while cw == 0:
	#		cw = np.random.random()
		D[i][j][k][l]["x"] = x
		D[i][j][k][l]["y"] = y
		D[i][j][k][l]["cw"] = synPerm




#S = {   "A" : np.array([1,0,0,1]),
 #       "B" : np.array([0,0,1,0]),
  #      "C" : np.array([0,1,0,1])}
#S = { "A" : np.array([1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]),
#      "B" : np.array([0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1]),
#      "C" : np.array([0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0]), 
#      "D" : np.array([0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0]), 
#      "X" : np.array([0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0]), 
#      "Y" : np.array([1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1]), 
#      "end" : np.array([0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0])  }	# encoding for each syllable

S = { "A" : np.array([0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
      "B" : np.array([0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0]),
      "C" : np.array([0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0]), 
      "D" : np.array([0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0]), 
      "X" : np.array([0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0]), 
      "Y" : np.array([1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1]), 
      "end" : np.array([0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0])  }	# encoding for each syllable


W = np.zeros((n), dtype = int)

#print "P", P['t']
#print "A", A['t']

seq1 = ["A", "B", "C", "D", "end"]
seq2 = ["X", "B", "C", "Y", "end"]

sequences = [seq1, seq2]

def reinforce(x, y, z, D, Dnew):
    delta = 0
    for l in np.ndindex(s):
 		syn = D[x][y][z][l]
 		if syn["cw"] > 0:
 			delta = pPos * A['t-1'][syn["x"]][syn["y"]] - pNeg 
		#	s["cw"] = s["cw"] + delta
		Dnew[x][y][z][l] = syn
		Dnew[x][y][z][l]["cw"] = syn["cw"] + delta
			    #delta = pPos * Dpos[i][j][k] * A['t-1'] - pNeg * Dpos[i][j][k]


def decay(x, y, z, D, Dnew):
    delta = 0
    #for s in D[x][y][z]:
    for l in np.ndindex(s):
    	syn = D[x][y][z][l]
    	if syn["cw"] > 0:
    		delta = pDecay 
		Dnew[x][y][z][l] = syn
		Dnew[x][y][z][l]["cw"] = syn["cw"] - delta
			    #delta = pDecay * Dpos[i][j][k]


def count_ones_in_product_Dcon(G, H):
	count = 0
	for syn in H:
		if syn["cw"] > beta and G[syn["x"], syn["y"]] == 1:
			count = count + 1
	return count

def count_ones_in_product_Dpos(G, H):
	count = 0
	for syn in H:
		if syn["cw"] > 0 and G[syn["x"], syn["y"]] == 1:
			count = count + 1
	return count

for seq in sequences:
	for ntrials in range(500):
	# ---- Step 2: Computing cell states ---- #
		for syllable in seq:
			
			W = S[syllable]									# marks winning columns
			A['t'] = 0
			active_cells = []
			for j in np.ndindex(n):
				if W[j] == 1:
					found_active = 0
					for i in np.ndindex(m):
						if P['t-1'][i][j] == 1:
							A['t'][i][j] = 1					# cell activated if present in winning column and if predicted previously
							found_active = 1
							active_cells.insert(0, (i,j))
					if found_active==0:
						for i in np.ndindex(m):
							A['t'][i][j] = 1				# cell activated if present in winning column and if no cell in the column had been predicted
							active_cells.insert(0, (i,j))
		    		
			for i, j in np.ndindex(m, n):					# computing predictive state for this time step
			    P['t'][i][j] = 0
			    for q in np.ndindex(d):
			    	if count_ones_in_product_Dcon(A['t'], D[i][j][q]) > theta:
			        #if sum(sum(A['t']*Dcon[i][j][q])) > theta:
			            P['t'][i][j] = 1					# if no. of connected synapses to active cells > theta in any distal segment of current cell
			            break
			            
			
			# ---- Step 3: Learning ---- #
			#Dpos = D > 0

			Dnew = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])
			for i, j, k in np.ndindex(m, n, d):				# active dendritic segment reinforced if cell predicted correctly
			    if W[j] == 1:
			    	if P['t-1'][i][j] > 0 and count_ones_in_product_Dcon(A['t-1'], D[i][j][k]) > theta:
			        #if P['t-1'][i][j] > 0 and sum(sum(Dcon[i][j][k] * A['t-1'])) > theta:
			        	reinforce(i, j, k, D, Dnew)

			for j in np.ndindex(n):						
			    if W[j] == 1:
			    	tempSum = 0
			    	for i in np.ndindex(m):				# to check if current winning column was predicted
			    		tempSum += P['t-1'][i][j]

			    	if tempSum == 0:					# if current winning column not predicted
			    		maxSum = 0
				    	for i, k in np.ndindex(m, d):	# to find distal segment in current column closest to activation
				    		currSum = count_ones_in_product_Dpos(A['t-1'], D[i][j][k])
				    	#	currSum = sum(sum(Dpos[i][j][k] * A['t-1']))
				    		maxSum = max(maxSum, currSum)
				        for i, k in np.ndindex(m, d):
				        	if count_ones_in_product_Dpos(A['t-1'], D[i][j][k]) == maxSum:
					    #    if sum(sum(Dpos[i][j][k] * A['t-1'])) == maxSum:
					        	reinforce(i, j, k, D, Dnew)
					        else:							# no match found, hence updating synapses
								list_active = active_cells	
								np.random.shuffle(list_active)
								for l in np.ndindex(s):
									if D[i][j][k][l]["cw"]<beta:
										if len(list_active)==0:		
											break
										r = np.random.randint(0, m*n)
										pos = r%len(list_active)			# randomly pick an active cell
										Dnew[i][j][k][l]["x"] = list_active[pos][0]		# replace existing synapse with synapse to active cell
										Dnew[i][j][k][l]["y"] = list_active[pos][1]
									#	Dnew[i][j][k][l]["cw"] = beta + np.random.random()*(1-beta)	# to ensure it has a chance to become a strong connection
										Dnew[i][j][k][l]["cw"] = synPerm	
										list_active.pop(pos)

			for i, j, k in np.ndindex(m, n, d):			# to negatively reinforce active segments of inactive cells
				if A['t'][i][j] == 0 and count_ones_in_product_Dcon(A['t-1'], D[i][j][k]) > theta:
			#	if A['t'][i][j] == 0 and sum(sum(Dcon[i][j][k] * A['t-1'])) > theta:
					decay(i, j, k, D, Dnew)
			


			out = "None"
			outputW = np.zeros(n, dtype=int)
			for j in np.ndindex(n):
				column_active = 0
				for i in np.ndindex(m):
					if A['t'][i][j]==1:
						column_active = 1
						outputW[j] = 1
						break
				
			for syll in seq:
				flag = 0
				for j in np.ndindex(n):
					if S[syll][j] != outputW[j]:
						flag = 1
						break
				if flag == 0:
					out = syll
					break

			pred = ""
			predW = np.zeros(n, dtype=int)
			for j in np.ndindex(n):
				column_active = 0
				for i in np.ndindex(m):
					if P['t-1'][i][j]==1:
						column_active = 1
						predW[j] = 1
						break
				
			for syll in seq:
				flag = 0
				for j in np.ndindex(n):
					if S[syll][j] == 1 and predW[j]!=1:
						flag = 1
						break
				if flag == 0:
					pred = pred+syll

			if (ntrials+1)%100==0:		
				print "trial: #", ntrials
				print "input syllable: ", syllable
				print W
				print "predicted: ", pred
				print P['t-1']
				print "output syllable: ", out
				print A['t']
	

			D = Dnew
			P['t-1'] = P['t']
			A['t-1'] = A['t']
			A['t'] = 0


		# reset on encountering "end" syllable
		P['t-1'] = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])	
		A['t-1'] = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])

print "Testing"

test1 = ["A", "B", "C"]
test2 = ["X", "B", "C"]
tests = [test1, test2]

for test in tests:
	print "New test: ", test
	for syllable in test:
		W = S[syllable]									# marks winning columns
		A['t'] = 0
		for j in np.ndindex(n):
			if W[j] == 1:
				found_active = 0
				for i in np.ndindex(m):
					if P['t-1'][i][j] == 1:
						A['t'][i][j] = 1					# cell activated if present in winning column and if predicted previously
						found_active = 0
				if found_active==0:
					for i in np.ndindex(m):
						A['t'][i][j] = 1				# cell activated if present in winning column and if no cell in the column had been predicted
						active_cells.insert(0, (i,j))
	    		
		for i, j in np.ndindex(m, n):					# computing predictive state for this time step
		    P['t'][i][j] = 0
		    for q in np.ndindex(d):
		    	if count_ones_in_product_Dcon(A['t'], D[i][j][q]) > theta:
		        #if sum(sum(A['t']*Dcon[i][j][q])) > theta:
		            P['t'][i][j] = 1					# if no. of connected synapses to active cells > theta in any distal segment of current cell

		out = "None"
		outputW = np.zeros(n, dtype=int)
		for j in np.ndindex(n):
			column_active = 0
			for i in np.ndindex(m):
				if A['t'][i][j]==1:
					column_active = 1
					outputW[j] = 1
					break
			
		for syll in seq:
			flag = 0
			for j in np.ndindex(n):
				if S[syll][j] != outputW[j]:
					flag = 1
					break
			if flag == 0:
				out = syll
				break

		pred = ""
		predW = np.zeros(n, dtype=int)
		for j in np.ndindex(n):
			column_active = 0
			for i in np.ndindex(m):
				if P['t'][i][j]==1:
					column_active = 1
					predW[j] = 1
					break
			
		for syll in seq:
			flag = 0
			for j in np.ndindex(n):
				if S[syll][j] == 1 and predW[j]!=1:
					flag = 1
					break
			if flag == 0:
				pred = pred+syll

		print "input syllable: ", syllable
		print W
		print "predicted: ", pred
		print P['t']
		print "output syllable: ", out
		print A['t']

		P['t-1'] = P['t']
		A['t-1'] = A['t']
		A['t'] = 0

	P['t-1'] = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])	
	A['t-1'] = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])
			            


print "ERR"
