# coding: utf-8

import numpy as np


n=21				# no. of columns in a layer
m=6					# no. of cells per column
d=16					# no. of distal segments per cell
s=16				# no. of potential synapses per segment

beta = 0.5
theta = 3
synPerm = 0.21		# initial synaptic permanence
nStepReplace = 20

pPos = 0.1	# 0.6
pNeg = 0.1	# 0.4
pDec = 0.01	# 0.008


# ---- Step 1: Initialisation ---- #
A = np.zeros((n), dtype = [("t", int), ("t-1", int)])
P = np.zeros((n), dtype = [("t", int), ("t-1", int)])
D = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])
Dnew = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])

for i, j, k in np.ndindex(m, n, d):								# initialising each distal segment with synapses to random cells with random strength
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
		D[i][j][k][l]["cw"] = cw		#	synPerm

S = { "A" : np.array([0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
      "B" : np.array([0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0]),
      "C" : np.array([0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0]), 
      "D" : np.array([0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0]), 
      "X" : np.array([0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0]), 
      "Y" : np.array([1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1]), 
      "end" : np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0])  }	# encoding for each syllable


W = np.zeros((n), dtype = int)								# feed-forward input representing a syllable

seq1 = ["A", "B", "C", "D", "end"]
seq2 = ["X", "B", "C", "Y", "end"]

sequences = [seq1, seq2]									# list of sequences to be trained on

def isSegmentActive(i, j, k, D, Mat):						# if no. of connected synapses to active cells > theta in any distal segment of current cell
	count = 0
	for l in np.ndindex(s):
 		synapse = D[i][j][k][l]
		if synapse["cw"] > beta and ((Mat[synapse["y"]] & (1<<synapse["x"]))!=0):
			count = count + 1
	return count > theta

def count_positive_synapses(Mat, Segment):
	count = 0
	for l in np.ndindex(s):
		synapse = Segment[l]
		if synapse["cw"] > 0 and (Mat[synapse["y"]] & (1<<synapse["x"]))!=0:
			count = count + 1
	return count

def reinforce(x, y, z, D, Dnew):
	for l in np.ndindex(s):
		synapseOld = D[x][y][z][l]
		synapseNew = Dnew[x][y][z][l]
		delta = 0.0
		if ((A['t-1'][synapseOld["y"]]) & (1<<synapseOld["x"])) != 0:
			delta = delta + pPos							# reinforcing synapses to active cells
		else:
			delta = delta - pNeg							# negatively reinforcing synapses to inactive cells
		synapseNew["cw"] = synapseOld["cw"] + delta
		if synapseNew["cw"] < 0.0:
			synapseNew["cw"] = 0.0
		if synapseNew["cw"] > 1.0:
			synapseNew["cw"] = 1.0

		Dnew[x][y][z][l]["x"] = synapseOld["x"]
		Dnew[x][y][z][l]["y"] = synapseOld["y"]
		Dnew[x][y][z][l]["cw"] = synapseNew["cw"]


def decay(x, y, z, D, Dnew):
	for l in np.ndindex(s):
		synapseOld = D[x][y][z][l]
		synapseNew = Dnew[x][y][z][l]
		synapseNew["cw"] = synapseOld["cw"] - pDec
		if synapseNew["cw"] < 0.0:
			synapseNew["cw"] = 0.0
		if synapseNew["cw"] > 1.0:
			synapseNew["cw"] = 1.0
		Dnew[x][y][z][l]["x"] = synapseOld["x"]
		Dnew[x][y][z][l]["y"] = synapseOld["y"]
		Dnew[x][y][z][l]["cw"] = synapseNew["cw"]

	
active_cells = []

for seq in sequences:
	for ntrials in range(500):
		for syllable in seq:
			W = S[syllable]									# marks winning columns
			A['t'] = 0
			P['t'] = 0

			# ---- Step 3: Learning ---- #
			Dnew = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])
			np.copyto(Dnew, D)  # Dnew = D

			for j in np.ndindex(n):							# active dendritic segment reinforced if cell predicted correctly
			    if W[j] == 1:
			    	if P['t-1'][j] != 0:
				    	for i in range(m):
				    		if ((P['t-1'][j] & (1<<i)) != 0):
				    			for k in range(d):
					    			if isSegmentActive(i, j, k, D, A['t-1']) == True:
				        				reinforce(i, j, k, D, Dnew)

			    	else:									# if current winning column not predicted
			    		maxSum = 0
			    		maxRow = np.random.random_integers(0, m-1)
			    		maxSeg = np.random.random_integers(0, d-1)
				    	for i, k in np.ndindex(m, d):		# to find distal segment in current column closest to activation
				    		currSum = count_positive_synapses(A['t-1'], D[i][j][k])
				    		if maxSum < currSum:
					    		maxSum = currSum
					    		maxRow = i
					    		maxSeg = k
				    	reinforce(maxRow, j, maxSeg, D, Dnew)

				    	if ntrials % nStepReplace == 0:
					       	for i, k in np.ndindex(m, d):
					        	if i!=maxRow or k!=maxSeg:						# no match found, hence updating synapses
					        		list_active = list(active_cells)
					        		np.random.shuffle(list_active)
					        		for l in np.ndindex(s):
					        			if len(list_active) == 0:
					        				break
										if D[i][j][k][l]["cw"] < beta:			# maybe it's needed for some other syllable if cw > beta
											r = np.random.randint(0, m*n)
											pos = r%len(list_active)			# randomly pick an active cell
											Dnew[i][j][k][l]["x"] = list_active[pos][0]		# replace existing synapse with synapse to active cell
											Dnew[i][j][k][l]["y"] = list_active[pos][1]
											Dnew[i][j][k][l]["cw"] = beta + np.random.random()*(1-beta)	# synPerm	# to ensure it has a chance to become a strong connection
											list_active.pop(pos)
									if len(list_active) == 0:
										break

	# ---- Step 2: Computing cell states ---- #

			active_cells = []

			for j in np.ndindex(n):
				A['t'][j] = 0
				P['t'][j] = 0
				if W[j] == 1:
					if P['t-1'][j] != 0:
						A['t'][j] = P['t-1'][j]				# cell activated if present in winning column and if predicted previously
					else:
						A['t'][j] = pow(2,m)-1				# cell activated if present in winning column and if no cell in the column had been predicted
					
					for i in range(m):
						if (A['t'][j] & (1 << i)) != 0:
							active_cells.append([i,j])

		    	for j in range(n):												# to negatively reinforce active segments of inactive cells
				if A['t'][j] == 0:
					for i, k in np.ndindex(m, d):
						if isSegmentActive(i, j, k, D, A['t-1']) == True:
							decay(i, j, k, D, Dnew)

			for i,j in np.ndindex(m,n):						# computing predictive state for this time step
			    for k in np.ndindex(d): 
			    	if isSegmentActive(i, j, k, Dnew, A['t']) == True:
			    		P['t'][j] = P['t'][j] | 1<<i
			    		break
			
			np.copyto(D, Dnew) # D = Dnew

			
			out = ""
			outputW = A['t'] > 0
				
			for syll in seq:
				if ((outputW & S[syll]) == S[syll]).all():
					out = out + syll

			pred = ""
			predW = P['t-1'] > 0
				
			for syll in seq:
				if ((predW & S[syll]) == S[syll]).all():
					pred = pred + syll

			if (ntrials+1)%100==0:		
				print "trial: #", ntrials
				print "input syllable: ", syllable
				print W
				print "predicted: ", pred
				if syllable == "B":
					print P['t-1']
				print "output syllable: ", out
				if syllable == "B":
					print A['t']
	
			P['t-1'] = 0
			A['t-1'] = 0
			P['t-1'] = P['t']
			A['t-1'] = A['t']
			A['t'] = 0
			P['t'] = 0
			
		# reset on encountering "end" syllable
		P['t-1'] = np.zeros((n), dtype = [("t", int), ("t-1", int)])	
		A['t-1'] = np.zeros((n), dtype = [("t", int), ("t-1", int)])
		active_cells = []
		



print "Testing"

test1 = ["A", "B", "C"]
test2 = ["X", "B", "C"]
tests = [test1, test2]

for test in tests:
	print "New test: ", test
	for syllable in test:
		W = S[syllable]									# marks winning columns
		A['t'] = 0
		P['t'] = 0
		active_cells = []
		for j in np.ndindex(n):
			A['t'][j] = 0
			P['t'][j] = 0
			if W[j] == 1:
				if P['t-1'][j] != 0:
					A['t'][j] = P['t-1'][j]			# cell activated if present in winning column and if predicted previously
				else:
					A['t'][j] = pow(2,m)-1				# cell activated if present in winning column and if no cell in the column had been predicted
				
				for i in range(m):
					if (A['t'][j] & (1 << i)) != 0:
						active_cells.insert(0, (i,j))
	    		
		for i,j in np.ndindex(m,n):						# computing predictive state for this time step
		    for k in np.ndindex(d):
		    	if isSegmentActive(i, j, k, D, A['t']) == True:
		    		P['t'][j] = P['t'][j] | 1<<i
		    		break

		out = ""
		outputW = A['t'] > 0
			
		for syll in seq:
			if ((outputW & S[syll]) == S[syll]).all():
				out = out + syll

		pred = ""
		predW = P['t'] > 0
			
		for syll in seq:
			if ((predW & S[syll]) == S[syll]).all():
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
		P['t'] = 0
	
	P['t-1'] = np.zeros((n), dtype = [("t", int), ("t-1", int)])	
	A['t-1'] = np.zeros((n), dtype = [("t", int), ("t-1", int)])
		
print "ERR"
