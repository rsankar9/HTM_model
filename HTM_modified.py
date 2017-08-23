import recipy

# coding: utf-8
import json
import numpy as np


n=21						# no. of columns in a layer
m=6							# no. of cells per column
d=10						# no. of distal segments per cell
s=16						# no. of potential synapses per segment

beta = 0.5					# synaptic connection threshold
theta = 3					# segment activation threshold
synPerm = 0.21				# initial synaptic permanence
nStepReplace = 20

pPos = 0.1					# long term potentiation
pNeg = 0.1					# long term depression
pDec = 0.01					# constant decay
nTrainingTrials = 50

rSeed = np.random.randint(0,1e7)
#rSeed = 0
np.random.seed(rSeed)

# ---- Step 1: Initialisation ---- #
A = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])
P = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])
D = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])
Dnew = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])

for i, j, k, l in np.ndindex(m, n, d, s):								# initialising each distal segment with synapses to random cells with random strength
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
	D[i][j][k][l]["cw"] = cw											#	synPerm

S = { "A" : np.array([0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
      "B" : np.array([0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0]),
      "C" : np.array([0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0]), 
      "D" : np.array([0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0]), 
      "E" : np.array([1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0]), 
      "F" : np.array([0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0]), 
      "G" : np.array([0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1]), 
      "X" : np.array([0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0]), 
      "Y" : np.array([1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1]), 
      "end" : np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0])  }	# encoding for each syllable

W = np.zeros((n), dtype = int)

# seq1 = ["A", "B", "C", "D", "E", "F", "G", "end"]
# seq2 = ["X", "B", "C", "D", "E", "F", "Y", "end"]

# seq1 = ["A", "B", "C", "D", "E", "F", "end"]
# seq2 = ["X", "B", "C", "D", "E", "Y", "end"]

#seq1 = ["A", "B", "C", "D", "E", "end"]
#seq2 = ["X", "B", "C", "D", "Y", "end"]

seq1 = ["A", "B", "C", "D", "end"]
seq2 = ["X", "B", "C", "Y", "end"]

sequences = [seq1, seq2]									# list of sequences to be trained on

def isSegmentActive(ActMat, Segment):						# if no. of connected synapses to active cells > theta in any distal segment of current cell
	count = 0
 	for syn in Segment:
 		if syn["cw"] > beta and ActMat[syn["x"], syn["y"]] == 1:
 			count = count + 1
	return count >= theta

def count_positive_synapses(ActMat, Segment):
	count = 0
	for syn in Segment:
		if syn["cw"] > 0 and ActMat[syn["x"], syn["y"]] == 1:
			count = count + 1
	return count

def reinforce(x, y, z, D, Dnew):    
	for l in np.ndindex(s):
		synapseOld = D[x][y][z][l]
    	synapseNew = Dnew[x][y][z][l]
    	delta = 0.0
    	if A['t-1'][synapseOld["x"]][synapseOld["y"]] == 1:
    		delta = delta + pPos
    	else:
    		delta = delta - pNeg
 		synapseNew["cw"] = synapseOld["cw"] + delta
 		if synapseNew["cw"] < 0.0:
 			synapseNew["cw"] = 0.0
 		if synapseNew["cw"] > 1.0:
 			syanpseNew["cw"] = 1.0

def decay(x, y, z, D, Dnew):
    delta = 0
    for l in np.ndindex(s):
    	syn = D[x][y][z][l]
    	Dnew[x][y][z][l]["x"] = syn["x"]
    	Dnew[x][y][z][l]["y"] = syn["y"]
    	Dnew[x][y][z][l]["cw"] = syn["cw"] - pDec



active_cells = []
training_results = []

flagAltSeq = 1												# 0 -> Alternating; 1-> Sequential

for seq in sequences:
	for ntrials in range(nTrainingTrials):
		current_result = []
		for syllable in seq:
			W = S[syllable]									# marks winning columns
			A['t'] = 0
			P['t'] = 0

			# ---- Step 3: Learning ---- #
			Dnew = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])
			Dnew[...] = D

			for j in np.ndindex(n):							# active dendritic segment reinforced if cell predicted correctly
			    if W[j] == 1:
			    	for i in np.ndindex(m):
				    	if P['t-1'][i][j] != 0:
					    			for k in np.ndindex(d):
						    			if isSegmentActive(A['t-1'], D[i][j][k]) == True:	#count_ones_in_product_Dcon(A['t-1'], D[i][j][k]) > theta
					        				reinforce(i, j, k, D, Dnew)

			# for i, j, k in np.ndindex(m, n, d):				# active dendritic segment reinforced if cell predicted correctly
			#     if W[j] == 1:
			#     	if P['t-1'][i][j] > 0 and count_ones_in_product_Dcon(A['t-1'], D[i][j][k]) > theta:
			#         #if P['t-1'][i][j] > 0 and sum(sum(Dcon[i][j][k] * A['t-1'])) > theta:
			#         	reinforce(i, j, k, D, Dnew)

			for j in np.ndindex(n):						
			    if W[j] == 1:
			    	tempSum = 0
			    	for i in np.ndindex(m):				# to check if current winning column was predicted
			    		tempSum += P['t-1'][i][j]
			    	if tempSum == 0:					# if current winning column not predicted
			    		maxSum = 0
			    		maxRow = np.random.random_integers(0, m-1)
			    		maxSeg = np.random.random_integers(0, d-1)
				    	for i, k in np.ndindex(m, d):	# to find distal segment in current column closest to activation
				    		#currSum = count_ones_in_product_Dpos(A['t-1'], D[i][j][k])
				    		currSum = count_positive_synapses(A['t-1'], D[i][j][k])
				    	#	currSum = sum(sum(Dpos[i][j][k] * A['t-1']))
				    		if maxSum < currSum:
					    		maxSum = currSum
					    		maxRow = i
					    		maxSeg = k
					    	reinforce(maxRow, j, maxSeg, D, Dnew)

						if (ntrials % nStepReplace) == 0:
							for i, k in np.ndindex(m, d):
								if i!=maxRow or k!=maxSeg:						# no match found, hence updating synapses
									list_active = list(active_cells)
									np.random.shuffle(list_active)
									for l in np.ndindex(s):
										if len(list_active)==0:
											break
										if D[i][j][k][l]["cw"] < beta:			# maybe it's needed for some other syllable
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
				if W[j] == 1:
					found_active = 0
					for i in np.ndindex(m):
						if P['t-1'][i][j] == 1:
							A['t'][i][j] = 1					# cell activated if present in winning column and if predicted previously
							found_active = 1
							active_cells.append([i,j])
					if found_active==0:
						for i in np.ndindex(m):
							A['t'][i][j] = 1				# cell activated if present in winning column and if no cell in the column had been predicted
							active_cells.append([i,j])
		    	else:
				    for i, k in np.ndindex(m, d):			# to negatively reinforce active segments of inactive cells
				    	if isSegmentActive(A['t-1'], D[i][j][k]) == True:
						#if count_ones_in_product_Dcon(A['t-1'], D[i][j][k]) > theta:
					#	if A['t'][i][j] == 0 and sum(sum(Dcon[i][j][k] * A['t-1'])) > theta:
							decay(i, j, k, D, Dnew)

			for i,j in np.ndindex(m,n):					# computing predictive state for this time step
			    for k in np.ndindex(d):
			    	if isSegmentActive(A['t'], D[i][j][k]) == True:
			    	#if count_ones_in_product_Dcon(A['t'], D[i][j][k]) > theta:
			        #if sum(sum(A['t']*Dcon[i][j][q])) > theta:
			            P['t'][i][j] = 1					# if no. of connected synapses to active cells > theta in any distal segment of current cell
			            break
			            
			D[...] = Dnew

	
			out = ""
			outputW = np.zeros(n, dtype=int)
			for j in np.ndindex(n):
				column_active = 0
				for i in np.ndindex(m):
					if A['t'][i][j]==1:
						column_active = 1
						outputW[j] = 1
						break
				
			for syll in S:
				flag = 0
				for j in np.ndindex(n):
					if (S[syll][j] == 1) and (outputW[j] != 1):
						flag = 1
						break
				if flag == 0:
					out = out + syll
					

			pred = ""
			predW = np.zeros(n, dtype=int)
			for j in np.ndindex(n):
				column_active = 0
				for i in np.ndindex(m):
					if P['t-1'][i][j]==1:
						column_active = 1
						predW[j] = 1
						break
				
			for syll in S:
				flag = 0
				for j in np.ndindex(n):
					if S[syll][j] == 1 and predW[j]!=1:
						flag = 1
						break
				if flag == 0:
					pred = pred + syll

			# if (ntrials+1)%100==0:		
			# 	print "trial: #", ntrials
			# 	print "input syllable: ", syllable
			# 	print W
			# 	print "predicted: ", pred
			# 	print P['t-1']
			# 	print "output syllable: ", out
			# 	print A['t']

			if ntrials == nTrainingTrials - 1:		
				syllable_result = {}
				syllable_result["prediction"] = pred
				syllable_result["P['t-1']"] = str(P['t-1'])
				syllable_result["output"] = out
				syllable_result["A['t']"] = str(A['t'])
				current_result.append(syllable_result)
	

			P['t-1'] = 0
			A['t-1'] = 0
			P['t-1'] = P['t']
			A['t-1'] = A['t']
			A['t'] = 0
			P['t'] = 0

		# reset on encountering "end" syllable
		# P['t-1'] = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])	
		# A['t-1'] = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])
		P['t-1'] = 0
		A['t-1'] = 0
		active_cells = []

		if ntrials == nTrainingTrials - 1:
			training_results.append({
				"Training Result": current_result 
			})

print "Testing"

test1 = ["A", "B", "C"]
test2 = ["X", "B", "C"]
tests = [test1, test2]

testing_results = []
for test in tests:
	current_result = []
	for syllable in test:
		W = S[syllable]									# marks winning columns
		A['t'] = 0
		P['t'] = 0

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
	    		
		for i, j in np.ndindex(m, n):					# computing predictive state for this time step
		    for k in np.ndindex(d):
		    	if isSegmentActive(A['t'], D[i][j][k]) == True:
		    	#if count_ones_in_product_Dcon(A['t'], D[i][j][k]) > theta:
		        #if sum(sum(A['t']*Dcon[i][j][q])) > theta:
		            P['t'][i][j] = 1					# if no. of connected synapses to active cells > theta in any distal segment of current cell
		            break

		out = ""
		outputW = np.zeros(n, dtype=int)
		for j in np.ndindex(n):
			column_active = 0
			for i in np.ndindex(m):
				if A['t'][i][j]==1:
					column_active = 1
					outputW[j] = 1
					break
			
		for syll in S:
			flag = 0
			for j in np.ndindex(n):
				if (S[syll][j] == 1) and (outputW[j] != 1):
					flag = 1
					break
			if flag == 0:
				out = out + syll
				

		pred = ""
		predW = np.zeros(n, dtype=int)
		for j in np.ndindex(n):
			column_active = 0
			for i in np.ndindex(m):
				if P['t-1'][i][j]==1:
					column_active = 1
					predW[j] = 1
					break
			
		for syll in S:
			flag = 0
			for j in np.ndindex(n):
				if S[syll][j] == 1 and predW[j]!=1:
					flag = 1
					break
			if flag == 0:
				pred = pred + syll

		syllable_result = {}
		syllable_result["prediction"] = pred
		syllable_result["P['t']"] = str(P['t'])
		syllable_result["output"] = out
		syllable_result["A['t']"] = str(A['t'])
		current_result.append(syllable_result)

		P['t-1'] = 0
		A['t-1'] = 0
		P['t-1'] = P['t']
		A['t-1'] = A['t']
		A['t'] = 0
		P['t'] = 0

	P['t-1'] = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])	
	A['t-1'] = np.zeros((m, n), dtype = [("t", int), ("t-1", int)])
	P['t-1'] = 0
	A['t-1'] = 0
	testing_results.append({
		"Test Result": current_result 
	})
			            
print "Writing to json file"
layer_parameters = {
	"no. of columns in a layer [n]": n,
	"no. of cells per column [m]": m,
	"no. of distal segments per cell [d]": d,
	"no. of potential synapses per segment [s]": s,
}
synapse_parameters = {
	"synapse connectivity threshold [beta]": beta,
	"segment activation threshold [theta]": theta,
	"synaptic permanence [synPerm]": synPerm,
	"synapses replaced after steps [nStepReplace]": nStepReplace,
}

learning_parameters = {
	"long term potentiation [pPos]": pPos,
	"long term depression [pNeg]": pNeg,
	"decay [pDec]": pDec,
	"no. of trials in training [nTrainingTrials]": nTrainingTrials,
	"alternating/sequential": flagAltSeq
}

random_parameters = {
	"seed [rSeed]": rSeed
}

input_parameters = {
	"Layer Parameters": layer_parameters,
	"Synapse Parameters": synapse_parameters,
	"Learning Parameters": learning_parameters,
	"Random Parameters": random_parameters,
}

results = {
	"Training Results": training_results,
	"Testing Results": testing_results
}

Data = {
	"Input": input_parameters,
	"Results": results
}

with open('bakwas.json', 'w') as outfile:  
    json.dump(Data, outfile, sort_keys=True, indent=4)
