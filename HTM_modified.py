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
nTrainingTrials = 100

# rSeed = np.random.randint(0,1e7)
rSeed = 0
np.random.seed(rSeed)

# ---- Step 1: Initialisation ---- #
A = np.zeros((n), dtype = [("t", int), ("t-1", int)])
P = np.zeros((n), dtype = [("t", int), ("t-1", int)])
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
	D[i][j][k][l]["cw"] = cw		#	synPerm

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

W = np.zeros((n), dtype = int)								# feed-forward input representing a syllable

seq1 = ["A", "B", "C", "D", "E", "F", "G", "end"]
seq2 = ["X", "B", "C", "D", "E", "F", "Y", "end"]

# seq1 = ["A", "B", "C", "D", "E", "F", "end"]
# seq2 = ["X", "B", "C", "D", "E", "Y", "end"]

#seq1 = ["A", "B", "C", "D", "E", "end"]
#seq2 = ["X", "B", "C", "D", "Y", "end"]

# seq1 = ["A", "B", "C", "D", "end"]
# seq2 = ["X", "B", "C", "Y", "end"]

sequences = [seq1, seq2]									# list of sequences to be trained on

def isSegmentActive(i, j, k, D, Mat):						# if no. of connected synapses to active cells > theta in any distal segment of current cell
	segment = D[i][j][k]
	connectedSynapses = segment["cw"] > beta
	synapsesToActiveCells = Mat[segment["y"]] & (1 << segment["x"])
	connectedToActive = connectedSynapses & (synapsesToActiveCells != 0)
	count = connectedToActive.sum()
	# for l in np.ndindex(s):
 	#	synapse = D[i][j][k][l]
	# 	if synapse["cw"] > beta and ((Mat[synapse["y"]] & (1<<synapse["x"]))!=0):
	# 		count = count + 1
	return count >= theta

def count_positive_synapses(Mat, Segment):
	positiveSynapses = Segment["cw"] > 0.0
	synapsesToActiveCells = Mat[Segment["y"]] & (1 << Segment["x"])
	positiveToActive = positiveSynapses & (synapsesToActiveCells != 0)
	count = positiveToActive.sum()
	#count = 0
	# for l in np.ndindex(s):
	# 	synapse = Segment[l]
	# 	if synapse["cw"] > 0 and (Mat[synapse["y"]] & (1<<synapse["x"]))!=0:
	# 		count = count + 1
	return count

def reinforce(x, y, z, D, Dnew):
	segmentOld = D[x][y][z]
	segmentNew = Dnew[x][y][z]
	deltaPos = ((A['t-1'][segmentOld["y"]]) & (1<<segmentOld["x"])) != 0
	deltaNeg = ~deltaPos
	deltaPos = pPos * deltaPos
	deltaNeg = pNeg * deltaNeg
	segmentNew["cw"] = segmentOld["cw"] + deltaPos - deltaNeg
	tempA = segmentNew["cw"] > 1.0
	tempB = (~tempA) * segmentNew["cw"]
	segmentNew["cw"] = tempA + tempB
	tempA = segmentNew["cw"] > 0.0
	segmentNew["cw"] = tempA * segmentNew["cw"]

	# for l in np.ndindex(s):
	# 	synapseOld = D[x][y][z][l]
	# 	synapseNew = Dnew[x][y][z][l]
	# 	delta = 0.0
	# 	if ((A['t-1'][synapseOld["y"]]) & (1<<synapseOld["x"])) != 0:
	# 		delta = delta + pPos							# reinforcing synapses to active cells
	# 	else:
	# 		delta = delta - pNeg							# negatively reinforcing synapses to inactive cells
	# 	synapseNew["cw"] = synapseOld["cw"] + delta
	# 	if synapseNew["cw"] < 0.0:
	# 		synapseNew["cw"] = 0.0
	# 	if synapseNew["cw"] > 1.0:
	# 		synapseNew["cw"] = 1.0

	# 	Dnew[x][y][z][l]["x"] = synapseOld["x"]
	# 	Dnew[x][y][z][l]["y"] = synapseOld["y"]
	# 	Dnew[x][y][z][l]["cw"] = synapseNew["cw"]



def decay(x, y, z, D, Dnew):
	segmentOld = D[x][y][z]
	segmentNew = Dnew[x][y][z]
	segmentNew["cw"] = segmentOld["cw"] - pDec
	tempA = segmentNew["cw"] > 1.0
	tempB = (~tempA) * segmentNew["cw"]
	segmentNew["cw"] = tempA + tempB
	tempA = segmentNew["cw"] > 0.0
	segmentNew["cw"] = tempA * segmentNew["cw"]

	# for l in np.ndindex(s):
	# 	synapseOld = D[x][y][z][l]
	# 	synapseNew = Dnew[x][y][z][l]
	# 	synapseNew["cw"] = synapseOld["cw"] - pDec
	# 	if synapseNew["cw"] < 0.0:
	# 		synapseNew["cw"] = 0.0
	# 	if synapseNew["cw"] > 1.0:
	# 		synapseNew["cw"] = 1.0
	# 	Dnew[x][y][z][l]["x"] = synapseOld["x"]
	# 	Dnew[x][y][z][l]["y"] = synapseOld["y"]
	# 	Dnew[x][y][z][l]["cw"] = synapseNew["cw"]
		

	
active_cells = []
training_results = []

flagAltSeq = 0												# 0 -> Alternating; 1-> Sequential

for ntrials in range(nTrainingTrials):
	for seq in sequences:
		current_result = []
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
				    			for k in np.ndindex(d):
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
					        		list_active = list(active_cells)					# NOTE: was list(active_cells) before for no reason
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

			Atemp1 = P['t-1'] * W
			Atemp2 = ((Atemp1 != 0) != W) * (pow(2,m)-1)
			A['t'] = Atemp1 + Atemp2

			for j in np.ndindex(n):
				if W[j] == 1:
					for i in range(m):
						if (A['t'][j] & (1 << i)) != 0:
							active_cells.append([i,j])
				else:
					for i, k in np.ndindex(m, d):
						if isSegmentActive(i, j, k, D, A['t-1']) == True:
							decay(i, j, k, D, Dnew)



			# for j in np.ndindex(n):
			# 	A['t'][j] = 0
			# 	P['t'][j] = 0
			# 	if W[j] == 1:
			# 		if P['t-1'][j] != 0:
			# 			A['t'][j] = P['t-1'][j]				# cell activated if present in winning column and if predicted previously
			# 		else:
			# 			A['t'][j] = pow(2,m)-1				# cell activated if present in winning column and if no cell in the column had been predicted
					
			# 		for i in range(m):
			# 			if (A['t'][j] & (1 << i)) != 0:
			# 				active_cells.append([i,j])

		  #   for j in np.ndindex(n):												# to negatively reinforce active segments of inactive cells
				# if A['t'][j] == 0:
				# 	for i, k in np.ndindex(m, d):
				# 		if isSegmentActive(i, j, k, D, A['t-1']) == True:
				# 			decay(i, j, k, D, Dnew)

			for i,j in np.ndindex(m,n):						# computing predictive state for this time step
				for k in np.ndindex(d):
					if isSegmentActive(i, j, k, D, A['t']) == True:
						P['t'][j] = P['t'][j] | 1<<i
			   			break
			
			np.copyto(D, Dnew) # D[...] = Dnew

			
			out = ""
			outputW = A['t'] > 0
				
			for syll in S:
				if ((outputW & S[syll]) == S[syll]).all():
					out = out + syll

			pred = ""
			predW = P['t-1'] > 0
				
			for syll in S:
				if ((predW & S[syll]) == S[syll]).all():
					pred = pred + syll

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
		# P['t-1'] = np.zeros((n), dtype = [("t", int), ("t-1", int)])	
		# A['t-1'] = np.zeros((n), dtype = [("t", int), ("t-1", int)])
		P['t-1'] = 0
		A['t-1'] = 0
		active_cells = []

		if ntrials == nTrainingTrials - 1:
			training_results.append({
				"Training Result": current_result 
			})

print "Testing"

test1 = ["A", "B", "C", "D"]
test2 = ["X", "B", "C", "D"]
tests = [test1, test2]

testing_results = []
for test in tests:
	current_result = []
	for syllable in test:
		W = S[syllable]										# marks winning columns
		A['t'] = 0
		P['t'] = 0

		Atemp1 = P['t-1'] * W
		Atemp2 = ((Atemp1 != 0) != W) * (pow(2,m)-1)
		A['t'] = Atemp1 + Atemp2

		# for j in np.ndindex(n):
		# 	# A['t'][j] = 0
		# 	# P['t'][j] = 0
		# 	if W[j] == 1:
		# 		if P['t-1'][j] != 0:
		# 			A['t'][j] = P['t-1'][j]					# cell activated if present in winning column and if predicted previously
		# 		else:
		# 			A['t'][j] = pow(2,m)-1					# cell activated if present in winning column and if no cell in the column had been predicted
					    		
		for i,j in np.ndindex(m,n):						# computing predictive state for this time step
			for k in np.ndindex(d):
				if isSegmentActive(i, j, k, D, A['t']) == True:
					P['t'][j] = P['t'][j] | 1<<i
		   			break

		out = ""
		outputW = A['t'] > 0
			
		for syll in S:
			if ((outputW & S[syll]) == S[syll]).all():
				out = out + syll

		pred = ""
		predW = P['t'] > 0
		
		for syll in S:
			if ((predW & S[syll]) == S[syll]).all():
				pred = pred+syll
		
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
	
	# P['t-1'] = np.zeros((n), dtype = [("t", int), ("t-1", int)])	
	# A['t-1'] = np.zeros((n), dtype = [("t", int), ("t-1", int)])
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

with open('Run12.json', 'w') as outfile:  
    json.dump(Data, outfile, sort_keys=True, indent=4)
