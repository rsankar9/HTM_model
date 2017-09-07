#import recipy

# coding: utf-8
import json
import numpy as np

def HTM(resFile):
	n=21						# no. of columns in a layer
	m=6							# no. of cells per column
	d=5						# no. of distal segments per cell

	beta = 0.5					# synaptic connection threshold
	theta = 3					# segment activation threshold
	synPerm = 0.21				# initial synaptic permanence

	pPos = 0.2					# long term potentiation
	pNeg = 0.2					# long term depression
	pDec = 0.02					# constant decay
	nTrainingTrials = 3
	nRepConsecutive = 10

	initialisingLimit = 0.5
	chooseMax = 0				# 1 -> on 0 -> off
	maxCondition = 1			# 0 -> count connected synapses 1-> find closest to activation by summation

	rSeed = np.random.randint(0,1e7)
	#rSeed = 0					# 2227572
	np.random.seed(rSeed)

	# ---- Step 1: Initialisation ---- #
	A = np.zeros((n), dtype = [("t", int), ("t-1", int)])
	P = np.zeros((n), dtype = [("t", int), ("t-1", int)])
	D = np.zeros((m, n, d, m, n), dtype = float)
	Dnew = np.zeros((m, n, d, m, n), dtype = float)

	for i, j, k, p, q in np.ndindex(m, n, d, m, n):							# initialising each distal segment with synapses to random cells with random strength
		if i!=p or j!=q:
			D[i][j][k][p][q] = np.random.uniform(0, initialisingLimit)								#	synPerm

	S = { "A" : np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
	      "B" : np.array([0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
	      "C" : np.array([0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]), 
	      "D" : np.array([0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0]), 
	      "E" : np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0]), 
	      "F" : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0]), 
	      "G" : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]),	}	# encoding for each syllable

	# seq1 = ["A", "B", "C", "D", "E"]
	# seq2 = ["X", "B", "C", "D", "Y", "end"]

	seq1 = ["A", "B", "C", "G", "E"]
	seq2 = ["D", "B", "C", "G", "F"]
	# seq3 = ["X", "Y", "end"]

	# seq1 = ["A", "B", "C", "D"]
	# seq2 = ["X", "B", "C", "Y"]

	sequences = [seq1, seq2]								# list of sequences to be trained on

	test1 = ["A", "B", "C", "G"]
	test2 = ["D", "B", "C", "G"]
	# test3 = ["X", "Y"]
	# tests = [test1, test2]
	#test2 = ["X", "B", "C", "Y"]
	#test3 = ["E", "B", "C", "F"]
	#test4 = ["X", "B", "F", "D"]

	tests = [test1, test2]										# list of sequences to be tested on

	W = np.zeros((n), dtype = int)								# feed-forward input representing a syllable
									
	def isSegmentActive(i, j, k, D, Mat):						# if no. of connected synapses to active cells > theta in any distal segment of current cell
		segment = D[i][j][k]
		count = 0
		for x, y in np.ndindex(m, n):
			if (Mat[y] & (1 << x) != 0) and (segment[x][y]>beta):
				count = count + 1
				if count >= theta:
					return True
		return False

	def count_connected_synapses(Mat, Segment):
		count = 0.0
		for x, y in np.ndindex(m, n):
			if (Mat[y] & (1 << x) != 0) and (Segment[x][y]>beta):
				count = count + 1
		return count

	def closest_to_connected_synapses(Mat, Segment):
		count = 0.0
		for x, y in np.ndindex(m, n):
			if (Mat[y] & (1 << x) != 0) and (Segment[x][y]>0.0):
				count = count + Segment[x][y]
		return count

	def reinforce(i, j, k, D, Dnew):
		Mat = A['t-1']
		segmentOld = D[i][j][k]
		segmentNew = Dnew[i][j][k]
		for x, y in np.ndindex(m, n):
			delta = 0.0
			if (Mat[y] & (1 << x) != 0): #and segmentOld[x][y] != 0:
				delta = delta + pPos
			else:
				delta = delta - pNeg
			segmentNew[x][y] = segmentOld[x][y] + delta
			segmentNew[x][y] = min(1, segmentNew[x][y])
			segmentNew[x][y] = max(0, segmentNew[x][y])
		# print segmentOld, segmentNew
		Dnew[i][j][k] = segmentNew

	def decay(x, y, z, D, Dnew):
		segmentOld = D[x][y][z]
		segmentNew = Dnew[x][y][z]
		segmentNew = segmentOld - pDec
		for i, j in np.ndindex(m, n):
			segmentNew[i][j] = max(0, segmentNew[i][j])
		Dnew[x][y][z] = segmentNew

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
			
	def repr_human(Mat):
		text = ""
		for i in range(m):
			row = ""
			for j in range(n):
				if Mat[j] & (1<<i) != 0:
					row += "+"									#	+ -> Active
				else:	
					row += "-"									#	- -> Inactive
			text += row
			text += '\n'
		return text
	
	active_cells = []
	training_results = []

	flagAltSeq = 2												# 0 -> Alternating; 1 -> Sequential; 2 -> Few each

	for ntrials in range(nTrainingTrials):
		for seq in sequences:
			for nr in range(nRepConsecutive):
				current_result = []
				for syllable in seq:
					W = S[syllable]									# marks winning columns
					A['t'] = 0
					P['t'] = 0

					# ---- Step 3: Learning ---- #
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
								maxCloseness = 0.0
								maxRow = np.random.random_integers(0, m-1)
								maxSeg = np.random.random_integers(0, d-1)
								if chooseMax == 1:
									for i, k in np.ndindex(m, d):		# to find distal segment in current column closest to activation
										if maxCondition == 0:
											currCloseness = count_connected_synapses(A['t-1'], D[i][j][k])
										else:
											currCloseness = closest_to_connected_synapses(A['t-1'], D[i][j][k])
										if maxCloseness < currCloseness:
											maxCloseness = currCloseness
											maxRow = i
											maxSeg = k
								# print "Here ", seq, syllable, ntrials*nRepConsecutive+nr, maxRow, j, maxSeg, maxCloseness
								
								reinforce(maxRow, j, maxSeg, D, Dnew)
								

			# ---- Step 2: Computing cell states ---- #

					Atemp1 = P['t-1'] * W
					Atemp2 = ((Atemp1 != 0) != W) * (pow(2,m)-1)
					A['t'] = Atemp1 + Atemp2

					for i,j in np.ndindex(m,n):
						if (A['t'][j] & 1<<i) == 0:
							for k in np.ndindex(d):
								if isSegmentActive(i, j, k, D, A['t-1']) == True:
									decay(i, j, k, D, Dnew)


					for i,j in np.ndindex(m,n):						# computing predictive state for this time step
						for k in np.ndindex(d):
							if isSegmentActive(i, j, k, Dnew, A['t']) == True:
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

					if nr == nRepConsecutive - 1:	
						syllable_result = {}
						syllable_result["prediction"] = pred
						syllable_result["P['t-1']"] = repr_human(P['t-1'])
						syllable_result["output"] = out
						syllable_result["A['t']"] = repr_human(A['t'])
						current_result.append(syllable_result)

					P['t-1'] = 0
					A['t-1'] = 0
					P['t-1'] = P['t']
					A['t-1'] = A['t']
					A['t'] = 0
					P['t'] = 0
					
				# reset on encountering "end" syllable
				P['t-1'] = 0
				A['t-1'] = 0
				active_cells = []

			if ntrials == nTrainingTrials - 1:
				training_results.append({
					"Training Result": current_result 
				})


	print "Testing"
	
	testing_results = []
	# predictions = []
	# got_1_flag = False
	# got_2_flag = False
	# got_both_flag = False

	for test in tests:
		current_result = []
		seq_predicted = [test[0]]
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
			syllable_result["P['t']"] = repr_human(P['t'])
			syllable_result["output"] = out
			syllable_result["A['t']"] = repr_human(A['t'])
			current_result.append(syllable_result)
			seq_predicted.append(pred)

			P['t-1'] = 0
			A['t-1'] = 0
			P['t-1'] = P['t']
			A['t-1'] = A['t']
			A['t'] = 0
			P['t'] = 0
		
		P['t-1'] = 0
		A['t-1'] = 0
		testing_results.append({
			"Test Result": current_result 
		})
	# 	predictions.append(seq_predicted)
	# if predictions[0] == sequences[0]:
	# 	got_1_flag = True
	# if predictions[1] == sequences[1]:
	# 	got_2_flag = True
	# if predictions == sequences:
	# 	got_both_flag = True

	print "Writing to json file: ", resFile
	layer_parameters = {
		"no. of columns in a layer [n]": n,
		"no. of cells per column [m]": m,
		"no. of distal segments per cell [d]": d,
	#	"no. of potential synapses per segment [s]": s,
	}
	synapse_parameters = {
		"synapse connectivity threshold [beta]": beta,
		"segment activation threshold [theta]": theta,
		"synaptic permanence [synPerm]": synPerm,
		"synapses initialised with connectivity weight upto": initialisingLimit 
	#	"synapses replaced after steps [nStepReplace]": nStepReplace,
	}

	learning_parameters = {
		"long term potentiation [pPos]": pPos,
		"long term depression [pNeg]": pNeg,
		"decay [pDec]": pDec,
		"no. of trials in training [nTrainingTrials]": nTrainingTrials,
		"no. of consecutive runs in each trial": nRepConsecutive,
		"alternating/sequential": flagAltSeq,
		"choose the segment with maximum match": chooseMax
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

	with open(resFile, 'w') as outfile:  
	    json.dump(Data, outfile, sort_keys=True, indent=4, separators=(',', ':\t'))
	# return got_1_flag, got_2_flag, got_both_flag
	return
