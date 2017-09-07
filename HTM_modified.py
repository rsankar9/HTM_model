#import recipy
# coding: utf-8

import json
import numpy as np

def HTM(resFile):
    n=21                        # no. of columns in a layer
    m=6                         # no. of cells per column
    d=1                     # no. of distal segments per cell
    s=7                        # no. of potential synapses per segment

    beta = 0.5                  # synaptic connection threshold
    theta = 3                   # segment activation threshold
    synPerm = 0.21              # initial synaptic permanence
    nStepReplace = 0#10

    pPos = 0.2                  # long term potentiation
    pNeg = 0.2                  # long term depression
    pDec = 0.02                 # constant decay
    nTrainingTrials = 5
    nRepConsecutive = 10

    initialisingLimit = beta
    chooseMax = 1               # 1 -> on 0 -> off
    maxCondition = 2            # 0 -> count connected synapses 1 -> find closest to activation by summation 2 -> count positive synapses
    replaceSynapses = 1         # 0 -> off 1 -> on

    rSeed = np.random.randint(0,1e7)
    #rSeed = 0                  # 2227572
    np.random.seed(rSeed)

    # ---- Step 1: Initialisation ---- #
    A = np.zeros((n), dtype = [("t", int), ("t-1", int)])
    P = np.zeros((n), dtype = [("t", int), ("t-1", int)])
    D = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])
    Dnew = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])

    for i, j, k, l in np.ndindex(m, n, d, s):                               # initialising each distal segment with synapses to random cells with random strength
        x = np.random.random_integers(0, m-1)
        y = np.random.random_integers(0, n-1)
        cw = np.random.random()
        while x == i and y==j:
            x = np.random.random_integers(0, m-1)
            y = np.random.random_integers(0, n-1)
        while cw == 0:
            #cw = np.random.random() 
            cw = np.random.uniform(0.0,initialisingLimit)
        D[i][j][k][l]["x"] = x
        D[i][j][k][l]["y"] = y
        D[i][j][k][l]["cw"] = cw                                            #   synPerm
        
    S = { "A" : np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
          "B" : np.array([0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
          "C" : np.array([0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]), 
          "D" : np.array([0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0]), 
          "E" : np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0]), 
          "F" : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0]), 
          "G" : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]),  }   # encoding for each syllable

    # seq1 = ["A", "B", "C", "D", "E"]
    # seq2 = ["X", "B", "C", "D", "Y", "end"]

    #seq1 = ["A", "B", "end"]
    #seq2 = ["X", "B", "C", "Y"]
    #seq3 = ["E", "B", "C", "F"]

    seq1 = ["A", "B", "C", "G", "E"]
    seq2 = ["D", "B", "C", "G", "F"]

    sequences = [seq1, seq2]                                # list of sequences to be trained on

    test1 = ["A", "B", "C", "G"]
    test2 = ["D", "B", "C", "G"]

    
    #test2 = ["X", "B", "C", "Y"]
    #test3 = ["E", "B", "C", "F"]
    #test4 = ["X", "B", "F", "D"]

    tests = [test1, test2]                                      # list of sequences to be tested on
    # tests = testingSequences

    W = np.zeros((n), dtype = int)                              # feed-forward input representing a syllable
                                    
    def isSegmentActive(i, j, k, D, Mat):                       # if no. of connected synapses to active cells > theta in any distal segment of current cell
        segment = D[i][j][k]
        connectedSynapses = segment["cw"] > beta
        synapsesToActiveCells = Mat[segment["y"]] & (1 << segment["x"])
        connectedToActive = connectedSynapses & (synapsesToActiveCells != 0)
        count = connectedToActive.sum()
        # for l in np.ndindex(s):
        #   synapse = D[i][j][k][l]
        #   if synapse["cw"] > beta and ((Mat[synapse["y"]] & (1<<synapse["x"]))!=0):
        #       count = count + 1
        return count

    def count_positive_synapses(Mat, Segment):
        positiveSynapses = Segment["cw"] > 0.0
        synapsesToActiveCells = Mat[Segment["y"]] & (1 << Segment["x"])
        positiveToActive = positiveSynapses & (synapsesToActiveCells != 0)
        count = positiveToActive.sum()
        #count = 0
        # for l in np.ndindex(s):
        #   synapse = Segment[l]
        #   if synapse["cw"] > 0 and (Mat[synapse["y"]] & (1<<synapse["x"]))!=0:
        #       count = count + 1
        return count

    def count_connected_synapses(Mat, Segment):
        connectedSynapses = Segment["cw"] > beta
        synapsesToActiveCells = Mat[Segment["y"]] & (1 << Segment["x"])
        connectedToActive = connectedSynapses & (synapsesToActiveCells != 0)
        count = connectedToActive.sum()
        #count = 0
        # for l in np.ndindex(s):
        #   synapse = Segment[l]
        #   if synapse["cw"] > 0 and (Mat[synapse["y"]] & (1<<synapse["x"]))!=0:
        #       count = count + 1
        return count

    def closest_to_connected_synapses(Mat, Segment):
        synapsesToActiveCells = Mat[Segment["y"]] & (1 << Segment["x"])
        connectedToActive = Segment["cw"] * (synapsesToActiveCells != 0)
        count = connectedToActive.sum()
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
        #   synapseOld = D[x][y][z][l]
        #   synapseNew = Dnew[x][y][z][l]
        #   delta = 0.0
        #   if ((A['t-1'][synapseOld["y"]]) & (1<<synapseOld["x"])) != 0:
        #       delta = delta + pPos                            # reinforcing synapses to active cells
        #   else:
        #       delta = delta - pNeg                            # negatively reinforcing synapses to inactive cells
        #   synapseNew["cw"] = synapseOld["cw"] + delta
        #   if synapseNew["cw"] < 0.0:
        #       synapseNew["cw"] = 0.0
        #   if synapseNew["cw"] > 1.0:
        #       synapseNew["cw"] = 1.0

        #   Dnew[x][y][z][l]["x"] = synapseOld["x"]
        #   Dnew[x][y][z][l]["y"] = synapseOld["y"]
        #   Dnew[x][y][z][l]["cw"] = synapseNew["cw"]



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
        #   synapseOld = D[x][y][z][l]
        #   synapseNew = Dnew[x][y][z][l]
        #   synapseNew["cw"] = synapseOld["cw"] - pDec
        #   if synapseNew["cw"] < 0.0:
        #       synapseNew["cw"] = 0.0
        #   if synapseNew["cw"] > 1.0:
        #       synapseNew["cw"] = 1.0
        #   Dnew[x][y][z][l]["x"] = synapseOld["x"]
        #   Dnew[x][y][z][l]["y"] = synapseOld["y"]
        #   Dnew[x][y][z][l]["cw"] = synapseNew["cw"]
            
    def repr_human(Mat):
        text = ""
        for i in range(m):
            row = ""
            for j in range(n):
                if Mat[j] & (1<<i) != 0:
                    row += "+"                                  #   + -> Active
                else:   
                    row += "-"                                  #   - -> Inactive
            text += row
            text += '\n'
        return text
    
    active_cells = []
    training_results = []

    flagAltSeq = 2                                              # 0 -> Alternating; 1 -> Sequential; 2 -> Few each

    for ntrials in range(nTrainingTrials):
        for seq in sequences:
            for nr in range(nRepConsecutive):
                current_result = []
                for syllable in seq:
                    W = S[syllable]                                 # marks winning columns
                    A['t'] = 0
                    P['t'] = 0

                    # ---- Step 3: Learning ---- #
                    Dnew = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])
                    np.copyto(Dnew, D)  # Dnew = D
                    for j in np.ndindex(n):                         # active dendritic segment reinforced if cell predicted correctly
                        if W[j] == 1:
                            if P['t-1'][j] != 0:
                                for i in range(m):
                                    if ((P['t-1'][j] & (1<<i)) != 0):
                                        for k in np.ndindex(d):
                                            iSA = isSegmentActive(i, j, k, D, A['t-1'])
                                            if iSA >= theta:
                                                reinforce(i, j, k, D, Dnew)
                                                # if syllable == "B":
                                                #     print "Prediction ", seq, syllable, ntrials*nRepConsecutive+nr, i, j, k, iSA
                                                
                            else:                                   # if current winning column not predicted   
                                        

                                maxCloseness = 0.0
                                maxRow = np.random.random_integers(0, m-1)
                                maxSeg = np.random.random_integers(0, d-1)
                                if chooseMax == 1:
                                    for i, k in np.ndindex(m, d):       # to find distal segment in current column closest to activation
                                        if maxCondition == 0:
                                            currCloseness = count_connected_synapses(A['t-1'], D[i][j][k])
                                        elif maxCondition == 1:
                                            currCloseness = closest_to_connected_synapses(A['t-1'], D[i][j][k])
                                        else:
                                            currCloseness = count_positive_synapses(A['t-1'], D[i][j][k])
                                        if maxCloseness < currCloseness:
                                            maxCloseness = currCloseness
                                            maxRow = i
                                            maxSeg = k
                                # if syllable == "B":
                                #     print "Here ", seq, syllable, ntrials*nRepConsecutive+nr, maxRow, j, maxSeg, maxCloseness
                                
                                
                                if replaceSynapses == 1 and len(active_cells)!=0:
                                    list_active = list(active_cells)
                                    np.random.shuffle(list_active)
                                    for l in np.ndindex(s):
                                        synX = D[maxRow][j][maxSeg][l]["x"]
                                        synY = D[maxRow][j][maxSeg][l]["y"]
                                        if A['t-1'][synY] & (1 << synX) == 0:          # maybe it's needed for some other syllable if cw > beta
                                            r = np.random.randint(0, m*n)
                                            pos = r%len(list_active)            # randomly pick an active cell
                                            Dnew[maxRow][j][maxSeg][l]["x"] = list_active[pos][0]     # replace existing synapse with synapse to active cell
                                            Dnew[maxRow][j][maxSeg][l]["y"] = list_active[pos][1]
                                            Dnew[maxRow][j][maxSeg][l]["cw"] = np.random.uniform(beta, 1.0) # synPerm   # to ensure it has a chance to become a strong connection
                                
                                reinforce(maxRow, j, maxSeg, D, Dnew)
                            
                                # if nStepReplace!=0 and ntrials % nStepReplace == 0:
                                #       for i, k in np.ndindex(m, d):
                                #       if i!=maxRow or k!=maxSeg:                      # no match found, hence updating synapses
                                # for k in np.ndindex(d):
                        #     if nStepReplace!=0 and ((ntrials*nRepConsecutive)+nr % nStepReplace)== 0:
                    #           for i, k in np.ndindex(m, d):       # to find distal segment in current column closest to activation
                            #       currSum = count_positive_synapses(A['t-1'], D[i][j][k])
                            #       if maxSum < currSum:
                            #           maxSum = currSum
                            #           maxRow = i
                            #           maxSeg = k

                                

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
                    for i,j in np.ndindex(m,n):
                        if (A['t'][j] & 1<<i) == 0:
                            for k in np.ndindex(d):
                                if isSegmentActive(i, j, k, D, A['t-1']) >= theta:
                                    decay(i, j, k, D, Dnew)

                    # for j in np.ndindex(n):
                    #   A['t'][j] = 0
                    #   P['t'][j] = 0
                    #   if W[j] == 1:
                    #       if P['t-1'][j] != 0:
                    #           A['t'][j] = P['t-1'][j]             # cell activated if present in winning column and if predicted previously
                    #       else:
                    #           A['t'][j] = pow(2,m)-1              # cell activated if present in winning column and if no cell in the column had been predicted
                            
                    #       for i in range(m):
                    #           if (A['t'][j] & (1 << i)) != 0:
                    #               active_cells.append([i,j])

                #   for j in np.ndindex(n):                                             # to negatively reinforce active segments of inactive cells
                        # if A['t'][j] == 0:
                        #   for i, k in np.ndindex(m, d):
                        #       if isSegmentActive(i, j, k, D, A['t-1']) == True:
                        #           decay(i, j, k, D, Dnew)

                    for i,j in np.ndindex(m,n):                     # computing predictive state for this time step
                        for k in np.ndindex(d):
                            if isSegmentActive(i, j, k, Dnew, A['t']) >= theta:
                                P['t'][j] = P['t'][j] | 1<<i
                                break
                    
                    # ActiveSegments = []
                    # for i,j in np.ndindex(m,n):                       # computing predictive state for this time step
                    #   for k in np.ndindex(d):
                    #       cC = isSegmentActive(i, j, k, Dnew, A['t'])
                    #       if cC >= theta:
                    #           ActiveSegments.append([cC, i, j, k])
                    #           # P['t'][j] = P['t'][j] | 1<<i
                    # ActiveSegments.sort()
                    # c = 0
                    # for Segment in ActiveSegments:
                    #   P['t'][Segment[2]] = P['t'][Segment[2]] | 1<<Segment[1]
                    #   c = c+1
                    #   if c==6:
                    #       break

                    
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
    predictions = []
    got_1_flag = False
    # got_2_flag = False
    # got_both_flag = False

    for test in tests:
        current_result = []
        seq_predicted = [test[0]]
        for syllable in test:
            W = S[syllable]                                     # marks winning columns
            A['t'] = 0
            P['t'] = 0

            Atemp1 = P['t-1'] * W
            Atemp2 = ((Atemp1 != 0) != W) * (pow(2,m)-1)
            A['t'] = Atemp1 + Atemp2

            # for j in np.ndindex(n):
            #   # A['t'][j] = 0
            #   # P['t'][j] = 0
            #   if W[j] == 1:
            #       if P['t-1'][j] != 0:
            #           A['t'][j] = P['t-1'][j]                 # cell activated if present in winning column and if predicted previously
            #       else:
            #           A['t'][j] = pow(2,m)-1                  # cell activated if present in winning column and if no cell in the column had been predicted
                                    
            for i,j in np.ndindex(m,n):                     # computing predictive state for this time step
                for k in np.ndindex(d):
                    if isSegmentActive(i, j, k, D, A['t']) >= theta:
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
        predictions.append(seq_predicted)
    if predictions[0] == sequences[0]:
        got_1_flag = True
    # if predictions[1] == sequences[1]:
    #   got_2_flag = True
    # if predictions == sequences:
    #   got_both_flag = True

    print "Writing to json file: ", resFile
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
        "upper limit of initialisation": initialisingLimit
    }

    learning_parameters = {
        "long term potentiation [pPos]": pPos,
        "long term depression [pNeg]": pNeg,
        "decay [pDec]": pDec,
        "no. of trials in training [nTrainingTrials]": nTrainingTrials,
        "no. of consecutive trials for same sequence": nRepConsecutive,
        "alternating/sequential": flagAltSeq,
        "choose maximum acc. to some algo": chooseMax,
        "choosing max acc to countConnectedSynapses(0) or closest to activation(1)": maxCondition
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
        "GitHash": "fe77f2f8faf5d7d5d7a058332bd7c938209048a9",
        "Input": input_parameters,
        "Results": results
    }

    with open(resFile, 'w') as outfile:  
        json.dump(Data, outfile, sort_keys=True, indent=4, separators=(',', ':\t'))
    # return got_1_flag, got_2_flag, got_both_flag
    return got_1_flag
