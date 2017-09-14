# coding: utf-8

import json
import numpy as np


def HTM(arg_resFile):                       # change parameters according to which one you want to test
    
    # ---- Parameters ---- #

    n = 21                                                  # no. of columns in a layer
    m = 6                                                   # no. of cells per column
    d = 1                                                   # no. of distal segments per cell
    s = 5                                                   # no. of potential synapses per segment

    beta = 0.5                                              # synaptic connection threshold
    theta = 3                                               # segment activation threshold
    initialisingLimit = beta                                # synapses are randomly initialised with weights between 0 and initialisingLimit

    pPos = 0.2                                              # long term potentiation
    pNeg = 0.2                                              # long term depression
    pDec = 0.02                                             # constant decay
    nTrainingTrials = 200                                   # total no. of trials
    nRepConsecutive = 5                                     # no. of consecutive trials during training for one sequence
    nTrainingBlocks = nTrainingTrials/nRepConsecutive       # no. of training blocks

    chooseMax = 0                                           # 1 -> on; 0 -> randomly chosen (if 0, maxCondition doesn't matter)
    maxCondition = 2                                        # 0 -> # connected synapses 1 -> closest to activation by summation 2 -> # positive synapses
    replaceSynapses = 1                                     # 0 -> off 1 -> on

    rSeed = np.random.randint(0,1e7)                        # seed
    np.random.seed(rSeed)

    resFile = arg_resFile                                   # name of result file
    repr_matrix_form = False                                # 1 -> as a matrix; 0 -> whole matrix in 1 line

    seq1 = ["A", "B", "C", "G", "E"]
    seq2 = ["D", "B", "C", "G", "F"]

    sequences = [seq1, seq2]                                # list of sequences to be trained on

    test1 = ["A", "B", "C", "G"]
    test2 = ["D", "B", "C", "G"]

    tests = [test1, test2]                                  # list of sequences to be tested on


    # ---- methods ---- #

    # a distal segment is active, if the no. of connected synapses to active cells is greater than theta in that segment
    def isSegmentActive(Mat, Segment):
        connectedSynapses = Segment["cw"] > beta
        synapsesToActiveCells = Mat[Segment["y"]] & (1 << Segment["x"])
        connectedToActive = connectedSynapses & (synapsesToActiveCells != 0)
        count = connectedToActive.sum()

        # count = 0
        # for l in np.ndindex(s):
        #   synapse = D[i][j][k][l]
        #   if synapse["cw"] > beta and ((Mat[synapse["y"]] & (1<<synapse["x"]))!=0):
        #       count = count + 1
        return count

    # returns the no. of synapses existing to active cells
    def count_positive_synapses(Mat, Segment):
        positiveSynapses = Segment["cw"] > 0.0
        synapsesToActiveCells = Mat[Segment["y"]] & (1 << Segment["x"])
        positiveToActive = positiveSynapses & (synapsesToActiveCells != 0)
        count = positiveToActive.sum()

        # count = 0
        # for l in np.ndindex(s):
        #   synapse = Segment[l]
        #   if synapse["cw"] > 0 and (Mat[synapse["y"]] & (1<<synapse["x"]))!=0:
        #       count = count + 1
        return count

    # returns the no. of synapses connected to active cells
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

    # returns the sum of connection weights of synapses to active cells
    def closest_to_connected_synapses(Mat, Segment):
        synapsesToActiveCells = Mat[Segment["y"]] & (1 << Segment["x"])
        connectedToActive = Segment["cw"] * (synapsesToActiveCells != 0)
        count = connectedToActive.sum()

        #count = 0
        # for l in np.ndindex(s):
        #   synapse = Segment[l]
        #   if synapse["cw"] > 0 and (Mat[synapse["y"]] & (1<<synapse["x"]))!=0:
        #       count = count + synapse["cw"]
        return count

    # increases weights of the synapses to active cells and decreases weights of the rest
    def reinforce(x, y, z, D, Dnew, Mat):
        segmentOld = D[x][y][z]
        segmentNew = Dnew[x][y][z]
        deltaPos = ((Mat[segmentOld["y"]]) & (1<<segmentOld["x"])) != 0
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
        #   synapseNew["cw"] = max(0, synapseNew["cw"]
        #   synapseNew["cw"] = min(1, synapseNew["cw"]

    # decreases weights of all synapses in the segment
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
        #   synapseNew["cw"] = max(0, synapseNew["cw"]
        #   synapseNew["cw"] = min(1, synapseNew["cw"]
            
    # to print the matrix as a single line
    def repr_human(Mat):

        if repr_matrix_form == False:
            text = ""
            for i in range(m):
                row = ""
                for j in range(n):
                    if Mat[j] & (1<<i) != 0:
                        row += "+"                              #   + -> Active
                    else:   
                        row += "-"                              #   - -> Inactive
                text += row
                text += '\n'
            return text

        else:
            matrix = []
            for i in range(m):
                row = ""
                for j in range(n):
                    if Mat[j] & (1<<i) != 0:
                        row += "+"                              #   + -> Active
                    else:   
                        row += "-"                              #   - -> Inactive
                matrix.append(row)
            return matrix


                     
    # ---- Step 1: Initialisation ---- #

    # encoding for each syllable
    S = { "A" : np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
          "B" : np.array([0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
          "C" : np.array([0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]),
          "D" : np.array([0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0]),
          "E" : np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0]),
          "F" : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0]),
          "G" : np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]),  }

    A = np.zeros((n), dtype = [("t", int), ("t-1", int)])                               # activation matrix
    P = np.zeros((n), dtype = [("t", int), ("t-1", int)])                               # prediction matrix
    D = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])         # matrix with synaptic weights
    W = np.zeros((n), dtype = int)                                                      # feed-forward input representing a syllable
    Dnew = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])      # temporary matrix to store updates in a trial

    active_cells = []                                                                   # to store the active cells
    training_results = []                                                               # to store the final results of training
    testing_results = []                                                                # to store the final results of testing
    predictions = []                                                                    # to check predictions
    got_all_flag = False                                                                # to denote if all the predictions were correct or not

    # initialising each distal segment with synapses to random cells with random weights
    for i, j, k, l in np.ndindex(m, n, d, s):
        x = np.random.randint(0, m)
        y = np.random.randint(0, n)
        cw = np.random.uniform(0.0,initialisingLimit)
        while x == i and y==j:
            x = np.random.random_integers(0, m-1)
            y = np.random.random_integers(0, n-1)
        while cw == 0:
            cw = np.random.uniform(0.0,initialisingLimit)
        D[i][j][k][l]["x"] = x
        D[i][j][k][l]["y"] = y
        D[i][j][k][l]["cw"] = cw               
    

    # ---- Training ---- #

    for ntrials in range(nTrainingBlocks):
        for seq in sequences:
            for nr in range(nRepConsecutive):
                current_result = []
                for syllable in seq:
                    W = S[syllable]
                    A['t'] = 0
                    P['t'] = 0

                    # ---- Step 3: Learning ---- #

                    np.copyto(Dnew, D)                                                              # Dnew = D

                    for j in np.ndindex(n):
                        if W[j] == 1:                                                               # selecting the "winning" columns

                            # if any cell is predicted in the winning column, the segment that caused this is reinforced
                            if P['t-1'][j] != 0:
                                for i in range(m):
                                    if ((P['t-1'][j] & (1<<i)) != 0):                               # accessing the 'i'th row in the 'j'th column
                                        for k in np.ndindex(d):
                                            if isSegmentActive(A['t-1'], D[i][j][k]) >= theta:
                                                reinforce(i, j, k, D, Dnew, A['t-1'])
                            
                            # if no cell in the winning column is predicted, a segment is chosen to represent it and reinforced
                            else:
                                maxCloseness = 0.0
                                maxRow = np.random.random_integers(0, m-1)
                                maxSeg = np.random.random_integers(0, d-1)

                                # if the 'best matching' segment is to be chosen [not random selection]
                                if chooseMax == 1:

                                    if maxCondition == 0:
                                        maxCloseness = count_connected_synapses(A['t-1'], D[maxRow][j][maxSeg])
                                    elif maxCondition == 1:
                                        maxCloseness = closest_to_connected_synapses(A['t-1'], D[maxRow][j][maxSeg])
                                    else:
                                        maxCloseness = count_positive_synapses(A['t-1'], D[maxRow][j][maxSeg])

                                    for i, k in np.ndindex(m, d):
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

                                # to replace an inactive synapse with a synapse connected to an active cells in the chosen segment
                                if replaceSynapses == 1 and len(active_cells) != 0:
                                        list_active = list(active_cells)
                                        np.random.shuffle(list_active)
                                        
                                        for l in np.ndindex(s):
                                            synX = D[maxRow][j][maxSeg][l]["x"]
                                            synY = D[maxRow][j][maxSeg][l]["y"]
                                            synCW = D[maxRow][j][maxSeg][l]["cw"]
                                            if (A['t-1'][synY] & (1 << synX) == 0):
                                                r = np.random.randint(0, m*n)
                                                pos = r%len(list_active)                                        # to randomly pick an active cell
                                                Dnew[maxRow][j][maxSeg][l]["x"] = list_active[pos][0]
                                                Dnew[maxRow][j][maxSeg][l]["y"] = list_active[pos][1]
                                                Dnew[maxRow][j][maxSeg][l]["cw"] = np.random.uniform(beta, 1.0)
                                                break
                                
                                reinforce(maxRow, j, maxSeg, D, Dnew, A['t-1'])
                    

                    # ---- Step 2: Computing cell states ---- #

                    active_cells = []


                    # to compute the activation matrix in the new timestep
                    Atemp1 = P['t-1'] * W                                               # a cell in the winning column is activated, if predicted previously
                    Atemp2 = ((Atemp1 != 0) != W) * (pow(2,m) - 1)                      # if no cell in the winning column is predicted, all cells are activated
                    A['t'] = Atemp1 + Atemp2

                    # for j in np.ndindex(n):
                    #   A['t'][j] = 0
                    #   P['t'][j] = 0
                    #   if W[j] == 1:
                    #       if P['t-1'][j] != 0:
                    #           A['t'][j] = P['t-1'][j]             
                    #       else:
                    #           A['t'][j] = pow(2,m) - 1            


                    # to update the list of active cells for the new timestep
                    for j in np.ndindex(n):
                        if W[j] == 1:
                            for i in range(m):
                                if (A['t'][j] & (1 << i)) != 0:
                                    active_cells.append([i,j])


                    # to decay active segments of cells that did not become active
                    for i,j in np.ndindex(m,n):
                        if (A['t'][j] & (1<<i)) == 0:
                            for k in np.ndindex(d):
                                if isSegmentActive(A['t-1'], D[i][j][k]) >= theta:
                                    decay(i, j, k, D, Dnew)


                    # to compute the predictive state for this time step
                    for i,j in np.ndindex(m,n):                     
                        for k in np.ndindex(d):
                            if isSegmentActive(A['t'], Dnew[i][j][k]) >= theta:
                                P['t'][j] = P['t'][j] | 1<<i
                                break
                   
                    
                    np.copyto(D, Dnew) # D[...] = Dnew

                    
                    # to interpret current state
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
                    
                # reset on encountering "end" syllable i.e. end of sequence
                P['t-1'] = 0
                A['t-1'] = 0
                active_cells = []

            if ntrials == nTrainingBlocks - 1:
                training_results.append({
                    "Training Result": current_result 
                })


    # ---- Testing ---- #
    print "Testing"

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

            for i,j in np.ndindex(m,n):                     
                for k in np.ndindex(d):
                    if isSegmentActive(A['t'], D[i][j][k]) >= theta:
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
    
    if predictions == sequences:
        got_all_flag = True

    
    # ---- Printing results to a json file ---- #

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
        "upper limit of initialisation": initialisingLimit
    }

    learning_parameters = {
        "long term potentiation [pPos]": pPos,
        "long term depression [pNeg]": pNeg,
        "decay [pDec]": pDec,
        "no. of trials in training [nTrainingTrials]": nTrainingTrials,
        "no. of consecutive trials for same sequence": nRepConsecutive,
        "choose maximum acc. to which algo": chooseMax,
        "choosing max acc to countConnectedSynapses(0) or closest to activation(1)": maxCondition,
        "synapse replacement": replaceSynapses
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
        "GitHash": "",
        "Input": input_parameters,
        "Results": results
    }

    with open(resFile, 'w') as outfile:  
        json.dump(Data, outfile, sort_keys=True, indent=4, separators=(',', ':\t'))
    
    return got_all_flag
