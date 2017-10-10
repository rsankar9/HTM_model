### ------------------------------------------------------------------------------------ ###

# This code is a modification of Numenta Inc.'s HTM model 
# by S. Remya, Nicolas Rougier and Arthur Leblois
# with reference to the paper mentioned below.
# https://doi.org/10.3389/fncir.2016.00023

# The model attempts to emulate a juvenile bird learning its song from a tutor.
# In the sensory phase of this learning, the young bird remembers the song it hears and
# in the sensorimotor phase, it tries to replicate it.
# This code is concerned with the sensory phase.

### ------------------------------------------------------------------------------------ ###

# coding: utf-8

import json
import numpy as np


def HTM(arg_resFile):                                       # change parameters according to which one you want to test
    
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
    nTrainingTrials = 250                                   # total no. of trials
    nRepConsecutive = 5                                     # no. of consecutive trials during training for one sequence
    nTrainingBlocks = nTrainingTrials/nRepConsecutive       # no. of training blocks

    chooseMax = 0                                           # 1 -> on; 0 -> randomly chosen (if 0, maxCondition doesn't matter)
    maxCondition = 2                                        # 0 -> # connected synapses 1 -> closest to activation by summation 2 -> # positive synapses
    replaceSynapses = 1                                     # 0 -> off 1 -> on

    testFreeFlag = 1                                        # 0 -> testing with feed-forward input; 1 -> free testing

    rSeed = np.random.randint(0,1e7)                        # seed
    np.random.seed(rSeed)

    resFile = arg_resFile                                   # name of result file
    repr_matrix_form = True                                 # 1 -> as a matrix; 0 -> whole matrix in 1 line
    got_all_flag = True                                     # to denote if all the predictions were correct or not

    seq1 = ["A", "B", "C", "G", "E"]
    seq2 = ["D", "B", "C", "G", "F"]

    sequences = [seq1, seq2]                                # list of sequences to be trained on

    test1 = ["A", "B", "C", "G", "E"]
    test2 = ["D", "B", "C", "G", "F"]

    tests = [test1, test2]                                  # list of sequences to be tested on


    # ---- methods ---- #

    # a distal segment is active, if the no. of connected synapses to active cells is greater than theta in that segment
    def isSegmentActive(Mat, Segment):
        count = 0
        for syn in Segment:
            if syn["cw"] > beta and Mat[syn["x"], syn["y"]] == 1:
                count = count + 1
        return (count >= theta)

    # returns the no. of synapses existing to active cells
    def count_positive_synapses(Mat, Segment):
        count = 0
        for syn in Segment:
            if syn["cw"] > 0.0 and Mat[syn["x"], syn["y"]] == 1:
                count = count + 1
        return count

    # returns the no. of synapses connected to active cells
    def count_connected_synapses(Mat, Segment):
        count = 0
        for syn in Segment:
            if syn["cw"] > beta and Mat[syn["x"], syn["y"]] == 1:
                count = count + 1
        return count

    # returns the sum of connection weights of synapses to active cells
    def count_connected_synapses(Mat, Segment):
        count = 0
        for syn in Segment:
            if syn["cw"] > 0.0 and Mat[syn["x"], syn["y"]] == 1:
                count = count + syn["cw"]
        return count

    # increases weights of the synapses to active cells and decreases weights of the rest
    def reinforce(x, y, z, Dnew, Mat):    
        for l in np.ndindex(s):
            synapseNew = Dnew[x][y][z][l]
            delta = 0.0
            if Mat[synapseNew["x"]][synapseNew["y"]] == 1:
                delta = delta + pPos
            else:
                delta = delta - pNeg

            synapseNew["cw"] = synapseNew["cw"] + delta
            synapseNew["cw"] = max(0, synapseNew["cw"])
            synapseNew["cw"] = min(1, synapseNew["cw"])

    def decay(x, y, z, Dnew):
        delta = 0
        for l in np.ndindex(s):
            syn = Dnew[x][y][z][l]
            Dnew[x][y][z][l]["x"] = syn["x"]
            Dnew[x][y][z][l]["y"] = syn["y"]
            Dnew[x][y][z][l]["cw"] = syn["cw"] - pDec
            Dnew[x][y][z][l]["cw"] = max(0, Dnew[x][y][z][l]["cw"])

    # to print the matrix as a single line
    def repr_human(Mat):
        if repr_matrix_form == False:
            text = ""
            for i in range(m):
                for j in range(n):
                    if Mat[i,j] == 1:
                        text += "+"                                 #   + -> Active
                    else:   
                        text += "-"                                 #   - -> Inactive
                text += "\n"
            return text

        else:
            matrix = []
            for i in range(m):
                row = ""
                for j in range(n):
                    if Mat[i,j] == 1:
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

    A = np.zeros((m, n), dtype = [("t", bool), ("t-1", bool)])                          # activation matrix
    P = np.zeros((m, n), dtype = [("t", bool), ("t-1", bool)])                          # prediction matrix
    D = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])         # matrix with synaptic weights
    W = np.zeros((n), dtype = int)                                                      # feed-forward input representing a syllable
    Dnew = np.zeros((m, n, d, s), dtype = [("x", int), ("y", int), ("cw", float)])      # temporary matrix to store updates in a trial

    active_cells = []                                                                   # to store the active cells
    training_results = []                                                               # to store the final results of training
    testing_results = []                                                                # to store the final results of testing
    
    # initialising each distal segment with synapses to random cells with random weights
    for i, j, k, l in np.ndindex(m, n, d, s):                               # initialising each distal segment with synapses to random cells with random strength
        x = np.random.random_integers(0, m-1)
        y = np.random.random_integers(0, n-1)
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

    for nBlocks in range(nTrainingBlocks):
        for seq in sequences:
            for nr in range(nRepConsecutive):
                current_result = []
                for syllable in seq:
                    W = S[syllable]
                    A['t'] = 0
                    P['t'] = 0

                    # ---- Step 3: Learning ---- #

                    np.copyto(Dnew, D)                                                              # Dnew = D
                    
                    # if any cell is predicted in the winning column, the segment that caused this is reinforced
                    for j in np.ndindex(n):
                        if W[j] == 1:
                            for i in np.ndindex(m):
                                if P['t-1'][i][j] != 0:
                                    for k in np.ndindex(d):
                                        if isSegmentActive(A['t-1'], D[i][j][k]) == True:
                                            reinforce(i, j, k, Dnew, A['t-1'])

                    # if no cell in the winning column is predicted, a segment is chosen to represent it and reinforced
                    for j in np.ndindex(n):                     
                        if W[j] == 1:
                            tempSum = 0
                            for i in np.ndindex(m):
                                tempSum += P['t-1'][i][j]
                            if tempSum == 0:
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
                                            if A['t-1'][synX][synY] == False:
                                                r = np.random.randint(0, m*n)
                                                pos = r%len(list_active)
                                                Dnew[maxRow][j][maxSeg][l]["x"] = list_active[pos][0]
                                                Dnew[maxRow][j][maxSeg][l]["y"] = list_active[pos][1]
                                                Dnew[maxRow][j][maxSeg][l]["cw"] = np.random.uniform(beta, 1.0)
                                                break
                                reinforce(maxRow, j, maxSeg, Dnew, A['t-1'])


                    # ---- Step 2: Computing cell states ---- #

                    active_cells = []

                    # to compute the activation matrix in the new timestep
                    # and to update the list of active cells for the new timestep
                    for j in np.ndindex(n):
                        if W[j] == 1:
                            found_active = 0
                            for i in np.ndindex(m):
                                if P['t-1'][i][j] == 1:
                                    A['t'][i][j] = 1
                                    found_active = 1
                                    active_cells.append([i,j])
                            if found_active==0:
                                for i in np.ndindex(m):
                                    A['t'][i][j] = 1
                                    active_cells.append([i,j])
                        else:
                            # to decay active segments of cells that did not become active
                            for i in np.ndindex(m):
                                if A['t'][i][j] == 0:
                                    for k in np.ndindex(d):
                                        if isSegmentActive(A['t-1'], D[i][j][k]) == True:
                                            decay(i, j, k, Dnew)


                    # to compute the predictive state for this time step
                    for i,j in np.ndindex(m,n):
                        for k in np.ndindex(d):
                            if isSegmentActive(A['t'], Dnew[i][j][k]) == True:
                                P['t'][i][j] = 1
                    
                    np.copyto(D, Dnew)                                                  # D[...] = Dnew

                    # to interpret activation in current state
                    out = ""
                    outputW = np.zeros(n, dtype=int)
                    
                    for j in np.ndindex(n):
                        for i in np.ndindex(m):
                            if A['t'][i][j]==1:
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
                            
                    # to interpret prediction in previous state
                    pred = ""
                    predW = np.zeros(n, dtype=int)

                    for j in np.ndindex(n):
                        for i in np.ndindex(m):
                            if P['t-1'][i][j]==1:
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

                    # to store results of current time step in last trial
                    if nr == nRepConsecutive - 1:       
                        syllable_result = {}
                        syllable_result["prediction"] = pred
                        syllable_result["P['t-1']"] = repr_human(P['t-1'])
                        syllable_result["output"] = out
                        syllable_result["A['t']"] = repr_human(A['t'])
                        current_result.append(syllable_result)

                    # reset for each timestep
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

            # to store results of last trial
            if nBlocks == nTrainingBlocks - 1:
                training_results.append({
                    "Training Result": current_result 
                })


    # ---- Testing ---- #
    
    print "Testing"

    for test in tests:
        current_result = []
        seq_predicted = [test[0]]
        for timeStep in range(len(test)-1):
            syllable = test[timeStep]
            W = S[syllable]                                     # marks winning columns
            A['t'] = 0
            P['t'] = 0            

            # computing activation for current time step
            A['t'] = P['t-1']                                   # for free testing

            if timeStep == 0 or testFreeFlag == 0:              # for constrained testing and first timestep
                for j in np.ndindex(n):
                    if W[j] == 1:
                        found_active = 0
                        for i in np.ndindex(m):
                            if P['t-1'][i][j] == 1:
                                A['t'][i][j] = 1
                                found_active = 1
                        if found_active==0:
                            for i in np.ndindex(m):
                                A['t'][i][j] = 1
                    
            # computing predictive state for current time step
            for i, j in np.ndindex(m, n):
                for k in np.ndindex(d):
                    if isSegmentActive(A['t'], D[i][j][k]) == True:
                        P['t'][i][j] = 1
                        break

            # to interpret activation in current state
            out = ""
            outputW = np.zeros(n, dtype=int)
                    
            for j in np.ndindex(n):
                for i in np.ndindex(m):
                    if A['t'][i][j]==1:
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
                    
            # to interpret prediction in current state
            pred = ""
            predW = np.zeros(n, dtype=int)

            for j in np.ndindex(n):
                for i in np.ndindex(m):
                    if P['t'][i][j]==1:
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
            
            # to store results of current time step
            syllable_result = {}
            syllable_result["prediction"] = pred
            syllable_result["P['t']"] = repr_human(P['t'])
            syllable_result["output"] = out
            syllable_result["A['t']"] = repr_human(A['t'])
            current_result.append(syllable_result)
            seq_predicted.append(pred)

            # reset for each timestep
            P['t-1'] = 0
            A['t-1'] = 0
            P['t-1'] = P['t']
            A['t-1'] = A['t']
            A['t'] = 0
            P['t'] = 0
        
        # reset on encountering "end" syllable i.e. end of sequence
        P['t-1'] = 0
        A['t-1'] = 0
        
        # to store final results of current test sequence
        testing_results.append({
            "Sequence Tested": test,
            "Sequence Predicted": seq_predicted,
            "Test Result": current_result 
        })
        
        # denotes if all sequences were predicted accurately
        got_all_flag = got_all_flag & (seq_predicted == test)
        

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

    testing_parameters = {
        "testing type (constrained(0) / free(1))": testFreeFlag,
        "seed [rSeed]": rSeed
    }

    input_parameters = {
        "Layer Parameters": layer_parameters,
        "Synapse Parameters": synapse_parameters,
        "Learning Parameters": learning_parameters,
        "Testing Parameters": testing_parameters,
    }

    results = {
        # "Training Results": training_results,
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
