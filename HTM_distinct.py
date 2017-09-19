# coding: utf-8

import json
import numpy as np


def HTM(arg_resFile):                                       # change parameters according to which one you want to test
    
    # ---- Parameters ---- #

    n = 21                                                  # no. of columns in a layer
    m = 6                                                   # no. of cells per column
    d = 1                                                   # no. of distal segments per cell
    
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
    maxCondition = 2                                        # 0 -> # connected synapses 1 -> closest to activation by summation
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
        for x, y in np.ndindex(m, n):
            if (Mat[y] & (1 << x) != 0) and (Segment[x][y]>beta):
                count = count + 1
                if count >= theta:
                    return True
        return False

    # returns the no. of synapses connected to active cells
    def count_connected_synapses(Mat, Segment):
        count = 0.0
        for x, y in np.ndindex(m, n):
            if (Mat[y] & (1 << x) != 0) and (Segment[x][y]>beta):
                count = count + 1
        return count

    # returns the sum of connection weights of synapses to active cells
    def closest_to_connected_synapses(Mat, Segment):
        count = 0.0
        for x, y in np.ndindex(m, n):
            if (Mat[y] & (1 << x) != 0) and (Segment[x][y]>0.0):
                count = count + Segment[x][y]
        return count

    # increases weights of the synapses to active cells and decreases weights of the rest
    def reinforce(x, y, z, D, Dnew, Mat):
        segmentOld = D[x][y][z]
        segmentNew = Dnew[x][y][z]
        for i, j in np.ndindex(m, n):
            delta = 0.0
            if (Mat[j] & (1 << i) != 0):
                delta = delta + pPos
            else:
                delta = delta - pNeg
            segmentNew[i][j] = segmentOld[i][j] + delta
            segmentNew[i][j] = min(1, segmentNew[i][j])
            segmentNew[i][j] = max(0, segmentNew[i][j])
        Dnew[x][y][z] = segmentNew

    # decreases weights of all synapses in the segment
    def decay(x, y, z, D, Dnew):
        segmentOld = D[x][y][z]
        segmentNew = Dnew[x][y][z]
        segmentNew = segmentOld - pDec
        for i, j in np.ndindex(m, n):
            segmentNew[i][j] = max(0, segmentNew[i][j])
        Dnew[x][y][z] = segmentNew

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

    A = np.zeros((n), dtype = [("t", int), ("t-1", int)])                   # activation matrix
    P = np.zeros((n), dtype = [("t", int), ("t-1", int)])                   # prediction matrix
    D = np.zeros((m, n, d, m, n), dtype = float)                            # matrix with synaptic weights
    W = np.zeros((n), dtype = int)                                          # feed-forward input representing a syllable
    Dnew = np.zeros((m, n, d, m, n), dtype = float)                         # temporary matrix to store updates in a trial

    training_results = []                                                               # to store the final results of training
    testing_results = []                                                                # to store the final results of testing
    
    # initialising each synapses with random weights
    for i, j, k, p, q in np.ndindex(m, n, d, m, n):
        if i!=p or j!=q:
            D[i][j][k][p][q] = np.random.uniform(0, initialisingLimit)


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

                    for j in np.ndindex(n):
                        if W[j] == 1:                                                               # selecting the "winning" columns

                            # if any cell is predicted in the winning column, the segment that caused this is reinforced
                            if P['t-1'][j] != 0:
                                for i in range(m):
                                    if ((P['t-1'][j] & (1<<i)) != 0):                               # accessing the 'i'th row in the 'j'th column
                                        for k in np.ndindex(d):
                                            if isSegmentActive(A['t-1'], D[i][j][k]) == True:
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

                                    for i, k in np.ndindex(m, d):
                                        if maxCondition == 0:
                                            currCloseness = count_connected_synapses(A['t-1'], D[i][j][k])
                                        elif maxCondition == 1:
                                            currCloseness = closest_to_connected_synapses(A['t-1'], D[i][j][k])

                                        if maxCloseness < currCloseness:
                                            maxCloseness = currCloseness
                                            maxRow = i
                                            maxSeg = k

                                reinforce(maxRow, j, maxSeg, D, Dnew, A['t-1'])
                                

            # ---- Step 2: Computing cell states ---- #

                    # to compute the activation matrix in the new timestep
                    Atemp1 = P['t-1'] * W
                    Atemp2 = ((Atemp1 != 0) != W) * (pow(2,m)-1)
                    A['t'] = Atemp1 + Atemp2

                    # for j in np.ndindex(n):
                    #   A['t'][j] = 0
                    #   P['t'][j] = 0
                    #   if W[j] == 1:
                    #       if P['t-1'][j] != 0:
                    #           A['t'][j] = P['t-1'][j]             
                    #       else:
                    #           A['t'][j] = pow(2,m) - 1            

                    # to decay active segments of cells that did not become active
                    for i,j in np.ndindex(m,n):
                        if (A['t'][j] & 1<<i) == 0:
                            for k in np.ndindex(d):
                                if isSegmentActive(A['t-1'], D[i][j][k]) == True:
                                    decay(i, j, k, D, Dnew)


                    # to compute the predictive state for this time step
                    for i,j in np.ndindex(m,n):
                        for k in np.ndindex(d):
                            if isSegmentActive(A['t'], Dnew[i][j][k]) == True:
                                P['t'][j] = P['t'][j] | 1<<i
                                break
                    np.copyto(D, Dnew) # D[...] = Dnew

                    
                    # to interpret activation in current state
                    out = ""
                    outputW = A['t'] > 0
                        
                    for syll in S:
                        if ((outputW & S[syll]) == S[syll]).all():
                            out = out + syll

                    # to interpret prediction in previous state
                    pred = ""
                    predW = P['t-1'] > 0
                        
                    for syll in S:
                        if ((predW & S[syll]) == S[syll]).all():
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
                Atemp1 = P['t-1'] * W
                Atemp2 = ((Atemp1 != 0) != W) * (pow(2,m)-1)
                A['t'] = Atemp1 + Atemp2

            # computing predictive state for current time step                      
            for i,j in np.ndindex(m,n):
                for k in np.ndindex(d):
                    if isSegmentActive(A['t'], D[i][j][k]) == True:
                        P['t'][j] = P['t'][j] | 1<<i
                        break

            # to interpret activation in current state
            out = ""
            outputW = A['t'] > 0
                
            for syll in S:
                if ((outputW & S[syll]) == S[syll]).all():
                    out = out + syll

            # to interpret prediction in current state
            pred = ""
            predW = P['t'] > 0
            
            for syll in S:
                if ((predW & S[syll]) == S[syll]).all():
                    pred = pred+syll
            
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
        # "no. of potential synapses per segment [s]": s,
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
