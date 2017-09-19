import HTM_s

# # # # For a single run:
file_count = 1
resFile = "Run" + str(file_count) + ".json"
HTM_s.HTM(resFile)


# # # --- For testing of 2-7 sequence length --- #
# # f = open('TestLength/gotLength','w')
# # f.write('resFile        seq_length  \n')    

# # seq1 = ["A", "B"]
# # seq2 = ["A", "B", "C"]
# # seq3 = ["A", "B", "C", "D"]
# # seq4 = ["A", "B", "C", "D", "E"]
# # seq5 = ["A", "B", "C", "D", "E", "F"]
# # seq6 = ["A", "B", "C", "D", "E", "F", "G"]
# # sequences = [seq1, seq2, seq3, seq4, seq5, seq6]

# # seq_length = 2
# # for seq in sequences:
# #     resFile = "TestLength/Test_length_" + str(seq_length) + ".json"
# #     testing_sequences = []
# #     for i in range(len(seq)-1):
# #         testing_sequences.append(seq[i:])
# #     training_sequences = [seq]
# #     flagGot = HTM_s.HTM(resFile, training_sequences, testing_sequences)
# #     if flagGot == True:
# #         f.write(resFile + '\t' + str(seq_length) + '\t' + '\n') 
# #     seq_length = seq_length + 1

# # f.close()


# # # --- For testing of consecutive sequences --- #
# f = open('TestConsecutive/gotAll','w')
# f.write('resFile        no. of sequences  \n')    

# training_sequences = []
# testing_sequences = []

# seq1 = ["A", "B"]
# seq2 = ["C", "D"]
# seq3 = ["E", "F"]
# sequences = [seq1, seq2, seq3]
# tests = [seq1, seq2, seq3]
# training_sequences.append(sequences)
# testing_sequences.append(tests)

# seq1 = ["A", "B", "C"]
# seq2 = ["C", "D", "E"]
# seq3 = ["E", "F", "G"]
# sequences = [seq1, seq2, seq3]
# tests = [seq1, seq2, seq3, seq1[1:], seq2[1:], seq3[1:]]
# training_sequences.append(sequences)
# testing_sequences.append(tests)

# seq1 = ["A", "B"]
# seq2 = ["B", "C"]
# seq3 = ["C", "D"]
# seq4 = ["D", "E"]
# sequences = [seq1, seq2, seq3, seq4]
# tests = [seq1, seq2, seq3, seq4]
# training_sequences.append(sequences)
# testing_sequences.append(tests)

# seq1 = ["A", "B"]
# seq2 = ["C", "B"]
# seq3 = ["D", "E"]
# seq4 = ["F", "E"]
# seq5 = ["G", "A"]
# sequences = [seq1, seq2, seq3, seq4, seq5]
# tests = [seq1, seq2, seq3, seq4, seq5]
# training_sequences.append(sequences)
# testing_sequences.append(tests)

# seq_length = 2
# for num in range(len(training_sequences)):
#     resFile = "TestConsecutive/Test_consecutive_" + str(num) + ".json"
#     flagGot = HTM_s.HTM(resFile, training_sequences[num], testing_sequences[num])
#     if flagGot == True:
#         f.write(resFile + '\t' + str(len(training_sequences[num])) + '\t' + '\n') 

# f.close()

# # # --- For testing of context memory in sequences of varying length --- #
# # f = open('TestContext/gotContext','w')
# # f.write('resFile        length  \n')    

# # training_sequences = []
# # testing_sequences = []

# # seq1 = ["A", "B", "C"]
# # seq2 = ["D", "B", "E"]
# # seq3 = ["F", "B", "G"]
# # sequences = [seq1, seq2, seq3]
# # tests = [seq1, seq2, seq3, ["B", "CEG"]]
# # training_sequences.append(sequences)
# # testing_sequences.append(tests)

# # seq1 = ["A", "B", "C", "D"]
# # seq2 = ["E", "B", "C", "F"]
# # sequences = [seq1, seq2]
# # tests = [seq1, seq2, ["B", "C", "DF"], ["C", "DF"]]
# # training_sequences.append(sequences)
# # testing_sequences.append(tests)

# # seq1 = ["A", "B", "C", "D", "E"]
# # seq2 = ["F", "B", "C", "D", "G"]
# # sequences = [seq1, seq2]
# # tests = [seq1, seq2, ["B", "C", "D", "EG"], ["C", "D", "EG"], ["D", "EG"]]
# # training_sequences.append(sequences)
# # testing_sequences.append(tests)

# # seq_length = 3
# # for num in range(len(training_sequences)):
# #     resFile = "TestContext/Test_context_" + str(seq_length) + ".json"
# #     flagGot = HTM_s.HTM(resFile, training_sequences[num], testing_sequences[num])
# #     if flagGot == True:
# #         f.write(resFile + '\t' + str(seq_length) + '\t' + '\n') 
# #     seq_length = seq_length + 1

# # f.close()


# # # --- For grid search --- #
# # # file_count = 1
# # # f = open('gotBoth','w')
# # # f.write('resFile      d   nTT \n')    
# # # for d in [2,5]:
# # #   for repeat in range(10):
# # #       resFile = "WithoutCondRun" + str(file_count) + ".json"
# # #       flagBoth = HTM_s.HTM(resFile, d, d*150)
# # #       if flagBoth == True:
# # #           f.write(resFile + '\t' + str(d) + '\t' + str(d*150) + '\n') 
# # #       file_count = file_count + 1

# # # f.close()
