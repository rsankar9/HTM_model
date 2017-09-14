import HTM_binary_s

# file_count = 1
# f = open('gotBoth','w')
# f.write('resFile		d	nTT	\n')	
# for d in [2,5]:
# 	for repeat in range(10):
# 		resFile = "WithoutCondRun" + str(file_count) + ".json"
# 		flagBoth = HTM_binary_s.HTM(resFile, d, d*150)
# 		if flagBoth == True:
# 			f.write(resFile + '\t' + str(d) + '\t' + str(d*150) + '\n')	
# 		file_count = file_count + 1

# f.close()

# For a single run:
file_count = 1
resFile = "Run" + str(file_count) + ".json"
HTM_binary_s.HTM(resFile)
