import HTM_diff

file_count = 1

# resFile = "WithCondRun" + str(file_count) + ".json"
# HTM_diff.HTM(resFile)
f = open('gotBoth','w')
f.write('resFile		d	s	nRC\n')	
for d in [1,2,5]:
    for s in [5,7,9]:
        for nRC in [5,10]:
            resFile = "WithoutCondRun" + str(file_count) + ".json"
            flagBoth = HTM_diff.HTM(resFile, s, d, nRC)
            if flagBoth == True:
                f.write(resFile+'\t'+str(d)+'\t'+str(s)+'\t'+str(nRC) +'\n')
            file_count = file_count + 1

f.close()

# for s in [5,8]:
# 	for file_count in range(1,10):
# 	    resFile = "WithoutCondRun" + str(file_count) + '_s' + str(s) + ".json"
# 	    HTM_diff.HTM(resFile, s)
