import HTM_diff

file_count = 1
f = open('got1','w')
g = open('got2','w')
h = open('gotBoth','w')
for d in [10, 15, 20]:
	for s in range(13,16):
		for nSR in [10, 15]:
			for nTT in [50, 100]:
				resFile = "Run" + str(file_count) + ".json"
				flag1, flag2, flagBoth = HTM_diff.HTM(d, s, nSR, nTT, resFile)
				if flag1 == True:
					f.write(resFile+'\n')
				if flag2 == True:
					g.write(resFile+'\n')
				if flagBoth == True:
					h.write(resFile+'\n')	
				file_count = file_count + 1
f.close()
g.close()
h.close()
