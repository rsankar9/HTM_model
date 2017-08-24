import HTM_diff

count = 1
for d in range(1, 3):
	for s in range(3, 5):
	#	for nSR in [0, 5, 10, 15, 20]:
		for nSR in [10]:
			for nTT in [50]:
				resFile = "Run" + str(count) + ".json"
				HTM_diff.HTM(d, s, nSR, nTT, resFile)
				count = count + 1
