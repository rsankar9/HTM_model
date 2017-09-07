import HTM_diff

file_count = 1
# f = open('got1','w')
# g = open('got2','w')
# h = open('gotBoth','w')
resFile = "Run" + str(file_count) + ".json"
HTM_diff.HTM(resFile)
	# flag1, flag2, flagBoth = HTM_diff.HTM(s, theta, resFile)
	# if flag1 == True:
	# 	f.write(resFile+'\n')
	# if flag2 == True:
	# 	g.write(resFile+'\n')
	# if flagBoth == True:
	# 	h.write(resFile+'\n')	
file_count = file_count + 1
# f.close()
# g.close()
# h.close()
