## Scenario

- To test if the model works with each segment having synapses to every cell in the layer.
- Note: The synapses are distinct within a segment, i.e. one synapse per cell in the layer.
- Githash: 74e366c4f8143bffb79c778cd825efbd949cf951
- Total trials = 50
- Theta = 3 = no. of "ON" bits in a syllable representation
- Tested multiple times for seq1 = ["A", "B", "C", "G", "E"], seq2 = ["D", "B", "C", "G", "F"] and shorter overlapping sequences.


## Observations

| Selection Choice | Initialising Limit | Consecutive Trials |     File     | Remarks |
|:----------------:|:------------------:|:------------------:|--------------|---------|
| Random           |         0.5        |         10         | Works11.json | switches between random segments before 1 overcomes, stablises around 25 trials and works always
| Random           |         0.5        |          1         | Works12.json | switches between random segments before 1 overcomes, stablises around 25 trials and works always
| Random           |         1.0        |         10         |              | initialising between 0 and 1 leads to theta connected synapses in many segments easily, so all cells are activated
| Random           |         1.0        |          1         |              | initialising between 0 and 1 leads to theta connected synapses in many segments easily, so all cells are activated
| Count connected  |         0.5        |         10         | Works13.json | switches between random segments before 20 trials of seq 1 and works always
| Count connected  |         0.5        |          1         |              | doesn't work because of same segment chosen by both C's
| Count connected  |         1.0        |         10         |              | doesn't work because everything's taken
| Count connected  |         1.0        |          1         |              | doesn't work because everything's taken
| Closest to connected |     0.5        |         10         | Works14.json | Works always and converges in 10trials each itself because second sequences's C activates new cell after a few trials
| Closest to connected |     0.5        |          1         |              | Never works as second sequence's C gets stuck trying to push same segment over the threshold, which is suppressed by the first sequence in each trial
| Closest to connected |     1.0        |         10         |              | Never works as first sequence captures all cells
| Closest to connected |     1.0        |          1         |              | Never works as all cells are activated from 1st trial itself
| Random           |         0.5        |         10         | Works15.json | Works for theta = 2

## Thoughts

2 things mainly:-

1. No. of cells to be chosen per column

 When there is no prediction from the previous time step, the algorithm  chooses one cell in the column, and reinforces it. The moment a prediction for a column occurs, no new cells are looked for. So, this ends up with only 1 active cell per column. Can this potentially lead to problems?

2. Theta

	[Assumption: There will be 1 cell activated per column, which is what happens with this algorithm.] If there are `w` "ON" bits in a representation, then to depict 1 syllable, you will have `w` active cells learnt, i.e. 1 cell will be learnt per column. Now, to predict the next syllable, you will need a segment with connections to `theta` active cells. In this version of the model, there are only distinct synapses in each segment. So, this gives rise to a few cases:
	- If `theta = w`: This will work fine as the previous time step had `w` active cells.
	- If`theta > w`: This will never work as the previous time step will never have more than `w` active cells, except in the case of the first syllable.
	- If `theta < w`:
If the representations for the syllables are non-overlapping, then this  works. [Tested for random selection algo]	

## To discuss

- [ ]  A better selection algorithm.
- [ ] To use or not to use a fixed initial synapse weight

### TLDR;
It works.
However, to me, it seems to be full of hacks.
