# RLALIGN
## Exploring Reinforcement Learning techniques in Multiple Sequence Alignment.
We use Asynchronous Advantage Actor Critic (A3C) for the MSA problem.

### Training 
This program trains the agent to learn to optimally align the sequences. The number of sequences and the gameboard size needs to be provided as input.

### Evaluation
The model is then evaluated on Needleman-Wunsch algorithm or the MAFFT(open source code needs to be installed. https://mafft.cbrc.jp/alignment/software/installation_without_root.html) algorithm.
1000 randomly selected samples are chosen and then the alignment scores between RLALIGN and NW are compared and the box-plot is created.
