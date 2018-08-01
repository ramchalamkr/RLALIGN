# RLALIGN
## Exploring Reinforcement Learning techniques in Multiple Sequence Alignment.
We use Asynchronous Advantage Actor Critic (A3C) for the MSA problem.

### Prerequisites
Install the latest versions of tensorflow, keras, numpy, pandas, matplotlib.

### Training 
This program trains the agent to learn to optimally align the sequences. The number of sequences and the gameboard size and number of nucleotides needs to be updated in the program.
To train the model, the Train_MSA.py needs to be executed. 

'''
python Train_MSA.py
'''

### Evaluation
The model is then evaluated on Needleman-Wunsch algorithm or the MAFFT(open source code needs to be installed. https://mafft.cbrc.jp/alignment/software/installation_without_root.html) algorithm.
1000 randomly selected samples are chosen and then the alignment scores between RLALIGN and NW are compared and the plots are generated.
