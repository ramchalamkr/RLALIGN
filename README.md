## RLALIGN
RLALIGN: A Reinforcement Learning based Multiple Sequence Alignment tool.
The model uses the state-of-the-art RL algorithm - Asynchronous Advantage Actor Critic model.

### Prerequisites
Install the latest versions of tensorflow, keras, numpy, pandas, matplotlib.

### Training 
This program trains the agent to learn to optimally align the sequences. The number of sequences, the gameboard size and number of nucleotides needs to be updated in the program.
To train the model, the Train_MSA.py needs to be executed. 

```
python Train_MSA.py
```
Another version is to execute using the nohup command.

```
nohup python -u Train_MSA.py>log.txt&
```
The results will be saved to the ```results\``` folder.
The model wil be saved to the ``` models\``` folder.

### Evaluation
The model is then evaluated on Needleman-Wunsch algorithm or the MAFFT(open source code needs to be installed. https://mafft.cbrc.jp/alignment/software/installation_without_root.html) algorithm.
1000 randomly selected samples are chosen and then the alignment scores between RLALIGN and NW are compared and the plots are generated.

To evaluate the model, the Test_MSA.py needs to be executed. 

```
python Test_MSA.py
```


#### References
http://web.mit.edu/pensieve/
