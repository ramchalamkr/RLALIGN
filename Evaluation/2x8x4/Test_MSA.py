import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import a3c
import random
import copy
import logging
from collections import deque
import itertools
#import mafft
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10



S_DIM = 16
A_DIM = 28
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.0001
saved_MODEL = './models/saved_model_eps_5000.ckpt'
PROB = [0.95]

#---------------------Added For RLALIGN-------------------------
#------------------Needleman Wunch algorithm------------------
#This function returns to values for case of match or mismatch
def Diagonal(n1,n2,pt):
    if(n1 == n2):
        return pt['MATCH']
    else:
        return pt['MISMATCH']

#------------------------------------------------------------   

def NW(s1,s2,match = 1,mismatch = -1, gap = -1):
    penalty = {'MATCH': match, 'MISMATCH': mismatch, 'GAP': gap} #A dictionary for all the penalty valuse.
    n = len(s1) + 1 #The dimension of the matrix columns.
    m = len(s2) + 1 #The dimension of the matrix rows.
    al_mat = np.zeros((m,n),dtype = int) #Initializes the alighment matrix with zeros.
    #Scans all the first rows element in the matrix and fill it with "gap penalty"
    for i in range(m):
        al_mat[i][0] = penalty['GAP'] * i
    #Scans all the first columns element in the matrix and fill it with "gap penalty"
    for j in range (n):
        al_mat[0][j] = penalty['GAP'] * j
    #Fill the matrix with the correct values.
    for i in range(1,m):
        for j in range(1,n):
            di = al_mat[i-1][j-1] + Diagonal(s1[j-1],s2[i-1],penalty) #The value for match/mismatch -  diagonal.
            ho = al_mat[i][j-1] + penalty['GAP'] #The value for gap - horizontal.(from the left cell)
            ve = al_mat[i-1][j] + penalty['GAP'] #The value for gap - vertical.(from the upper cell)
            al_mat[i][j] = max(di,ho,ve) #Fill the matrix with the maximal value.(based on the python default maximum)
    return int(np.matrix(al_mat[m-1][n-1]))

#---------
class GymEnv:
    def __init__(self,boardParameters):
        #self.sequences = sequences
        self.noOfRows=boardParameters['Rows']
        self.noOfCols=boardParameters['Cols']
        #self.vectorized = self.GetStateVector()#1X40 vectorized representation
        self.ActionSpace=self.GetActionSpace()
        self.observation_space=np.asarray([0, 4, 4, 1, 4, 2, 3, 4, 4, 0, 4, 4, 4, 1, 2, 3])
        self.action_space=(self.noOfRows*self.noOfCols * 2) - (2* self.noOfRows)
       
    def observation_space(self):
        n=(self.noOfRows*self.noOfCols,)
        return n

    def reset(self):
        nucleotideList=['A','C','G','T']
        row ={}
        NoOfNucleotidesToUse = 4
        for j in range(self.noOfRows):
            row[j] =""
        for i in range(NoOfNucleotidesToUse):
            for j in range(self.noOfRows):
                randomNo=random.sample(range(0,4),1)
                row[j] += nucleotideList[randomNo[0]]
        temp1 ={}
        string ={}
        for j in range(self.noOfRows):
            temp1[j] = sorted(random.sample(range(0,self.noOfCols),len(row[j])))
            string[j] = ["-"]*self.noOfCols
            #string[j] = list("------")
        for j in range(self.noOfRows):
            for index,random_number in enumerate(temp1[j]):
                string[j][random_number]=row[j][index]
            string[j]="".join(string[j])
        listOfString=[]
        for j in range(self.noOfRows):
            listOfString.append(string[j])
        return listOfString


    def one_hot_encode(self,simpleEncodedList):
        # print "simpleEncoded List Is",simpleEncodedList
        b = np.zeros((self.noOfRows*self.noOfCols, 5))
        b[np.arange(self.noOfRows*self.noOfCols), simpleEncodedList] = 1
        b=b.reshape((self.noOfRows,self.noOfCols*5))
        simpleEncodedList = b
        return np.asarray(simpleEncodedList)
            
        
    def GetActionSpace(self):
        TempActionSpace = []
        for i in range(0,(self.noOfRows*self.noOfCols)):
            if(i%self.noOfCols!=0 and i%self.noOfCols!=self.noOfCols-1):#first col and last col of each row is ignored
                TempActionSpace.append(str(i)+'_L')
                TempActionSpace.append(str(i)+'_R')
            if (i%self.noOfCols==0):
                TempActionSpace.append(str(i)+'_R')
            if (i%self.noOfCols==self.noOfCols-1):
                TempActionSpace.append(str(i)+'_L')
        return TempActionSpace
        
    def GetStateVector(self,sequences):
        Tempvectorized =[]
        for i in sequences:
            for j in i:
                if(j=="A"):
                    Tempvectorized.append(0)
                if(j=="C"):
                    Tempvectorized.append(1)
                if(j=="G"):
                    Tempvectorized.append(2)
                if(j=="T"):
                    Tempvectorized.append(3)
                if(j=="-"):
                    Tempvectorized.append(4)
                #print Tempvectorized
        oneHotEncodedTempVectorize = self.one_hot_encode(Tempvectorized)
        return oneHotEncodedTempVectorize
    
    def get_sequences_from_state(self,state):
        Tempvectorized = state.reshape(self.noOfRows,self.noOfCols)
        sequences=[]
        for i in range(self.noOfRows):
            tempString=""
            for j in range(self.noOfCols):
                if Tempvectorized[i][j]==0:
                    tempString+="A"
                if Tempvectorized[i][j]==1:
                    tempString+="C"
                if Tempvectorized[i][j]==2:
                    tempString+="G"
                if Tempvectorized[i][j]==3:
                    tempString+="T"
                if Tempvectorized[i][j]==4:
                    tempString+="-"
            sequences.append(tempString)
        return sequences
        
    def getLegalActions(self,state,actionSpace):
        legalActions=[]
        for enum,i in enumerate(actionSpace):
            actionIndex=int(i[0:i.index('_')])
            if state[actionIndex]!=4:
                legalActions.append(i)
        return legalActions
    
    def GetReward(self,state):
        allPermutations = itertools.combinations(range(self.noOfRows),2)
        Tempvectorized = state.reshape(self.noOfRows,self.noOfCols)
        TempReward = 0
        for i in allPermutations:
            for j in range(Tempvectorized.shape[1]):
                if(Tempvectorized[i[0]][j] == Tempvectorized[i[1]][j] and Tempvectorized[i[0]][j] in (0,1,2,3)):
                    TempReward+=1
                if(Tempvectorized[i[0]][j] != Tempvectorized[i[1]][j] and Tempvectorized[i[1]][j] in (0,1,2,3)
                   and Tempvectorized[i[0]][j] in (0,1,2,3)):
                    TempReward-=1
                if(Tempvectorized[i[0]][j] != Tempvectorized[i[1]][j] and Tempvectorized[i[1]][j] in (0,1,2,3)
                   and Tempvectorized[i[0]][j] not in (0,1,2,3)):
                    TempReward-=1
                if(Tempvectorized[i[0]][j] != Tempvectorized[i[1]][j] and Tempvectorized[i[1]][j] not in (0,1,2,3)
                   and Tempvectorized[i[0]][j]  in (0,1,2,3)):
                    TempReward-=1
        
        return TempReward
            
   
    def step(self,currentState,action,GoalAlignment):
        #Check if action is legal 
        #Check if its moving any nucleotide or is it moving a gap
        #action is defined as 1_L,1_R,etc
        #print "Input State is",self.get_sequences_from_state(currentState)
        currentState = self.de_one_hot_encode(currentState.reshape(self.noOfRows,self.noOfCols*5))
        valid,Reward,NewState = self.legalaction(currentState,action)
        if (Reward>=GoalAlignment):
            Reward=self.GetReward(NewState)
            done=True
        #High negative rewards
        elif (Reward!=-40):
            done=False
            Reward=self.GetReward(NewState)
        else:
            done=False
        #print "output state is",self.get_sequences_from_state(currentState)
        NewStateNumpy=np.asarray(NewState)
        NewStateNumpy = self.one_hot_encode(NewStateNumpy.reshape(self.noOfRows*self.noOfCols))
        #if NewStateNumpy.shape==(2,8):
        #    NewStateNumpy=np.reshape(NewStateNumpy,(16,))
        #print NewStateNumpy
        return float(Reward),NewStateNumpy,done,{}
        
        
    def legalaction(self,oldState,action):
        #print "Inside legal action function"
        input_state=oldState
        action=self.ActionSpace[action]
        #print "oldState is",oldState
        Tempvectorized = input_state.reshape(self.noOfRows,self.noOfCols)
        actionindex = int(action.split('_')[0])
        actiondirection = action.split('_')[1]
        valid = False
        i,j=actionindex/self.noOfCols,actionindex%self.noOfCols
        #print i,j,Tempvectorized[i][j]
        if(Tempvectorized[i][j] == 4): #if current index position is a blank,then invalid action
            #print "here"
            return valid,-40,input_state 
        else:
            if(actiondirection =="L"):#if direction is left, go left till the last location check for blank
                #print "Left Action detected"
                for k in reversed(range(0,j)):
                    if(Tempvectorized[i][k] == 4):#if found, then shift all those positions
                        valid = True
                        shiftindex = k
                        break
                if(valid == True):
                    #print shiftindex,j
                    for k in range(shiftindex,j+1):
                        #print k
                        if (k+1<self.noOfCols):
                            Tempvectorized[i][k] = Tempvectorized[i][k+1]
                    #print k
                    Tempvectorized[i][k]=4
                    #print "state is",env.get_sequences_from_state(input_state)
                    #print "temp vectorized",Tempvectorized
                    return valid,self.GetReward(Tempvectorized),Tempvectorized
                else:
                    return valid,-40,input_state
            else:
                #print "Right Action detected"
                for k in (range(j+1,self.noOfCols)):
                    if(Tempvectorized[i][k] == 4):
                        valid = True
                        shiftindex = k
                        break
                if(valid == True):
                    for k in reversed(range(j,shiftindex+1)):
                        if (k-1>=0):
                            Tempvectorized[i][k] = Tempvectorized[i][k-1]
                    Tempvectorized[i][k]=4
                    #print "state is",env.get_sequences_from_state(input_state)
                    #print "temp vectorized",Tempvectorized
                    return valid,self.GetReward(Tempvectorized),Tempvectorized
                else:
                    return valid,-40,input_state
            return valid,0,Tempvectorized

    def de_one_hot_encode(self,oneHotEncodedState):
        """
        Input numpy array of size (self.noOfRows,self.noOfCols,5)
        Returns numpy array of size (self.noOfRows*self.noOfCols)
        """
        returnList = []
        flattenedNumpyArray = oneHotEncodedState.reshape(self.noOfCols*self.noOfRows*5)
        for i in range(0,len(flattenedNumpyArray),5):
            if list(flattenedNumpyArray[i:i+5]).index(1) == 0:
                returnList.append(0)
            if list(flattenedNumpyArray[i:i+5]).index(1) == 1:
                returnList.append(1)
            if list(flattenedNumpyArray[i:i+5]).index(1) == 2:
                returnList.append(2)
            if list(flattenedNumpyArray[i:i+5]).index(1) == 3:
                returnList.append(3)
            if list(flattenedNumpyArray[i:i+5]).index(1) == 4:
                returnList.append(4)
        return np.asarray(returnList)

    def getProbForActionEpsilonGreedy(self,actions,percent):
        actionSorted = sorted(actions,reverse = True)
        p =[]
        Tot = len(actionSorted) - 1
        for ind,i in enumerate(actionSorted):
            if(ind == 0):
                p.append(percent)
            else:
                p.append((1.0-percent)/Tot)
        p /= np.sum(p)
        return p,actionSorted

#---------
class Gym:
    def __init__(self):
        pass 

    def make(self,garbage):
        boardParameters={}
        boardParameters['Rows']=2
        boardParameters['Cols']=8
        env=GymEnv(boardParameters)
        return env

"""
##########################################################
"""
gym=Gym()

def main():
    env = gym.make("RLALIGN")
    with tf.Session() as sess:
        actor = a3c.ActorNetwork(sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE,Rows = env.noOfRows,Cols = env.noOfCols)
        critic = a3c.CriticNetwork(sess, state_dim=S_DIM, learning_rate=CRITIC_LR_RATE,Rows = env.noOfRows,Cols = env.noOfCols)
        saver = tf.train.Saver()
        saver.restore(sess, saved_MODEL)
        #stepsize =[10,15,25,45,75,85,105,125,200,400,800]
        #interval =[10,15,25,45,75,85,105,125,200,400,800]
        stepsize =[10,20,50,100,200]
        interval =[10,20,50,100,200]
        FinalAccPlot =[]
        GlobalAcc = {}
        for inter in PROB:
            GlobalAcc[str(inter)] = []
        for probability in PROB:
            Acc ={}
            for inter in interval:
                Acc[str(inter)] = []
            print "probability",probability
            AccuracyPlot =[]
            for v in stepsize:
                print "current step ", v
                MeanAccuracy =0
                for eps in xrange(1000):
                    obs = env.reset()
                    testSequence=env.reset()
                    seq = {}
                    for i in range(len(testSequence)):
                        seq[i] = testSequence[i].replace("-","")
                    GoalAlignment =0
                    #temp=mafft.mafft_Score(seq,100)
                    #GoalAlignment = temp[0]
                    temp = NW(seq[0],seq[1])
                    GoalAlignment = temp
                    #print "GoalAlignemnt Score:",GoalAlignment
                    #print "alignment", temp[1]
                    #print "Test Sequence",testSequence
                    listOfStates=[]
                    listOfStates.append(testSequence)
                    #print "Test Sequence",testSequence
                    #print actionSpace
                    for i in listOfStates:
                        count=1    
                        StateToGetAction=env.de_one_hot_encode(env.GetStateVector(i))#1*12 shape
                        #print StateToGetAction
                        #StateToGetAction = np.reshape(StateToGetAction,(2,6))
                        testState=np.reshape(env.GetStateVector(i),[env.noOfRows,env.noOfCols*5,1])
                        #print legalStateIndices
                        rew =[]
                        while True:
                            #print "Move ",count
                            legalStates=env.getLegalActions(StateToGetAction,env.ActionSpace)
                            #print legalStates
                            legalStateIndices=[]
                            for j in legalStates:
                                legalStateIndices.append(env.ActionSpace.index(j))
                            #print legalStateIndices

                            #print "Input State", env.get_sequences_from_state(env.de_one_hot_encode(testState))
                            #print testState.shape
                            testState = np.reshape(testState,[1,testState.shape[0],testState.shape[1],testState.shape[2]])
                            prediction=actor.predict(testState)
                            #print prediction
                            #print "Prediction",prediction
                            #print prediction.shape
                            predictionToUse=[]
                            actionPredicted1=np.argmax(prediction)
                            #print "action Predicted over all actions",env.ActionSpace[actionPredicted1]
                            for k in legalStateIndices:
                                predictionToUse.append(prediction[0][k])
                            prob,actions = env.getProbForActionEpsilonGreedy(predictionToUse,probability)
                            #print actions
                            actionPredicted=legalStateIndices[predictionToUse.index(np.random.choice(actions, p=prob))]
                            #actionPredicted=legalStateIndices[np.argmax(predictionToUse)]
                            #print "best action from legal states " + str(env.ActionSpace[legalStateIndices[np.argmax(predictionToUse)]])
                            #print "action Predicted over legal action",env.ActionSpace[actionPredicted]
                            next_sequence=env.step(testState,actionPredicted,GoalAlignment)
                            #print "Next State",env.get_sequences_from_state(env.de_one_hot_encode(next_sequence[1]))
                            #print "reward", next_sequence[0]
                            rew.append(next_sequence[0])
                            StateToGetAction = env.de_one_hot_encode(next_sequence[1])
                            testState=np.reshape(next_sequence[1],[env.noOfRows,env.noOfCols*5,1])
                            count+=1
                            if(next_sequence[0] >=GoalAlignment):
                                Acc[str(v)].append(next_sequence[0]-GoalAlignment)
                                MeanAccuracy+=1
                                break
                            if count==v:
                                Acc[str(v)].append(max(rew)-GoalAlignment)
                            #    if(max(rew)>GoalAlignment):
                                    #print "Wohoo Prediction better than Mafft!"
                            #        MeanAccuracy+=1
                            #    elif(max(rew)==GoalAlignment):
                                    #print "Yaay prediction correct"
                            #        MeanAccuracy+=1
                                #else:
                                    #print "Reward Abs Diff", abs(GoalAlignment - max(rew))
                                #res.append(max(rew) - GoalAlignment)
                                break
                print "Average percentage " + str((MeanAccuracy/1000.0)*100.0)
                AccuracyPlot.append((MeanAccuracy/1000.0)*100.0)
                print AccuracyPlot
            FinalAccPlot.append(AccuracyPlot)
            for inter in interval:
                GlobalAcc[str(probability)].append(Acc[str(inter)])
    
    '''
    plt.subplot(141)
    plt.title('Random Factor 0%')
    sns.boxplot(data=GlobalAcc[str(PROB[3])])
    plt.ylabel("Alignment Score Difference")
    plt.xlabel("No of Steps")
    plt.xticks(range(0, len(interval), 1), interval)
    plt.subplot(142)
    plt.title('Random Factor 10%')
    sns.boxplot(data=GlobalAcc[str(PROB[2])])
    plt.xlabel("No of Steps")
    plt.xticks(range(0, len(interval), 1), interval)
    plt.subplot(143)
    plt.title('Random Factor 20%')
    plt.xlabel("No of Steps")
    sns.boxplot(data=GlobalAcc[str(PROB[1])])
    plt.xticks(range(0, len(interval), 1), interval)
    plt.subplot(144)
    plt.title('Random Factor 30%')
    '''
    plt.ylabel("Alignment Score Difference",fontsize=25)
    plt.xlabel("Number of Steps",fontsize=25)
    sns.boxplot(data=GlobalAcc[str(PROB[0])])
    plt.xticks(range(0, len(interval), 1), interval,fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('MSA_2x8x4.png')

    #plt.clf()
    '''
    plt.plot(stepsize,FinalAccPlot[0],label = "Random Factor 30%")
    plt.plot(stepsize,FinalAccPlot[1],label = "Random Factor 20%")
    plt.plot(stepsize,FinalAccPlot[2],label = "Random Factor 10%")
    plt.plot(stepsize,FinalAccPlot[3],label = "Random Factor 0%")
    #plt.title("Accuracy vs No of steps")
    plt.xlabel("No of Steps")
    plt.ylabel("Accuracy")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    #plt.savefig('AccuracyMSA2Seq10_with10Nucleotide_MAFFT.png')
    '''

if __name__ == '__main__':
    main()
