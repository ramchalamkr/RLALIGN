import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import logging
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import a3c
import random
import copy
#import mafft
import itertools

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
                #print j
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
            done=False
        #High negative rewards
        elif (Reward!=-40):
            done=False
            Reward=self.GetReward(NewState)
        else:
            done=False
        NewStateNumpy=np.asarray(NewState)
        NewStateNumpy = self.one_hot_encode(NewStateNumpy.reshape(self.noOfRows*self.noOfCols))
        return float(Reward),NewStateNumpy,done,{}
        
        
    def legalaction(self,oldState,action):
        #print "Inside legal action function"
        input_state=oldState
        action=self.ActionSpace[action]
        Tempvectorized = input_state.reshape(self.noOfRows,self.noOfCols)
        actionindex = int(action.split('_')[0])
        actiondirection = action.split('_')[1]
        valid = False
        i,j=actionindex/self.noOfCols,actionindex%self.noOfCols
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
                    for k in range(shiftindex,j+1):
                        if (k+1<self.noOfCols):
                            Tempvectorized[i][k] = Tempvectorized[i][k+1]
                    Tempvectorized[i][k]=4
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

#-------------------------------------------------------------------------------------
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
S_DIM = 16
A_DIM = 28
EPSILON = 0.01
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 50  # take as a train batch
TRAIN_EPOCH = 5001
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.0001
MODEL_SAVE_INTERVAL = 1000
RAND_RANGE = 1000
SUMMARY_DIR = './results'
MODEL_DIR = './models'
#saved_MODEL = './models/saved_model_eps_18000.ckpt'
saved_MODEL = None

def central_agent(net_params_queues, exp_queues):
    env1 = gym.make("RLALIGN")

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    with tf.Session() as sess, open(SUMMARY_DIR + '/log_central', 'wb') as log_file:

        actor = a3c.ActorNetwork(sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE,
            Rows = env1.noOfRows,Cols = env1.noOfCols)
        critic = a3c.CriticNetwork(sess, state_dim=S_DIM, learning_rate=CRITIC_LR_RATE,
            Rows = env1.noOfRows,Cols = env1.noOfCols)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = saved_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        # while True:  # assemble experiences from agents, compute the gradients
        for ep in xrange(TRAIN_EPOCH): 
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in xrange(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_agents = 0.0 

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in xrange(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch= np.asarray(s_batch),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0

            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)

            for i in xrange(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_batch_len

            log_file.write('Epoch: ' + str(ep) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward_per_agent: ' + str(avg_reward) )
            log_file.write("\n")
            log_file.flush()

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward
            })

            writer.add_summary(summary_str, ep)
            writer.flush()

            if ep % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, MODEL_DIR + "/saved_model_eps_" +
                                       str(ep) + ".ckpt")


def agent(agent_id, net_params_queue, exp_queue):

    env = gym.make("RLALIGN")

    with tf.Session() as sess, open(SUMMARY_DIR + '/log_agent_' + str(agent_id), 'wb') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=S_DIM, action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE,Rows = env.noOfRows,Cols = env.noOfCols)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=S_DIM,
                                   learning_rate=CRITIC_LR_RATE,Rows = env.noOfRows,Cols = env.noOfCols)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        time_stamp = 0
        for ep in xrange(TRAIN_EPOCH): 
            obs = env.reset()
            seq = {}
            for i in range(len(obs)):
                seq[i] = obs[i].replace("-","")
            GoalAlignment =0
            #temp=mafft.mafft_Score(seq,agent_id)
            temp = NW(seq[0],seq[1])
            GoalAlignment = temp
            #print seq
            #print temp[1]
            #print GoalAlignment
            stateOriginal = env.GetStateVector(obs)
            stateOriginal = np.reshape(stateOriginal, [env.noOfRows,env.noOfCols*5,1])
            seq=env.get_sequences_from_state(env.de_one_hot_encode(stateOriginal))

            s_batch = []
            a_batch = []
            r_batch = []

            for step in xrange(TRAIN_SEQ_LEN):

                s_batch.append(stateOriginal)

                #action_prob = actor.predict(np.reshape(obs, (1, S_DIM)))
                #print env.get_sequences_from_state(env.de_one_hot_encode(stateOriginal))
                stateBuffer=copy.deepcopy(stateOriginal)
                stateBuffer = np.reshape(stateOriginal,[1,stateOriginal.shape[0],stateOriginal.shape[1],stateOriginal.shape[2]])
                StateToGetAction=env.de_one_hot_encode(stateOriginal)
                action_prob = actor.predict(stateBuffer)
                action_cumsum = np.cumsum(action_prob)
                a=(action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                #a = np.random.choice(A_DIM, p=action_prob[0])
                #action_prob = actor.predict(stateBuffer)
                #print action_prob
                #action_prob[0] = action_prob[0] - np.finfo(np.float32).epsneg
                #histogram = np.random.multinomial(1, action_prob[0])
                #a = int(np.nonzero(histogram)[0])

                #if random.random() < EPSILON:
                #    a= random.randint(0, A_DIM-1)
                #else:
                #    a = np.random.choice(A_DIM, p=action_prob[0])
                    

                action_vec = np.zeros(A_DIM)
                action_vec[a] = 1
                a_batch.append(action_vec)
                rew,s_, done, info = env.step(stateOriginal,a,GoalAlignment)
                #print rew
                s_ = np.reshape(s_, [s_.shape[0],s_.shape[1],1])
                stateOriginal = s_
                r_batch.append(rew)

            ind = r_batch.index(max(r_batch))
            rewardDiff = max(r_batch) - GoalAlignment
            reachedGoal = "No"
            if(max(r_batch) >= GoalAlignment):
                reachedGoal = "Yes"
            if not done:
            #    ind = r_batch.index(max(r_batch))
                done = True
                exp_queue.put([s_batch[0:ind+1], a_batch[0:ind+1], r_batch[0:ind+1], done])
                log_file.write('seq '+ str(seq) +' epoch ' + str(ep) + ' reward ' + str(np.sum(r_batch[0:ind+1])) + ' step ' + str(len(r_batch[0:ind+1])) + ' AlignmentDiff '+ str(rewardDiff) + ' reachedGoal ' + reachedGoal)
                log_file.write("\n")
                log_file.flush()
            #exp_queue.put([s_batch, a_batch, r_batch, done])
            actor_net_params, critic_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)
            critic.set_network_params(critic_net_params)

            

            


def main():

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in xrange(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a central agent and multiple agent processes
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in xrange(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in xrange(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
