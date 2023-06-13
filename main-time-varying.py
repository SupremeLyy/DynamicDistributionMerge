from random import random
from tensorboardX import SummaryWriter
from function1 import distributionStoreUnit
import random
import numpy as np
import os
import time
import copy
import joblib
import pickle

startTime = time.localtime(time.time())
time_tag = "{}.{}-{}:{}".format(startTime.tm_mon,startTime.tm_mday,startTime.tm_hour,startTime.tm_min)
root = '.'   # root path

expLogDir = '{}/runlog/Exp[{}]/'.format(root,time_tag)  # store running log curve
expModel = '{}/model/Exp[{}]/'.format(root,time_tag)
os.mkdir(expLogDir)
os.mkdir(expModel)

################################ parameter ################################

alpha = 0.8
gamma = 0.9
epsilon = 0.8
epsilonDecayStep = 3000     # epsilon decay every epsilonDecayStep episode
epsilonDecayRatio = 0.95
render = False
trainEpisode = 260000+1 
demonstratorEpisode = int(trainEpisode*0.5)
maxStep = 1000
sampleTrajectoryLength = 30000
transitionProbVaryingTime = 40000       # how frequent the environment change
sampleTrajectoryErrorProb = 0.1
OVERLAP_THRESHOLD = 0.45
EXP_ENV = 'mazeChase'
REWARD_NOISE = True       
environmentChange = True
mergedVarScaler = 2        
parameter = open("{}/parameter.txt".format(expLogDir),'w')
parameter.write("Exp Time:{}\n".format(time_tag))
parameter.write("alpha = {}\n".format(alpha))
parameter.write("gamma = {}\n".format(gamma))
parameter.write("epsilon = {}\n".format(epsilon))
parameter.write("epsilonDecayStep = {}\n".format(epsilonDecayStep))
parameter.write("epsilonDecayRatio = {}\n".format(epsilonDecayRatio))
parameter.write("trainEpisode = {}\n".format(trainEpisode))
parameter.write("demonstratorEpisode = {}\n".format(demonstratorEpisode))
parameter.write("maxStep = {}\n".format(maxStep))
parameter.write("sampleTrajectoryLength = {}\n".format(sampleTrajectoryLength))
parameter.write("sampleTrajectoryErrorProb = {}\n".format(sampleTrajectoryErrorProb))
parameter.write("transitionProbVaryingTime = {}\n".format(transitionProbVaryingTime))
parameter.write("OVERLAP_THRESHOLD = {}\n".format(OVERLAP_THRESHOLD))
parameter.write("Experiment:{}\n".format(EXP_ENV))      
parameter.write("Reward Noise:{}\n".format(REWARD_NOISE))     
parameter.write("mergedVarScaler:{}\n".format(mergedVarScaler)) 
parameter.write("environmentChange:{}\n".format(environmentChange))
parameter.close()
################################ parameter ################################


################ time varying MDP parameter ##############

# the list of moving probability
pigMoveProb = list()
enemyMoveProb = list()

# [0] is equal in every directions
pigMoveProb.append([0.25,0.25,0.25,0.25])
enemyMoveProb.append([0.25,0.25,0.25,0.25])

moveProbPointer = 0       

for i in range(30):         
    pigMoveProb.append(np.random.dirichlet(np.ones(4),size=1)[0].tolist())          # dirichlet distribution
    enemyMoveProb.append(np.random.dirichlet(np.ones(4),size=1)[0].tolist())

from mazeChase import mazeChase
env = mazeChase()
env.reset()


################ Basline 1 Run Q-Learning Begin ################

# set the pig and enemy moving probability
env.pig_move_prob = [0.25,0.25,0.25,0.25]      
env.enemy_move_prob = [0.25,0.25,0.25,0.25]

# initialize Q table
qTable = dict()
for s in env.getStateSpace():
    qTable[s] = np.zeros(len(env.action_space))
    
# init SummaryWriter
os.mkdir("{}/basline_Q_Learning".format(expLogDir))
writer = SummaryWriter(log_dir="{}/basline_Q_Learning".format(expLogDir))

average_episode_reward = 0
for episode in range(1,demonstratorEpisode):
    state = env.reset()
    done = False
    episode_reward = 0
    if episode % epsilonDecayStep == 0:      # exploration rate decay
        epsilon = epsilon * epsilonDecayRatio
        writer.add_scalar('epsilon',epsilon,episode)
    for step in range(maxStep):
        if random.random() < epsilon:   
            action = env.randomSampleAction()       # exploration
        else:
            action = np.argmax(qTable[state])       # exploitation
        state_next,reward,done,msg = env.step(action)
        # add some noise to reward. to imitate the mearuse noise on the state
        # Noise ~ N(0,1)
        if REWARD_NOISE:
            reward += random.random()
        # do Q learning
        qTable[state][action] = qTable[state][action] + alpha*(reward + gamma*np.max(qTable[state_next]) - qTable[state][action])
        if render:
            env.render()
            time.sleep(0.15)
        state = state_next
        episode_reward += reward
        if done:
            # print("done, step:{}, msg:{}".format(step,msg))
            break
    average_episode_reward += episode_reward
    writer.add_scalar('Episode reward',episode_reward,episode)
    if episode % 250 == 0:
        writer.add_scalar('Ave_Episode_reward',average_episode_reward/250,episode)
        average_episode_reward = 0
        print("Q Learning: Episode:{}, total reward:{},msg:{}".format(episode,episode_reward,msg),end='\r')
joblib.dump(qTable,'{}/Baseline_QLearning_model.m'.format(expModel))
print("Basline 1 Run Q-Learning Finished !")
################ Basline 1 Run Q-Learning End ################

################ Test demonstrator performance Begin ################
env.reset()
moveProbPointer = 0
env.pig_move_prob = pigMoveProb[moveProbPointer]
env.enemy_move_prob = enemyMoveProb[moveProbPointer]

# record demonstrator's performance curve
os.mkdir("{}/DemonstratorPerformance".format(expLogDir))

writer = SummaryWriter(log_dir="{}/DemonstratorPerformance".format(expLogDir))
average_episode_reward = 0
for episode in range(1,trainEpisode):
    state = env.reset()
    done = False
    episode_reward = 0
    if episode % transitionProbVaryingTime == 0:    # transition probability varying with time
        moveProbPointer += 1
        if moveProbPointer == len(pigMoveProb):     # incase this pointer out of range
            moveProbPointer = 0 
        env.pig_move_prob = pigMoveProb[moveProbPointer]
        env.enemy_move_prob = enemyMoveProb[moveProbPointer]
    for step in range(maxStep):
        if random.random() < sampleTrajectoryErrorProb:
            # imitate the demonstrator make mistakes 
            action = env.randomSampleAction()
        else:
            action = np.argmax(qTable[state])
        state_next,reward,done,msg = env.step(action)
        if render:
            env.render()
            time.sleep(0.15)
        state = state_next
        episode_reward += reward
        if done:
            # print("done, step:{}, msg:{}".format(step,msg))
            break
    average_episode_reward += episode_reward
    writer.add_scalar('Episode reward',episode_reward,episode)
    if episode % 250 == 0:
        writer.add_scalar('Ave_Episode_reward',average_episode_reward/250,episode)
        average_episode_reward = 0
        print("demonstrator test: Episode:{}, total reward:{},msg:{}".format(episode,episode_reward,msg),end='\r')
################ Test demonstrator performance End ################

################ Sample Trajectory Begin ################
trajectory = list()     # init empty trajectory
env.reset()
env.pig_move_prob = [0.25,0.25,0.25,0.25]       # demonstrator在老环境[0.25,0.25,0.25,0.25]里采集demonstration trajectory
env.enemy_move_prob = [0.25,0.25,0.25,0.25]
for i in range(0,sampleTrajectoryLength):
    state = env.reset()
    done = False
    episode_reward = 0
    ls = list()
    for step in range(maxStep):
        if random.random() < sampleTrajectoryErrorProb:
            # 模拟人在演示的时候出错
            action = env.randomSampleAction()
        else:
            # 理智的选择动作
            action = np.argmax(qTable[state])
        state_next,reward,done,msg = env.step(action)
        if render:
            env.render()
            time.sleep(0.15)
        # Noise ~ N(0,1)
        if REWARD_NOISE:
            reward += random.random()
        ls.append((state,action,reward,state_next))
        state = state_next
        episode_reward += reward
        if done:
            # print("done, step:{}, msg:{}".format(step,msg))
            break
    print("Sample Trajectory : {}/{}".format(i,sampleTrajectoryLength),end='\r')
    trajectory.append(ls)
print("Sample Trajectory Finished!")


env.reset()
epsilon = 0.8 # epsilon is decayed before, so we must set it back

moveProbPointer = 0
env.pig_move_prob = pigMoveProb[moveProbPointer]
env.enemy_move_prob = enemyMoveProb[moveProbPointer]



    
os.mkdir("{}/newIdea".format(expLogDir))
writer = SummaryWriter(log_dir="{}/newIdea".format(expLogDir))

agentExploreDist = dict()   # agentExploreDist only store actually sampled value:(r+gamma*Q(s',a)) by agent exploring env

for s in env.getStateSpace():
    agentExploreDist[s] = [distributionStoreUnit() for _ in (env.action_space)]

priorDist = dict()
for s in env.getStateSpace():
    priorDist[s] = [distributionStoreUnit() for _ in (env.action_space)]

# learn priorDist
for epoch in range(0,30):  # 遍历trajectory 30遍
    for T in trajectory:
        for (s,a,r,s_) in T:
            priorDist[s][a].add( r + gamma*np.max([v.mean for v in priorDist[s_] ]) )
    print("new idea learn Trajectory at epoch {}.".format(epoch),end='\r')


from function1 import calculateOverlap_scipy

average_episode_reward = 0
for episode in range(1,trainEpisode):
    state = env.reset()
    done = False
    episode_reward = 0
    if episode % epsilonDecayStep == 0:      # exploration rate decay
        epsilon = epsilon * epsilonDecayRatio
    if episode % transitionProbVaryingTime == 0:    # transition probability varying with time
        moveProbPointer += 1
        if moveProbPointer == len(pigMoveProb):     # incase this pointer out of range
            moveProbPointer = 0 
        env.pig_move_prob = pigMoveProb[moveProbPointer]
        env.enemy_move_prob = enemyMoveProb[moveProbPointer]
    for step in range(maxStep):
        if random.random() < epsilon:   # epsilon greedy exploration
            action = env.randomSampleAction()
        else:
            action = np.argmax([v.mean for v in priorDist[state]])
        # do action
        state_next,reward,done,msg = env.step(action)
        if REWARD_NOISE:
            reward += random.random()
        agentExploreDist[state][action].add( reward + gamma*np.max([v.mean for v in priorDist[state_next] ]) )
        if agentExploreDist[state][action].n > 3:   
            # measure the distance between 
            overlap = calculateOverlap_scipy(priorDist[state][action].mean,priorDist[state][action].var,agentExploreDist[state][action].mean,agentExploreDist[state][action].var)
            if overlap > OVERLAP_THRESHOLD:
                # the distance is within threshold, then do the merge
                priorMean = priorDist[state][action].mean
                priorVar = priorDist[state][action].var
                observeMean = agentExploreDist[state][action].mean
                observeVar = agentExploreDist[state][action].var
                merge_mean = (priorDist[state][action].mean*agentExploreDist[state][action].var + agentExploreDist[state][action].mean*priorDist[state][action].var)/(agentExploreDist[state][action].var+priorDist[state][action].var)
                merge_var = mergedVarScaler*agentExploreDist[state][action].var * priorDist[state][action].var/(agentExploreDist[state][action].var + priorDist[state][action].var)
                # update to prior
                priorDist[state][action].mean = merge_mean
                priorDist[state][action].var = merge_var
            else:
                # reject the prior knowledge and use agent's exploration instead
                priorDist[state][action].mean = agentExploreDist[state][action].mean
                priorDist[state][action].var = agentExploreDist[state][action].var
        else:
            pass            # 不进行learning
        if render:
            env.render()
            time.sleep(0.15)
        state = state_next
        episode_reward += reward
        if done:
            # print("done, step:{}, msg:{}".format(step,msg))
            break
    average_episode_reward += episode_reward
    writer.add_scalar('Episode reward',episode_reward,episode) 
    if episode % 250 == 0:
        writer.add_scalar('Ave_Episode_reward',average_episode_reward/250,episode)
        average_episode_reward = 0
        print("newIdea: Episode:{}, total reward:{},msg:{}".format(episode,episode_reward,msg),end='\r')
        
joblib.dump(priorDist,'{}/ourMethodModel.m'.format(expModel))
