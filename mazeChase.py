from logging import exception, raiseExceptions
import string
from time import sleep
from turtle import isdown
from typing import Tuple
import gym 
import numpy as np
from gym import spaces
import random
import logging
import math
import copy

from scipy import rand

"""
In this game, you will control an agent(Blue circle) to chase a pig(pink circle) and avoiding be captured by enemy(red circle) 
and you can't move to obstacle(brown block)

action space = {0,1,2,3}
0:up
1:down
2:left
3:right
"""

class mazeChase(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        self.grid_word_width = 6
        self.grid_word_height = 6
        self.grid_pixel = 75        # how much pixel a grid occupied
        self.pig_pos = (0,0)
        self.agent_pos = (0,0)
        self.enemy_pos = (0,0)
        self.obstacle_pos = list()
        self.viewer = None
        self.action_space = (0,1,2,3)   # action space
        self.prob_pig_stationaty = 0.3  # pig will have this probability stay stationary (incase agent never chase the pig)
        # this parameter you can modify
        self.obstacle_num = 4
        self.pig_policy_level = 0   # 0: pig move randomly, 1: pig try to keep away from agent
        self.enemy_policy_level = 0 # 0: enemy move randomly, 1: enemy try to hunt agent  
        self.pig_move_prob = [0.25,0.25,0.25,0.25]
        self.enemy_move_prob = [0.25,0.25,0.25,0.25]


    def seed(self,seed):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self):
        self.obstacle_pos = list()
        # init obstacle pos randomly
        # obstacle_x = random.sample(range(0,self.grid_word_width-1),self.obstacle_num)
        # obstacle_y = random.sample(range(0,self.grid_word_height-1),self.obstacle_num)
        # for i in range(self.obstacle_num):
        #     self.obstacle_pos.append((obstacle_x[i],obstacle_y[i]))
        # init obstacle pos fixed
        self.obstacle_pos = [(2,2),(2,3),(2,4),(3,4)]
        self.pig_pos = (random.randint(0,self.grid_word_width-1),random.randint(0,self.grid_word_height-1))
        while self.pig_pos in self.obstacle_pos:    # prevent initial pig position in the obstacle 
            self.pig_pos = (random.randint(0,self.grid_word_width-1),random.randint(0,self.grid_word_height-1))
        self.enemy_pos = (random.randint(0,self.grid_word_width-1),random.randint(0,self.grid_word_height-1))
        while self.enemy_pos in self.obstacle_pos or self.enemy_pos == self.pig_pos:
            self.enemy_pos = (random.randint(0,self.grid_word_width-1),random.randint(0,self.grid_word_height-1))
        self.agent_pos = (random.randint(0,self.grid_word_width-1),random.randint(0,self.grid_word_height-1))
        while self.agent_pos in self.obstacle_pos or self.agent_pos == self.enemy_pos or self.agent_pos == self.pig_pos:
            self.agent_pos = (random.randint(0,self.grid_word_width-1),random.randint(0,self.grid_word_height-1))
        state = (self.agent_pos,self.pig_pos,self.enemy_pos)
        return state

    def render(self,mode='human'):
        screen_width = self.grid_word_width * self.grid_pixel
        screen_height = self.grid_word_height * self.grid_pixel
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width,screen_height)
            # draw background color
            green_l = [i*(1/self.grid_word_height)+0.5 for i in range(self.grid_word_height)]
            for i in range(self.grid_word_height):
                color_block = rendering.make_polygon([(0,i*self.grid_pixel),(0,(i+1)*self.grid_pixel),(screen_width,(i+1)*self.grid_pixel),(screen_width,i*self.grid_pixel)])
                color_block.set_color(0,green_l[i],0)
                self.viewer.add_geom(color_block)
            
            # draw the obstacle
            # print("Obstacle position:{}".format(self.obstacle_pos))
            for i in self.obstacle_pos:
                x = i[0]*self.grid_pixel
                y = i[1]*self.grid_pixel
                obstacle_block = rendering.make_polygon([(x,y),(x,y+self.grid_pixel),(x+self.grid_pixel,y+self.grid_pixel),(x+self.grid_pixel,y)])
                obstacle_block.set_color(0.54,0.31,0.1)
                self.viewer.add_geom(obstacle_block)
            
            # draw the grid line
            for i in range(self.grid_word_width):
                line = rendering.Line((i*self.grid_pixel,0),(i*self.grid_pixel,screen_height))
                line.set_color(0,0,0)
                self.viewer.add_geom(line)
            for i in range(self.grid_word_height):
                line = rendering.Line((0,i*self.grid_pixel),(screen_width,i*self.grid_pixel))
                line.set_color(0,0,0)
                self.viewer.add_geom(line)
            
            # draw pig
            # self.pig = rendering.make_circle(int(self.grid_pixel/3))
            # self.pig.set_color(255/255,192/255,203/255)
            # self.pig_move = rendering.Transform()
            # self.pig.add_attr(self.pig_move)
            # self.viewer.add_geom(self.pig)

            from os import path
            # pig img
            fname = path.join(path.dirname(__file__),"assets/pig.png")
            self.pig_img = rendering.Image(fname, 65, 65)
            self.pig_imgtrans = rendering.Transform()
            self.pig_img.add_attr(self.pig_imgtrans)

            # draw agent
            # self.agent = rendering.make_circle(int(self.grid_pixel/2.5))
            # self.agent.set_color(0,192/255,203/255)
            # self.agent_move = rendering.Transform()
            # self.agent.add_attr(self.agent_move)
            # self.viewer.add_geom(self.agent)
            fname = path.join(path.dirname(__file__),"assets/agent.png")
            self.agent_img = rendering.Image(fname, 70, 70)
            self.agent_imgtrans = rendering.Transform()
            self.agent_img.add_attr(self.agent_imgtrans)

            # draw enemy
            # self.enemy = rendering.make_circle(int(self.grid_pixel/2.2))
            # self.enemy.set_color(1,0,0)
            # self.enemy_move = rendering.Transform()
            # self.enemy.add_attr(self.enemy_move)
            # self.viewer.add_geom(self.enemy)
            fname = path.join(path.dirname(__file__),"assets/enemy.png")
            self.enemy_img = rendering.Image(fname, 70, 70)
            self.enemy_imgtrans = rendering.Transform()
            self.enemy_img.add_attr(self.enemy_imgtrans)


        self.viewer.add_onetime(self.pig_img)
        self.viewer.add_onetime(self.enemy_img)
        self.viewer.add_onetime(self.agent_img)
        # move the pig, enemy, agent
        x_pig_pixel = int((self.pig_pos[0]+0.5)*self.grid_pixel)        # add 0.5 is for pig could be in center of the grid
        y_pig_pixel = int((self.pig_pos[1]+0.5)*self.grid_pixel) 
        # self.pig_move.set_translation(x_pig_pixel,y_pig_pixel)
        self.pig_imgtrans.set_translation(x_pig_pixel,y_pig_pixel)

        x_agent_pixel = int((self.agent_pos[0]+0.5)*self.grid_pixel)        # add 0.5 is for pig could be in center of the grid
        y_agent_pixel = int((self.agent_pos[1]+0.5)*self.grid_pixel) 
        # self.agent_move.set_translation(x_agent_pixel,y_agent_pixel)
        self.agent_imgtrans.set_translation(x_agent_pixel,y_agent_pixel)

        x_enemy_pixel = int((self.enemy_pos[0]+0.5)*self.grid_pixel)        # add 0.5 is for pig could be in center of the grid
        y_enemy_pixel = int((self.enemy_pos[1]+0.5)*self.grid_pixel) 
        # self.enemy_move.set_translation(x_enemy_pixel,y_enemy_pixel)
        self.enemy_imgtrans.set_translation(x_enemy_pixel,y_enemy_pixel)

        return self.viewer.render(return_rgb_array=mode)

    def outOfWord(self,pos):
        """
        return true, if pos is out of grid word
        e.g. if grid_word_width = 10, then x should be in range [0,9]
        """
        if pos[0] < 0 or pos[0] >= self.grid_word_width or pos[1]<0 or pos[1]>=self.grid_word_height:
            return True     # out of word
        else:
            return False

    def move(self,old_pos,action):
        """calculate new postion responsed to the input action, and return the new postion
        if the new postion is out of word or hit the obstacle, return old position

        Args:
            old_pos (_type_): old object's position, must be (x,y)
            action (int): object's action

        Raises:
            Exception: if action is out of action space, raise exception

        Returns:
            Tuple: return the new position of object
        """
        new_pos = copy.deepcopy(old_pos)
        if action not in self.action_space:
            raise Exception('Input action out of action space')
        elif action == 0:   # up
            new_pos = (old_pos[0],old_pos[1]+1)
        elif action == 1:   # down
            new_pos = (old_pos[0],old_pos[1]-1)
        elif action == 2:   # left
            new_pos = (old_pos[0]-1,old_pos[1])
        elif action == 3:   # right
            new_pos = (old_pos[0]+1,old_pos[1])
        else:
            print("error")
        if new_pos in self.obstacle_pos or self.outOfWord(new_pos): # hit obstacle or out of grid word
            return old_pos
        else:
            return new_pos
    
    def distance(sefl,pos1:Tuple,pos2:Tuple)->float:
        """calculate the distance between two pos

        Args:
            sefl (_type_): _description_
            pos1 (Tuple): position1
            pos2 (Tuple): position2

        Returns:
            float: the distance between position1 and position2
        """
        return math.sqrt(math.pow(pos1[0]-pos2[0],2)+math.pow(pos1[1]+pos2[1],2))

    def pig_policy(self):
        if self.pig_policy_level == 0:
            # return self.action_space[random.randint(0,len(self.action_space)-1)]
            return probabilisticSample(self.pig_move_prob)
        else:
            # this case wil make pig go to corner of the word every time, and make game too easy
            d = []
            for a in self.action_space:     # try every action possible and find one which can lead maximum distance between me(pig) and agent
                new_pig_pos = self.move(self.pig_pos,a)
                d.append(self.distance(self.agent_pos,new_pig_pos))
            return np.argmax(d)
    def enemy_policy(self):
        if self.enemy_policy_level == 0:
            # return self.action_space[random.randint(0,len(self.action_space)-1)]
            return probabilisticSample(self.enemy_move_prob)
        else:
            d = []
            for a in self.action_space:     # try every action possible and find one which can lead minimum distance between me(enemy) and agent
                new_enemy_pos = self.move(self.enemy_pos,a)
                d.append(self.distance(self.agent_pos,new_enemy_pos))
            return np.argmin(d)
    
    def step(self, action:int)->Tuple:
        """gym env step function, achieve the state transition

        Args:
            action (int): the action of agent to do

        Returns:
            Tuple: (next state,reward,done,msg)
        """
        self.agent_pos = self.move(self.agent_pos,action)
        action_pig = self.pig_policy()
        self.pig_pos = self.move(self.pig_pos,action_pig)
        while(self.pig_pos == self.enemy_pos):  # prevent pig and enemy fall in same position
            action_pig = self.pig_policy()
            self.pig_pos = self.move(self.pig_pos,action_pig)
        action_enemy = self.enemy_policy()
        self.enemy_pos = self.move(self.enemy_pos,action_enemy)
        while(self.enemy_pos==self.pig_pos):    # prevent pig and enemy fall in same position
            action_enemy = self.enemy_policy()
            self.enemy_pos = self.move(self.enemy_pos,action_enemy)
        reward = 0
        msg = ''
        if self.isDone() == 'pass':     # if game is not over
            done = False
            reward = -self.distance(self.agent_pos,self.pig_pos)  
        else:       # agent chase the pig or be captured by enemy, game over!
            done = True
            if self.isDone() == 'chase':
                reward = 100
                msg = 'agent success'
            else:
                reward = -100
                msg = 'agent die'
        state_next = (self.agent_pos,self.pig_pos,self.enemy_pos)
        return state_next,reward,done,msg

    def isDone(self)->string:
        """recognize the situation

        Returns:
            string: return 'captured' if enemy capture the agent, 'chase' if agent get the pig, 'pass' if no one win the game
        """
        if self.agent_pos == self.enemy_pos:
            # agent be captured by enemy
            return 'captured'
        elif self.agent_pos == self.pig_pos:
            # agent chase the pig successfully
            return 'chase'
        else:
            return 'pass'
    
    def getStateSpace(self)->list:
        """return the whole state space
            the length of state space is ((grid_width x grid_height)^2-obstacle_nums)^3
        Returns:
            list: the list of state space. e.g. ((0, 0), (0, 2), (3, 1)) mean agent position=(0, 0), pig position=(0, 2) and enemy position=(3, 1)
            
        """
        all_available_posi = list()
        for i in range(0,self.grid_word_width):
            for j in range(0,self.grid_word_height):
                p = (i,j)
                if p not in self.obstacle_pos:
                    all_available_posi.append(p)
        state_space = list()
        for agent_pos in all_available_posi:
            for pig_pos in all_available_posi:
                for enemy_pos in all_available_posi:
                    state_space.append((agent_pos,pig_pos,enemy_pos))
        return state_space
        
    def smartAI(self)->int:
        val = list()  
        for action in self.action_space:
            agent_new_pos = self.move(self.agent_pos,action)
            val.append(self.distance(self.pig_pos,agent_new_pos))

    def randomSampleAction(self) ->int:
        """return a action from action space randomly

        Returns:
            int: randomly sampled action
        """
        return self.action_space[random.randint(0,len(self.action_space)-1)]

def probabilisticSample(prob:list)->int:
    """按照概率采样

    Args:
        prob (list): 需要采样的概率,例如输入[0.8,0.15,0.05],那么返回0的概率就是80%,返回1的概率是15%,返回2的概率是0.05

    Returns:
        int: sample result
    """
    assert np.sum(prob)>0.999 and np.sum(prob)<1.001,"sum of probably is not 1"
    if len(prob) == 0:  # prob = [1]
        return 0
    u = random.random()
    flag = prob[0]
    if u < flag:
        return 0
    for i in range(1,len(prob)):
        if u > flag and u < flag+prob[i]:
            return i
        else:
            flag += prob[i]
# #test probabilisticSample
# sampleCount = np.array([0,0,0])
# for i in range(100000):
#     p = probabilisticSample([0.05,0.8,0.15])
#     sampleCount[p] += 1
# print(sampleCount/100000)


if __name__ == '__main__':
    episode = 10
    max_step = 200
    env = mazeChase()
    env.reset()
    for i in range(episode):
        env.reset()
        for i in range(max_step):
            a = random.randint(0,3)
            s_,r,done,msg = env.step(a)
            env.render()
            if done:
                print("done,{}".format(msg))
                sleep(1)
                break
            sleep(0.05)
        # print(env.agent_pos,env.enemy_pos,env.pig_pos)