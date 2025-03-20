import gym
import numpy as np
import importlib.util
import time
import random
from simple_custom_taxi_env import SimpleTaxiEnv
#from student_agent import get_action
import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchPolicy:
    def __init__(self):
        self.state_size = (2,  2, 2, 2, 2,  3, 3)
        self.action_size = 4
        self.get_passenger = 0
        self.target = 0
        self.policy = torch.zeros(*(self.state_size), self.action_size)
        self.reset = 0
        random.seed(48763)

    def get_action(self, obs):
        state = self.get_agent_state(obs)
        probs = torch.softmax(self.policy[state], dim=0)
        action = torch.multinomial(probs, 1).item()

        if (obs[0], obs[1]) in self.station:
            if self.get_passenger == 0 and obs[-2] == 1:
                action = 4
            elif self.get_passenger == 1 and obs[-1] == 1:
                action = 5
        
        self.update_state(obs, action)
        return action

    def get_agent_state(self, obs):
        trow, tcol = self.station[self.target]
        row, col = obs[0], obs[1]
        if trow < row:
            roww = 0
        elif trow == row:
            roww = 1
        else:
            roww = 2
        if tcol < col:
            coll = 0
        elif tcol == col:
            coll = 1
        else:
            coll = 2
        return (self.get_passenger, 
                obs[-5], obs[-6], obs[-4], obs[-3],
                roww, coll)

    def reset_state(self, obs):
        self.get_passenger = 0
        self.target = 0
        self.des = -1
        self.station = []
        self.station.append((obs[2], obs[3]))
        self.station.append((obs[4], obs[5]))
        self.station.append((obs[6], obs[7]))
        self.station.append((obs[8], obs[9]))
        self.reset = 1

    def update_state(self, obs, action):
        if (obs[0], obs[1]) in self.station:
            if self.get_passenger == 0 and obs[-2] == 1 and action == 4:
                self.get_passenger = 1
                self.target = 0
                if self.des >= 0:
                    self.target = self.des
            elif obs[-1] == 1:
                self.des = self.station.index((obs[0], obs[1]))
            
            if (obs[0], obs[1]) == self.station[self.target]:
                self.target = (self.target + 1) % 4