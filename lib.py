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
    def __init__(self, state_size, action_size, lr=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.policy = torch.rand(*state_size, action_size)
        self.optimizer = optim.Adam([self.policy], lr=lr)
        # self.loss_fn = nn.CrossEntropyLoss()
        self.station = [(0, 0), (0, 4), (4, 0), (4,4)]
        self.get_passenger = 0
        self.target = 0

    def get_action(self, obs):
        state = self.get_agent_state(obs)
        probs = torch.softmax(self.policy[state], dim=0)[:4]
        '''
        if (state[0], state[1]) == self.station[state[3]]:
            if self.get_passenger == 0 and obs[-2] == True:
                probs[4] += 10
            elif self.get_passenger == 1 and obs[-1] == True:
                probs[5] += 10
        else:
            probs[4] = 0.0
            probs[5] = 0.0
        '''
        action = torch.multinomial(probs, 1).item()
        
        self.update_state(state, obs, action)
        return action

    def get_agent_state(self, obs):
        return (obs[0], obs[1], self.get_passenger, self.target, obs[-6], obs[-5], obs[-4], obs[-3])

    def reset_state(self):
        self.get_passenger = 0
        self.target = 0
        self.cnt = 0

    def update_state(self, state, obs, action):
        if (state[0], state[1]) == self.station[self.target]:
            if self.get_passenger == 0 and obs[-2] == True and action == 4:
                self.get_passenger = 1
                self.target = 0
            else:
                self.target = (self.target + 1) % 4
            self.cnt = self.cnt + 1
        
        if self.get_passenger == 1 and action == 5:
            self.get_passenger = 0
            self.target = 0
            self.cnt = 0