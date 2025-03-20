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
        self.policy = torch.zeros(*state_size, action_size)
        self.optimizer = optim.Adam([self.policy], lr=lr)
        # self.loss_fn = nn.CrossEntropyLoss()
        self.get_passenger = 0
        self.reset = 0

    def get_action(self, obs):
        state = self.get_agent_state(obs)
        probs = torch.softmax(self.policy[state], dim=0)

        neighbor = [obs[-5], obs[-6], obs[-4], obs[-3]]
        
        for i in range(0, 4, 1):
            if neighbor[i] == 1:
                probs[i] = 0.0

        action = torch.multinomial(probs, 1).item()

        if (state[0], state[1]) in self.station:
            if self.get_passenger == 0 and obs[-2] == True:
                action = 4
            elif self.get_passenger == 1 and obs[-1] == True:
                action = 5
        
        self.update_state(state, obs, action)
        return action

    def get_agent_state(self, obs):
        return (obs[0], obs[1], self.get_passenger, obs[-6], obs[-5], obs[-4], obs[-3])

    def reset_state(self, obs):
        self.get_passenger = 0
        self.station = []
        self.station.append((obs[2], obs[3]))
        self.station.append((obs[4], obs[5]))
        self.station.append((obs[6], obs[7]))
        self.station.append((obs[8], obs[9]))
        self.reset = 1

    def update_state(self, state, obs, action):
        if (state[0], state[1]) in self.station:
            if self.get_passenger == 0 and obs[-2] == True and action == 4:
                self.get_passenger = 1