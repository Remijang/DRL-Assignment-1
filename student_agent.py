# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from lib import PyTorchPolicy

state_size = (5, 5, 2, 4, 2, 2, 2, 2)
action_size = 6
table = PyTorchPolicy(state_size, action_size)

with open("policy_model", 'rb') as f:
    table.policy = pickle.load(f)
    
table.reset_state()

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    
    return table.get_action(obs)

    # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

