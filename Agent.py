import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time
import random, argparse, logging, os
from collections import namedtuple
from minatar import Environment
import matplotlib.pyplot as plt
import numpy as np


NUM_FRAMES = 1000
MAX_EVALS = 5000
device = "cpu"

class Network(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(Network, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 4
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 32 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 4
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=32)

        # Output layer:
        self.output = nn.Linear(in_features=32, out_features=num_actions)

    def forward(self, x):
        x = f.relu(self.conv(x))
        x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))
        return self.output(x)

    def set_params(self, params):
        a = torch.tensor(params, device=device).float()
        torch.nn.utils.vector_to_parameters(a, self.parameters())

    def get_params(self):
        with torch.no_grad():
            params = self.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.to(device).numpy()



def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


def distBall(state):

    ballx = np.where(state[:,:,1] == 1)[0][0]

    pos_player = np.where(state[:,:,0] == 1)[0][0]
    return  np.abs(ballx - pos_player)






def play(policy_net,game = "breakout", display=False):
    env = Environment(game, sticky_action_prob=0.0, random_seed=0)
    env.reset()
    is_terminated = False
    total_reward = 0.0
    t = 0

    behaviour_act = np.zeros(env.num_actions()) #The behaviour of the agent
    behaviour_dist = 0
    n = 0.
    largeur = len(env.state())

    while (not is_terminated) and t < NUM_FRAMES:
        s = get_state(env.state())
        with torch.no_grad():
            action = torch.argmax(policy_net(s))

        behaviour_act[action] += 1 ## add one to the corresponding behaviour

        if game == "breakout":
            behaviour_dist += distBall(env.state())
        n += 1

        reward, is_terminated = env.act(action)
        total_reward += reward
        t += 1
        if display:
            env.display_state(1)


    behaviour_dist /= n
    behaviour_act /= n

    return total_reward, (behaviour_act,behaviour_dist)




if __name__ == "main":
    env = Environment("breakout", sticky_action_prob=0.0, random_seed=0)
    env.reset()

    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()
    policy_net = Network(in_channels, num_actions).to(device)

















