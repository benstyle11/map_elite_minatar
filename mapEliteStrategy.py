
from archive import *
from Agent import *

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




def pos(inputs):
    return inputs/np.sum(inputs)



def map_elite_strategy(game = "breakout", NUM_FRAMES = 1000, MAX_EVALS = 5000):

    env = Environment("breakout", sticky_action_prob=0.0, random_seed=0)
    env.reset()

    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()
    policy_net = Network(in_channels, num_actions).to(device)
    genes = policy_net.get_params()

    taille_genes = len(genes)
    genes = np.random.randn(taille_genes)

    policy_net.set_params(genes)


    archive = {}

    for i in range(MAX_EVALS):
        finess, behaviour = play(policy_net,game)
        specie = Species(genes, behaviour, fitness)
        addToArchive(archive, specie)

        gene = mutate(archive)





