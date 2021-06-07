


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

from archive import *
from Agent import *




def pos(behaviour):

    staticite = behaviour[0]/np.sum(behaviour) #normalize the behaviour by the nb of inputs

    npoints = 9

    return np.floor(npoints*staticite) #np.floor(npoints*staticite)



def map_elite_strategy(game = "breakout", NUM_FRAMES = 1000, MAX_EVALS = 5000):

    env = Environment(game, sticky_action_prob=0.0, random_seed=0)
    env.reset()

    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()
    policy_net = Network(in_channels, num_actions).to(device)
    genes = policy_net.get_params()

    taille_genes = len(genes)
    genes = np.random.randn(taille_genes)

    policy_net.set_params(genes)

    archive = {} #archive (dictionnaire)

    for i in range(MAX_EVALS):

        policy_net.set_params(genes)

        fitness, behaviour = play(policy_net,game)



        specie = Species(genes, pos(behaviour) , fitness)

        addToArchive(archive, specie)

        genes = mutate(archive)



    listIndiv = archive.values()

    print([i.fitness for i in listIndiv])
    print(len(listIndiv))
    return archive


archive = map_elite_strategy()





