


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


'''
def pos(behaviour, game):
    n_points = 1

    if game == "breakout" :
        staticite = (behaviour[0] + behaviour[2] + behaviour[3] + behaviour[5]) \
            / np.sum(behaviour) #normalize the behaviour by the nb of inputs
        position = np.floor(n_points*staticite)

    else :
        taux_up = behaviour[2] / np.sum(behaviour)
        taux_down = behaviour[4] / np.sum(behaviour)
        position = (np.floor(n_points*taux_up), np.floor(n_points*taux_down))

    return position
'''

def pos(behaviour,game):

    return np.floor(behaviour)



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



        specie = Species(genes, pos(behaviour,game) , fitness)

        addToArchive(archive, specie)

        genes = mutate(archive)



    listIndiv = archive.values()

    play(policy_net,game,True)

    print([i.fitness for i in listIndiv])
    print(len(listIndiv))
    return archive


archive = map_elite_strategy()





