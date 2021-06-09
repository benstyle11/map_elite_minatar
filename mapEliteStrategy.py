


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

    behaviour_act,behaviour_dist = behaviour

    n_points = 5

    if game == "breakout" :
        staticite = (behaviour_act[0] + behaviour_act[2] + behaviour_act[3] + behaviour_act[5]) \
            / np.sum(behaviour_act) #normalize the behaviour by the nb of inputs
        position = np.floor(n_points*staticite)


        return (position,np.floor(behaviour_dist))

    else :
        taux_up = behaviour_act[2] / np.sum(behaviour_act)
        taux_down = behaviour_act[4] / np.sum(behaviour_act)
        position = (np.floor(n_points*taux_up - n_points*taux_down))


    return position









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

    num_random_archive = min(200, MAX_EVALS//2)

    for i in range(num_random_archive):
        genes = np.random.randn(taille_genes)
        policy_net.set_params(genes)
        fitness, behaviour = play(policy_net,game)
        specie = Species(genes, pos(behaviour,game) , fitness)
        addToArchive(archive,specie)


    listMax = []



    for i in range(MAX_EVALS):

        policy_net.set_params(genes)

        fitness, behaviour = play(policy_net,game)



        specie = Species(genes, pos(behaviour,game) , fitness)

        addToArchive(archive, specie)
        listIndiv = archive.values()

        listMax.append(max([i.fitness for i in listIndiv]))

        genes = mutate(archive,True,True)


    plt.plot(listMax)
    plt.savefig(game + "_" + str(NUM_FRAMES) + "_" + str(MAX_EVALS) + "_" + str(time.time()) + ".png")

    plt.show()


    listIndiv = archive.values()

    print([i.fitness for i in listIndiv])
    print(len(listIndiv))
    return archive


def testAgent(specie, game):
    env = Environment(game, sticky_action_prob=0.0, random_seed=0)
    env.reset()

    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()
    policy_net = Network(in_channels, num_actions).to(device)
    policy_net.set_params(specie.genotype)
    play(policy_net,game,True)


def displayArchive(archive):
    fit = np.zeros((10,10))
    m_min = 20
    m_max = 0
    l = archive.values()
    for i in l :
        x,y = (int(i.pos[0]),int(i.pos[1]))
        fit[x,y] = i.fitness
        if i.fitness < m_min:
            m_min = i.fitness
        if i.fitness > m_max:
            m_max = i.fitness


    plt.imshow(np.array(fit))
    plt.clim(m_min, m_max)


archive = map_elite_strategy("breakout",1000,2000)

displayArchive(archive)


