





class Species:
    def __init__(self,genotype,pos,fitness):
        self.genotype = genotype ## tableau 1D de taille le nb de dof
        self.pos = pos # la position, utilisee comme clef pour l'Archyves
        self.fitness = fitness # le fitness de l'espece


import random
import numpy as np


def addToArchive(archive, specie):
    pos = specie.pos ## la position dans l espace des comportements
    if (not pos in archive) or (archive[pos].fitness < specie.fitness):
        archive[pos] = specie #si on a personne a cet endroit, ou un meilleur fitness, on met la nouvelle espece





#return un genotype
def crossover(archive):
    rate = 0.5
    parent1 = random.choice(list(archive.values()))
    parent2 = random.choice(list(archive.values()))
    genotype_child = parent1.genotype
    for i in range(len(genotype_child)):
        if random.random()>rate:
            genotype_child[i] = parent2.genotype[i]
    return genotype_child

def crossover_one_point(archive):
    parent1 = random.choice(list(archive.values()))
    parent2 = random.choice(list(archive.values()))
    genotype_child = parent1.genotype
    n = random.randint(0,len(genotype_child)-1)
    genotype_child[n:] = parent2.genotype[n:]
    return genotype_child






def mutate(archive, cross = False, move_away = True):

    if cross:
        #Crossover
        new_genotype = crossover(archive)
    else :
        specie = random.choice(list(archive.values()))
        new_genotype = specie.genotype

    nb_genes = len(new_genotype)

    #Adding a random noise
    scale_factor = 0.1
    noise = np.random.randn(nb_genes)
    new_genotype = np.mod(new_genotype \
                     + 1 \
                     + noise * scale_factor, 2) \
                   - 1

    if move_away:
        #Going away from another specie
        scale_factor = 0.1
        noise = np.random.randn(nb_genes)
        other_specie = random.choice(list(archive.values()))
        other_genotype = other_specie.genotype
        new_genotype = np.mod(new_genotype \
                        + 1 \
                        + (new_genotype - other_genotype) * noise * scale_factor, 2) \
                    - 1

    return(new_genotype)