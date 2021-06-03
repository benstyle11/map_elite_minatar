import random
from numpy import array, mod, ones



class Species:
    def init(self,genotype,pos,fitness):
        self.genotype = genotype
        self.pos = pos
        self.fitness = fitness



def addToArchive(archive, specie):
    pos = specie.pos
    if (not pos in archive) or (archive[pos].fitness < specie.fitness):
        archive[pos] = specie


def mutate(archive, cross = False, move_away = True):

    if cross:
        #Crossover
        new_genotype = crossover(archive)
    else :
        specie = random.choice(list(archive.values))
        new_genotype = specie.genotype

    nb_genes = len(new_genotype)
    array_of_ones = ones(nb_genes)

    #Adding a random noise
    scale_factor = 1
    noise = random.randn(nb_genes)
    new_genotype = mod(new_genotype \
                     + array_of_ones \
                     + noise * scale_factor, 2) \
                   - array_of_ones

    if move_away:
        #Going away from another specie
        scale_factor = 1
        noise = random.randn(nb_genes))
        other_specie = random.choice(list(archive.values))
        other_genotype = other_specie.genotype
        new_genotype = mod(new_genotype \
                        + array_of_ones \
                        + (new_genotype - other_genotype) * noise * scale_factor, 2) \
                    - array_of_ones

    return(new_genotype)

