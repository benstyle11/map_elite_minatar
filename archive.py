


class Species:
    def __init__(self,genotype,pos,fitness):
        self.genotype = genotype ## tableau 1D de taille le nb de dof
        self.pos = pos # la position, utilisee comme clef pour l'Archyves
        self.fitness = fitness # le fitness de l'espece



def addToArchive(archive, specie):
    pos = specie.pos ## la position dans l espace des comportements
    if (not pos in archive) or (archive[pos].fitness < specie.fitness):
        archive[pos] = specie #si on a personne a cet endroit, ou un meilleur fitness, on met la nouvelle espece


def mutate(archive):
    pass