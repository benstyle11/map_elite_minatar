


class Species:
    def init(self,genotype,pos,fitness):
        self.genoptype = genotype
        self.pos = pos
        self.fitness = fitness



def addToArchive(archive, specie):
    pos = specie.pos
    if (not pos in archive) or (archive[pos].fitness < specie.fitness):
        archive[pos] = specie


def mutate(archive):
    pass