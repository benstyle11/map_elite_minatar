


class Species:
    def init(self,genotype,pos,fitness):
        self.genoptype = genotype
        self.pos = pos
        self.fitness = fitness



def addToArchive(archive, species):
    pos = species.pos
    if (not pos in archive) or (archive[pos].fitness < species.fitness):
        archive[pos] = species

