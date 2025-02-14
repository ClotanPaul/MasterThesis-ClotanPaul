import random
import numpy as np
from abc import ABC

from src.genetic.crossover.crossover_interface import CrossoverInterface
from src.genetic.individual import Individual


class IdentityCrossover(CrossoverInterface, ABC):

    @staticmethod
    def apply_crossover(parent_1: Individual, parent_2: Individual) -> tuple[np.ndarray, np.ndarray]:

        child_1 = parent_1.chromosome[0:]
        child_2 = parent_2.chromosome[0:]

        return child_1, child_2
