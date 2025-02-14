from abc import ABC
from multiprocessing.pool import ThreadPool

from src.genetic.survival.survival_interface import SurvivalInterface
from src.genetic.individual import Individual


class ElitismSurvival(SurvivalInterface, ABC):

    @staticmethod
    def survival_select(population: list[Individual], population_size: int, workers=1) -> list[Individual]:

        if workers == 1:
            for individual in population:
                _ = individual.fitness

        else:
            with ThreadPool(processes=workers) as pool:
                pool.map(lambda x: x.fitness, population)

        population.sort(reverse=True, key=lambda x: x.fitness)

        return population
