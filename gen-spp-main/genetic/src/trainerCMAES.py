import cma
import numpy as np
from src.trainer import GeneticTrainer
import gc
import time
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

from typing import Type
import numpy as np
import matplotlib.pyplot as plt

import torch
from src.genetic.selection.selection_interface import SelectionInterface
from src.genetic.crossover.crossover_interface import CrossoverInterface
from src.genetic.mutation.mutation_interface import MutationInterface

from src.genetic.survival.survival_interface import SurvivalInterface

from src.genetic.individual import Individual
from src.model.highlight_extractor import HighlightExtractor

class GeneticTrainerCMAES(GeneticTrainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # CMA-ES Parameters
        self.cma_sigma = 0.5  # Initial mutation step size
        # TO DO: MEMORY ALLOCATION FAIL.
        self.cma_population_size = self.population_size  # Set CMA-ES population size

        # Determine chromosome size
        ### TO DO NEED TO DETERMINE HOW TO GET THIS DIMENSION CORRECTLY
        #self.chromosome_size = 873#2081

        self.chromosome_size = self._infer_chromosome_size()

        # Initialize CMA-ES optimizer
        self.es = cma.CMAEvolutionStrategy(self.chromosome_size * [0], self.cma_sigma, {'popsize': self.cma_population_size})


    def _infer_chromosome_size(self) -> int:
        """
        Builds a temporary individual to infer the length of the chromosome.
        """
        temp_model = HighlightExtractor(**self.model_params)
        temp_individual = Individual(model=temp_model, **self.individual_params)
        size = len(temp_individual.chromosome)
        del temp_model
        del temp_individual
        gc.collect()
        return size

    ## TO MODIFY
    def cmaes_fitness(self, individual):
        """
        Computes the fitness function for CMA-ES.
        Args:
            chromosome (np.ndarray): Candidate solution (chromosome).
        Returns:
            float: Fitness value (lower is better).
        """

        loss = 1.0 / individual.fitness 

        return loss  # CMA-ES minimizes


    def __run_generation(self) -> None:
        """
        Runs one generation of evolution using CMA-ES instead of traditional genetic operations.
        """

        # Generate new candidate chromosomes from CMA-ES
        print("Running CMA-ES optimization for mutation")
        chromosomes = self.es.ask()  


        #Create indivituals with the new chromosomes.
        print(f"Updating the individuals with the new chromosomes.")
        for chromosome in chromosomes:
            current_individual = self.__create_individual(chromosome=chromosome)
            self.population.append(current_individual)

        #fitness_values = [self.cmaes_fitness(individual) for individual in self.population] #testing purposes

        if self.refine:
            self.__apply_sgd_refinement()
        print(f"Computing fitness values...")
        fitness_values = [self.cmaes_fitness(individual) for individual in self.population]



        self.population = self.survival_strategy.survival_select(self.population,
                                                                self.population_size,
                                                                workers=self.workers)
        self.__remove_extra_individuals()

        best_chromosomes = [ind.chromosome for ind in self.population]  

        fitness_values = [self.cmaes_fitness(ind) for ind in self.population]
        
        #sorted_fitness_values = sorted(fitness_values)

        #print("Sorted fitness values:", sorted_fitness_values)

        # Update the CMA-ES optimizer with the top 50 chromosomes
        assert len(best_chromosomes) == len(fitness_values), f"Mismatch: {len(best_chromosomes)} solutions but {len(fitness_values)} fitness values"
        self.es.tell(best_chromosomes, fitness_values)



    def train(self, plot_results: bool = True) -> None:

        self.initialize()

        n_params = sum(p.numel() for p in self.population[0].model.parameters() if p.requires_grad)
        print("Trainable variables: {}".format(n_params))

        if self.refine:
            self.__apply_sgd_refinement()

        print("Training started for {} generations".format(self.n_generations))

        for _ in range(self.n_generations):
            try:
                start_time = time.time()
                self.__run_generation()
                self.current_generation += 1
                best_individual = self.get_best_individual()
                best_fitness = best_individual.fitness
                best_loss = 1.0/best_fitness

                end_time = time.time()
                print("Generation {}/{} completed in {}s -> ".format(self.current_generation,
                                                                     self.n_generations,
                                                                     int(end_time - start_time)))
                print("\n\nLowest loss: {}\n\n".format(best_loss))
                print("Cross-Entropy: {} - Sparsity: {} - Confidence: {} - Contiguity: {}".format(best_individual.ce,
                                                                                                  best_individual.sparsity,
                                                                                                  best_individual.confidence,
                                                                                                  best_individual.contiguity))

                self.training_progress.append(best_loss)
                if best_loss <= self.stop_threshold:
                    break
            except KeyboardInterrupt:
                break

        if plot_results:
            self.plot_training_progress()





#####################################################################################
## TO DO: use the methods from the inherited class, and delete them from here.

    def __build_models_pool(self) -> tuple[dict[HighlightExtractor, Individual | None],
                                           dict[HighlightExtractor, dict]]:

        pool: dict[HighlightExtractor, Individual | None] = dict()
        initial_weights: dict[HighlightExtractor, dict] = dict()

        print("Building models...")

        for _ in tqdm(range(self.max_population_size)):

            model = HighlightExtractor(**self.model_params)

            if not self.run_eagerly:
                model = torch.compile(model)

            gc.collect()

            pool[model] = None
            if self.train_generator_only:
                initial_weights[model] = model.classifier.state_dict()
            else:
                initial_weights[model] = model.state_dict()
        return pool, initial_weights

    def __allocate_new_model(self) -> HighlightExtractor:

        print("WARNING: Allocating new model: repeating this operation many times could cause a memory leak")

        model = HighlightExtractor(**self.model_params)

        if not self.run_eagerly:
            model = torch.compile(model)

        gc.collect()

        self.models_pool[model] = None
        if self.train_generator_only:
            self.initial_weights[model] = model.classifier.state_dict()
        else:
            self.initial_weights[model] = model.state_dict()

        return model

    def __create_individual(self, chromosome: np.ndarray | None = None) -> Individual:

        model = self.__get_free_model()

        if model is None:
            model = self.__allocate_new_model()
        else:
            weights = self.initial_weights[model]
            if self.train_generator_only:
                model.classifier.load_state_dict(weights)
            else:
                model.load_state_dict(weights)

        self.individual_params["model"] = model
        individual = Individual(**self.individual_params)
        self.models_pool[model] = individual

        if chromosome is not None:
            individual.update_chromosome(chromosome)

        return individual

    def __get_free_model(self) -> HighlightExtractor | None:

        for model in self.models_pool:
            if self.models_pool[model] is None:
                return model

        return None

    def __get_model_from_individual(self, individual: Individual) -> HighlightExtractor | None:

        for model in self.models_pool:
            if self.models_pool[model] == individual:
                return model

        return None

    def __apply_sgd_refinement(self) -> None:

        print("Refining models...")

        if self.workers == 1:
            for individual in tqdm(self.population):
                individual.refine()
                gc.collect()

        else:
            with ThreadPool(processes=self.workers) as pool:
                pool.map(lambda x: x.refine(), self.population)
            gc.collect()

    def __remove_extra_individuals(self) -> None:

        n_extra = self.max_population_size - self.population_size
        print(f"The extra population is:")

        for _ in range(n_extra):
            individual = self.population[-1]
            model = self.__get_model_from_individual(individual)
            if model is None:
                raise Exception("No model for the individual")
            self.models_pool[model] = None
            del self.population[-1]
        print(f"The actual length of the popution is: {len(self.population)}")
        print(f"The population size param is: {self.population_size}")
        assert len(self.population) == self.population_size

        gc.collect()
