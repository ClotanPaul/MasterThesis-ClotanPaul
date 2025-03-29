import os

from src.genetic.selection.roulette_wheel import RouletteWheelSelection
from src.genetic.crossover.one_point import OnePointCrossover
from src.genetic.mutation.gaussian import GaussianMutation
from src.genetic.mutation.split_mutator import SplitMutation
from src.genetic.survival.half_elitism import HalfElitismSurvival

from src.model.layers.encoder import EncoderType, EncoderBuilder

# Paths
TOY_DATASET_PATHS = [os.path.join("data", "toy_dataset.pkl")]

HATE_DATASET_PATHS = [os.path.join("data", "hatexplain", "train.pkl"),
                      os.path.join("data", "hatexplain", "test.pkl"),
                      os.path.join("data", "hatexplain", "val.pkl")]

MODEL_SAVE_PATH = os.path.join("results", "weights", "model_{}")
GLOVE_TWITTER_PATH = os.path.join("data", "glove", "twitter", "glove.twitter.27B.{}d.txt")
MASKS_SAVE_PATH = os.path.join("results", "outputs", "masks_{}.txt")
METRICS_SAVE_PATH = os.path.join("results", "outputs", "metrics_{}.txt")

# Toy dataset
TOY_DATASET_MAX_LEN = 20
TOY_DATASET_TRAIN_PERCENTAGE = 0.8
TOY_TOKEN_EMBEDDING_DIM = 26
TOY_HIDDEN_EMBEDDING_DIM = 8
TOY_CE_EXPECTED_LOSS = 0.1

# Hatexplain dataset
HATE_EXPLAIN_FILTER_MAX_LEN = 30
HATE_EXPLAIN_TOKEN_EMBEDDING_DIM = 25
HATE_EXPLAIN_HIDDEN_EMBEDDING_DIM = 16
HATE_EXPLAIN_CE_EXPECTED_LOSS = 0.6

N_GENERATIONS = 100
POPULATION_SIZE = 50
SELECTION_RATE = 0.5
MUTATION_PROB = 1.0
SELECTION_STRATEGY = RouletteWheelSelection
CROSSOVER_STRATEGY = OnePointCrossover
MUTATION_STRATEGY = SplitMutation(sections=[(0, -1), (-1, -1)],
                                  mutations=[GaussianMutation(mean=0.0, std=0.05, noise=True),
                                             GaussianMutation(mean=0.0, std=0.10, noise=True)],
                                  probabilities=[MUTATION_PROB, MUTATION_PROB])
SURVIVAL_STRATEGY = HalfElitismSurvival
STOP_THRESHOLD = 0.01
REFINE_WITH_SGD = True

# Model's parameters
GEN_ENCODER_BUILDER = EncoderBuilder(EncoderType.RECURRENT)
CL_ENCODER_BUILDER = EncoderBuilder(EncoderType.RECURRENT)

GEN_ENCODER_PARAMS = {
    "apply_positional_encoding": False,
    "dropout": 0.0,
    "bidirectional": False
}

CL_ENCODER_PARAMS = {
    "apply_positional_encoding": False,
    "dropout": 0.0,
    "bidirectional": False
}

MODEL_PARAMETERS = {
    "generator_hidden_units": [],
    "classifier_hidden_units": [],
    "gen_encoder_builder": GEN_ENCODER_BUILDER,
    "cl_encoder_builder": CL_ENCODER_BUILDER,
    "gen_encoder_params": GEN_ENCODER_PARAMS,
    "cl_encoder_params": CL_ENCODER_PARAMS,
    "shared_encoder": False
}

VAL_SET_PERCENTAGE = 0.2
RUN_EAGERLY = False

# Individual's parameters
INDIVIDUAL_PARAMETERS = {
    "batch_size": 64,
    "training_epochs": 3,
    "max_trainings": 1,
    "train_generator_only": True,
    "learning_rate": 1e-2,
    "metric": "all",
    "reduce_multi_masks": True,
    "use_confidence_in_fitness": False
}
