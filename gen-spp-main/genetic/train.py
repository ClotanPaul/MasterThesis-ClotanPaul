import numpy as np
import random
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.trainer import GeneticTrainer
from src.preprocessing.dataset_reader.dataset_reader_interface import DatasetReaderInterface
from src.preprocessing.dataset_reader.toy_dataset_reader import ToyDatasetReader
from src.preprocessing.dataset_reader.hatexplain_dataset_reader import HatexplainDatasetReader

from src.utils.saving import save_to_txt, save_metrics
from src.utils.config import *
from src.utils.metrics import StatisticalMetrics


def set_random_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_arguments():

    warning = "WARNING: w > 1 can only affect training on CPU"

    parser = argparse.ArgumentParser(description='Gen-SPP')
    parser.add_argument('dataset_name', type=str, help='Dataset to be used: either "toy" or "hatexplain"')
    parser.add_argument('test_name', type=str, help='Name associated to the current test', default="test")
    parser.add_argument('--n', type=int, help='Number of complete training runs for statistics', default=5)
    parser.add_argument("--w", type=int, help="Number of workers to train in parallel." + warning, default=1)
    parser.add_argument("--cpu", action='store_true', help="Whether to run on CPU or GPU (if available)")

    _args = parser.parse_args()

    return _args


def run_training(dataset_name: str, test_name: str, workers: int, run_count: int, device, plot_train_loss=True) -> dict:

    if dataset_name not in ["toy", "hatexplain"]:
        raise Exception("Invalid dataset name, please choose one between 'toy' and 'hatexplain'")

    test_name += "_{}"

    if dataset_name == "toy":
        dataset_reader: DatasetReaderInterface = ToyDatasetReader(TOY_DATASET_MAX_LEN, TOY_DATASET_TRAIN_PERCENTAGE)
        dataset = dataset_reader.read(TOY_DATASET_PATHS)
        GEN_ENCODER_PARAMS["initial_embedding_size"] = TOY_TOKEN_EMBEDDING_DIM
        GEN_ENCODER_PARAMS["embedding_size"] = TOY_HIDDEN_EMBEDDING_DIM
        CL_ENCODER_PARAMS["initial_embedding_size"] = TOY_TOKEN_EMBEDDING_DIM
        CL_ENCODER_PARAMS["embedding_size"] = TOY_HIDDEN_EMBEDDING_DIM
        INDIVIDUAL_PARAMETERS["ce_expected_loss"] = TOY_CE_EXPECTED_LOSS

    else:
        dataset_reader: DatasetReaderInterface = HatexplainDatasetReader(GLOVE_TWITTER_PATH,
                                                                         HATE_EXPLAIN_TOKEN_EMBEDDING_DIM,
                                                                         HATE_EXPLAIN_FILTER_MAX_LEN)
        dataset = dataset_reader.read(HATE_DATASET_PATHS)
        GEN_ENCODER_PARAMS["initial_embedding_size"] = HATE_EXPLAIN_TOKEN_EMBEDDING_DIM
        GEN_ENCODER_PARAMS["embedding_size"] = HATE_EXPLAIN_HIDDEN_EMBEDDING_DIM
        CL_ENCODER_PARAMS["initial_embedding_size"] = HATE_EXPLAIN_TOKEN_EMBEDDING_DIM
        CL_ENCODER_PARAMS["embedding_size"] = HATE_EXPLAIN_HIDDEN_EMBEDDING_DIM
        INDIVIDUAL_PARAMETERS["ce_expected_loss"] = HATE_EXPLAIN_CE_EXPECTED_LOSS

    train_set, test_set = dataset.get_train_and_test_sets()

    x_train, labels_train, highlight_masks_train = train_set
    x_test, labels_test, highlight_masks_test = test_set

    val_samples = int(len(labels_train) * VAL_SET_PERCENTAGE)
    permutation = np.random.permutation(len(labels_train))
    x_val_text = np.take(x_train[0], permutation, axis=0)
    x_val_text = x_val_text[0:val_samples]
    x_val_mask = np.take(x_train[1], permutation, axis=0)
    x_val_mask = x_val_mask[0:val_samples]
    x_val = (x_val_text, x_val_mask)
    labels_val = np.take(labels_train, permutation, axis=0)
    labels_val = labels_val[0:val_samples]

    dataset_train = TensorDataset(torch.Tensor(x_train[0]), torch.Tensor(x_train[1]), torch.Tensor(labels_train))
    train_loader = DataLoader(dataset_train, batch_size=INDIVIDUAL_PARAMETERS["batch_size"], shuffle=True)

    dataset_val = TensorDataset(torch.Tensor(x_val[0]), torch.Tensor(x_val[1]), torch.Tensor(labels_val))
    val_loader = DataLoader(dataset_val, batch_size=INDIVIDUAL_PARAMETERS["batch_size"], shuffle=False)

    MODEL_PARAMETERS["n_classes"] = dataset.get_classes_count()
    MODEL_PARAMETERS["gen_encoder_params"]["max_len"] = dataset.max_len
    MODEL_PARAMETERS["cl_encoder_params"]["max_len"] = dataset.max_len
    INDIVIDUAL_PARAMETERS["train_loader"] = train_loader
    INDIVIDUAL_PARAMETERS["val_loader"] = val_loader
    INDIVIDUAL_PARAMETERS["original_val_masks"] = x_val[1]
    INDIVIDUAL_PARAMETERS["device"] = device

    trainer = GeneticTrainer(N_GENERATIONS, POPULATION_SIZE, SELECTION_RATE, MUTATION_PROB, SELECTION_STRATEGY,
                             CROSSOVER_STRATEGY, MUTATION_STRATEGY, SURVIVAL_STRATEGY, MODEL_PARAMETERS,
                             INDIVIDUAL_PARAMETERS, dataset.embedding_dim, dataset.max_len,
                             RUN_EAGERLY, INDIVIDUAL_PARAMETERS["train_generator_only"],
                             STOP_THRESHOLD, REFINE_WITH_SGD, workers)

    trainer.train(plot_results=plot_train_loss)

    trainer.save_best_individual(test_name.format(run_count), MODEL_SAVE_PATH)

    train_set_metrics = trainer.get_metrics(x_train, labels_train, highlight_masks_train)
    test_set_metrics = trainer.get_metrics(x_test, labels_test, highlight_masks_test)

    test_set_masks = trainer.get_masks(x_test)
    train_texts, test_texts = dataset.get_train_and_test_texts()

    print()
    print("Metrics - Training set:")
    print(train_set_metrics)
    print()

    print("Metrics - Test set:")
    print(test_set_metrics)
    print()

    predicted_labels = trainer.get_labels(x_test)

    save_to_txt(test_name.format(run_count), MASKS_SAVE_PATH, test_texts, list(test_set_masks),
                highlight_masks_test, predicted_classes=list(predicted_labels), real_classes=labels_test)

    save_metrics(test_name.format(run_count), METRICS_SAVE_PATH, test_set_metrics)

    return test_set_metrics


if __name__ == '__main__':

    args = parse_arguments()
    n_runs = args.n
    _dataset_name = args.dataset_name
    _test_name = args.test_name
    _workers = args.w
    use_cpu = args.cpu

    if use_cpu:
        dev = torch.device("cpu")
        if _workers > 1:
            RUN_EAGERLY = True
    else:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            _workers = 1
        else:
            dev = torch.device("cpu")
            if _workers > 1:
                RUN_EAGERLY = True

    print("Model running on device:", dev)

    statistical_metrics = StatisticalMetrics()

    assert n_runs >= 1
    assert _workers >= 1

    for i in range(n_runs):
        print("RUNNING TRAINING {}/{}".format(i+1, n_runs))
        set_random_seed(i+1)
        metrics = run_training(_dataset_name, _test_name, _workers, i+1, dev, plot_train_loss=False)
        statistical_metrics.add_metrics(metrics)

    statistics = statistical_metrics.get_textual_statistics()

    for line in statistics:
        print(line)
