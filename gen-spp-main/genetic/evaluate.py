import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.highlight_extractor import HighlightExtractor
from src.genetic.individual import Individual
from src.preprocessing.dataset_reader.dataset_reader_interface import DatasetReaderInterface
from src.preprocessing.dataset_reader.toy_dataset_reader import ToyDatasetReader
from src.preprocessing.dataset_reader.hatexplain_dataset_reader import HatexplainDatasetReader

from src.utils.config import *
from src.utils.saving import *

FINE_TUNING_LR = 1e-3


def parse_arguments():

    parser = argparse.ArgumentParser(description='Gen-SPP')
    parser.add_argument('dataset_name', type=str, help='Dataset to be used: either "toy" or "hatexplain"')
    parser.add_argument('file_name', type=str, help='Name of the file to be evaluated (e.g. test_1')
    parser.add_argument('--e', type=int, help='Number of fine-tuning epochs', default=5)

    _args = parser.parse_args()

    return _args


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model running on device:", device)

    args = parse_arguments()
    dataset_name = args.dataset_name
    file_name = args.file_name
    epochs = args.e
    load_path = os.path.join("results", "weights", "model_" + file_name + ".pt")

    if dataset_name not in ["toy", "hatexplain"]:
        raise Exception("Invalid dataset name, please choose one between 'toy' and 'hatexplain'")

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
    INDIVIDUAL_PARAMETERS["training_epochs"] = epochs
    INDIVIDUAL_PARAMETERS["learning_rate"] = FINE_TUNING_LR
    INDIVIDUAL_PARAMETERS["max_trainings"] = 1

    model = HighlightExtractor(**MODEL_PARAMETERS)

    if not RUN_EAGERLY:
        model = torch.compile(model)

    model.load_state_dict(torch.load(load_path))

    INDIVIDUAL_PARAMETERS["model"] = model

    individual = Individual(**INDIVIDUAL_PARAMETERS)

    individual.refine()

    metrics = individual.compute_metrics(x_test, labels_test, highlight_masks_test)

    predicted_labels = individual.compute_labels(x_test)
    test_set_masks = individual.compute_masks(x_test)

    train_texts, test_texts = dataset.get_train_and_test_texts()

    save_to_txt(file_name, MASKS_SAVE_PATH, test_texts, list(test_set_masks), highlight_masks_test,
                predicted_classes=list(predicted_labels), real_classes=labels_test)

    save_metrics(file_name, METRICS_SAVE_PATH, metrics)

    for key, value in metrics.items():
        print("{}: {}".format(key, value))
