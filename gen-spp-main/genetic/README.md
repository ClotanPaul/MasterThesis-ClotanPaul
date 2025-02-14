# Gen-SPP

This repository is the official implementation of the method proposed in **Interlocking-free Selective Rationalization Through Genetic-based Learning**.

## How to train

In order to launch the experiments on the two datasets of *Toy* and *HateXplain*, follow these steps:

1. Download and extract GloVe's embeddings into the corresponding folder following the [instructions](data/glove/README.md) [only needed for HateXplain]
2. Edit the file `src/utils/config.py` [**Optional: keep the configuration as it is to replicate results from the paper**]
3. Run the script `train.py` passing the parameters:
    - `dataset_name`: it can either be *toy* or *hatexplain*; other strings will be considered invalid
    - `test_name`: name to associate to the current experiment, used for saving results [**Optional - Default: "test"**]
    - `--n`: integer indicating how many complete runs to perform for statistics [**Optional - Default: 5**]
    - `--w`:  number of workers (threads) to train individuals in parallel, effective only when training on CPU [**Optional - Default: 1**]
    - `--cpu`: flag to force training on CPU [**Optional - Default: False (check for GPU)**]

Example:
`python3 train.py toy toy_experiment --n=2`

This execution performs both training on the train set and evaluation on the test set, saving the best model and the corresponding results.

## Results

Results are saved inside the folder `results/outputs` and consist in files named `masks_[test_name]` and `metrics[test_name]`, respectively containing the output highlights generated on the test set and the metrics relative to the model.

## How to evaluate

It is possible to fine-tune the trained models (their predictors) and then execute a new evaluation as follows:

1. Check to have the same configuration in `src/utils/config.py` as the one used for training
2. Run the script `evaluate.py` passing the parameters:
    - `dataset_name`: it can either be *toy* or *hatexplain*; other strings will be considered invalid
    - `file_name`: name of the file associated to the trained model
    - `--e`: number of fine-tuning epochs [**Optional - Default: 5**]

Example:
`python3 evaluate.py toy toy_experiment_1 --e=10`

Results will override the existing ones.