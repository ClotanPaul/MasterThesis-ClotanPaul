# Baselines

This repository is the official implementation of the selective rationalization baselines evaluated in **Interlocking-free Selective Rationalization Through Genetic-based Learning**.

## Structure

The code is organized as follows:

- ``components``: contains all business logic, including data loaders, lightning models, text processors, metrics, and callbacks.
- ``configurations``: the ``model.py`` script contains all baseline configurations for Toy and HateXplain datasets.
- ``modeling``: contains pytorch implementation for models, constraints, and model-specific layers.
- ``runnables``: contains all executable scripts for reproducing our results.

## Preliminaries

Before running a script, make sure you address the following preliminaries:

* Install requirements: ``pip install -r requirements.txt``
* [HateXplain] Download GloVe embeddings and put them into ``embeddings`` folder.
* Check model configurations in ``configurations/model.py``.

## Reproducing our experiments

All baseline models are implemented in Pytorch Lightning and trained to ensure reproducibility.
The ``runnables`` folder contains a script for testing each model on each dataset.
In particular, the scripts can be organized as follows:

### Toy scripts

The following scripts reproduce our results on Toy dataset.
In each script, the corresponding model configuration defined in ``configurations/model.py`` is loaded (see Configuration section for more details).

**Benchmark evaluation**
* ``toy_fr_train.py``
* ``toy_mgr_train.py``
* ``toy_mcd_train.py``
* ``toy_grat_train.py``

**Synthetic Skew**
* ``toy_fr_train_skew.py``
* ``toy_mgr_train_skew.py``
* ``toy_mcd_train_skew.py``
* ``toy_grat_train_skew.py``

### HateXplain scripts

The following scripts reproduce our results on Toy dataset.
In each script, the corresponding model configuration defined in ``configurations/model.py`` is loaded (see Configuration section for more details).

**Benchmark evaluation**
* ``hatexplain_fr_train.py``
* ``hatexplain_mgr_train.py``
* ``hatexplain_mcd_train.py``
* ``hatexplain_grat_train.py``

**Synthetic Skew**
* ``hatexplain_fr_train_skew.py``
* ``hatexplain_mgr_train_skew.py``
* ``hatexplain_mcd_train_skew.py``
* ``hatexplain_grat_train_skew.py``

### How to run

**Benchmark evaluation**

It is sufficient to run each script without additional arguments.

Example (training FR on HateXplain):
```commandline
python3 runnables/hatexplain_fr_train.py
```

**Synthetic Skew**

It is sufficient to run each script by providing the skew K epochs parameter.

Example (training FR skew on HateXplain with K = 10)
```commandline
python3 runnables/hatexplain_fr_train_skew.py -s 10
```

## Results

Results are stored in the `results` folder.
In particular, each model run is structured as follows:

```
results
   |- toy
      |- fr
         |- checkpoints
         |- metrics.npy
         |- *predictions* 
      |- ... 
   - hatexplain
      |- ...
```

Where ``checkpoints`` contains each seed run model checkpoints for reproducibility, 
``metrics.npy`` stores all validation and test partitions metrics, and `*predictions*` is a placeholder denoting each seed run model predictions.

Example (FR on Toy)
```
results
   |- toy
      |- fr
         |- checkpoints
         |- metrics.npy
         |- predictions_seed=1337.csv
         |- predictions_seed=15451.csv
         |- predictions_seed=2001.csv
         |- predictions_seed=2023.csv
         |- predictions_seed=2080.csv
```

## Configurations

We organize model configurations in Python classes.
In this way, configurations can be easily extended and maintained.

A specific model configuration is implemented as a Python function, referenced via a compound key (`ConfigKey` in `configurations/base.py`).

Example (FR in Toy)
```python
    @classmethod
    def toy(
            cls
    ):
        return cls(
            seeds=[
                2023,
                15451,
                1337,
                2001,
                2080,
            ],
            optimizer_class=th.optim.Adam,
            optimizer_kwargs={'lr': 1e-03},
            batch_size=64,
            embedding_dim=25,
            hidden_size=8,
            classification_head=lambda: th.nn.Sequential(
                th.nn.Linear(16, 3)
            ),
            selection_head=lambda: th.nn.Sequential(
                th.nn.Linear(16, 2)
            ),
            num_classes=3,
            dropout_rate=0.0,
            freeze_embeddings=True,
            add_sparsity_loss=True,
            sparsity_coefficient=1.0,
            sparsity_level=0.15,
            add_continuity_loss=True,
            continuity_coefficient=1.0,
            classification_coefficient=1.0
        )
```

### Adding a configuration

Adding a new configuration is straightforward:

* Define a new classmethod function to define your model configuration
* Define a `ConfigKey` to reference your configuration

Example (new FR config)
```python
# In configurations/model.py
class FRConfig(BaseConfig):
    configs = {
        # Toy
        ConfigKey(dataset='toy', tags={'spp', 'fr', 'custom'}): 'toy_custom',
    }

    @classmethod
    def toy(
        cls
    ):
        # customize
        return cls(...)
    
# In a runnables script
config = FRConfig.from_config(key=ConfigKey(dataset='toy', tags={'spp', 'fr', 'custom'}))

```

## Random Baselines

We provide scripts for reproducing random baselines on Toy and HateXplain datasets.

```commandline
python3 runnables/hatexplain_random.py
python3 runnables/toy_random.py
```

## Data Inspection

We provide scripts to analyze HateXplain highlights (Figure 1 in Supplementary Materials) and their sparsity rate.

```commandline
python3 runnables/hatexplain_analyze_highlights.py      # highlight fragmentation
python3 runnables/inspect_hatexplain.py                 # sparsity rate
```

## Toy pattern Baseline

We provide a script to evaluate string matching baselines (see Supplementary Material).

```commandline
python3 runnables/toy_pattern.py
```

## Showing results

We provide a utility script to quickly retrieve model results

```commandline
python3 runnables/show_results.py
```
