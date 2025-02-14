import logging
from pathlib import Path

import numpy as np
import torch as th
from lightning.pytorch import seed_everything
from torch.nn.utils.rnn import pad_sequence
from torchmetrics import MetricCollection

from components.data_loader import ToyLoader
from components.metrics import BinaryHighlightF1Score, SelectionSize, SelectionRate
from components.processing import HighlightDataset


def pad_highlights(
        highlights
):
    max_highlights = max([len(hl_group) for hl_group in highlights])
    for highlight_idx in range(len(highlights)):
        hl_group = highlights[highlight_idx]
        if len(hl_group) < max_highlights:
            for idx in range(len(hl_group), max_highlights):
                highlights[highlight_idx].append([-1] * len(highlights[0]))

    # flatten
    highlights = [seq for hl_group in highlights for seq in hl_group]

    highlights = pad_sequence([th.tensor(highlight, dtype=th.int32) for highlight in highlights],
                              batch_first=True,
                              padding_value=-1)
    highlights = highlights.reshape(len(highlights), -1)

    return highlights


def random_hl(
        hl_true,
        hl_size
):
    valid_indexes = np.where(hl_true.detach().cpu().numpy() != -1)[0]
    hl_hat = np.zeros(len(hl_true))[valid_indexes]
    sampled_indexes = np.random.choice(np.arange(len(valid_indexes)), size=hl_size, replace=False)
    hl_hat[sampled_indexes] = 1
    return hl_hat


def execute_random(
        highlights,
        hl_size
):
    hl_hat = [random_hl(hl_true=hl_true, hl_size=hl_size) for hl_true in highlights]
    return hl_hat


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Config
    # -------------

    data_dir = Path(__file__).parent.parent.resolve().joinpath('data')

    # -------------
    seed_everything(seed=15000)

    loader = ToyLoader(data_dir=data_dir)
    train_df, val_df, test_df = loader.get_splits()

    train_data = HighlightDataset(texts=train_df.text.values,
                                  masks=train_df['mask'].values,
                                  labels=train_df.label.values,
                                  highlights=train_df.highlight.values,
                                  sample_ids=train_df.sample_id.values)

    val_data = HighlightDataset(texts=val_df.text.values,
                                masks=val_df['mask'].values,
                                labels=val_df.label.values,
                                highlights=val_df.highlight.values,
                                sample_ids=val_df.sample_id.values)

    test_data = HighlightDataset(texts=test_df.text.values,
                                 masks=test_df['mask'].values,
                                 labels=test_df.label.values,
                                 highlights=test_df.highlight.values,
                                 sample_ids=test_df.sample_id.values)

    hl_sizes = [
        2,
        3,
        4,
    ]

    seeds = [
        2023,
        15451,
        1337,
        2001,
        2080,
    ]

    th_metrics = MetricCollection(
        {
            'hl_pos_f1': BinaryHighlightF1Score(pos_label=1),
            'hl_rate': SelectionRate(),
            'hl_size': SelectionSize()
        }
    )

    metrics = {}
    for hl_size in hl_sizes:
        for seed in seeds:
            seed_everything(seed=seed)

            hls_true = pad_highlights(highlights=test_data.highlights)
            hls_hat = execute_random(highlights=hls_true, hl_size=hl_size)
            hls_hat = pad_sequence([th.tensor(hl) for hl in hls_hat], batch_first=True, padding_value=-1)

            th_metrics.reset()
            th_metric_values = th_metrics(hls_hat, hls_true)
            for key, value in th_metric_values.items():
                metrics.setdefault(hl_size, {}).setdefault(key, []).append(float(value.detach().cpu().numpy()))

        metric_keys = list(metrics[hl_size].keys())
        for key in metric_keys:
            metrics[hl_size][f'avg_{key}'] = (np.mean(metrics[hl_size][key]), np.std(metrics[hl_size][key]))

    for hl_size in hl_sizes:
        print(f'Size: {hl_size}')
        for key, value in metrics[hl_size].items():
            if key.startswith('avg'):
                print(f'{key}: {metrics[hl_size][key][0]:.4f} +/- {metrics[hl_size][key][1]:.4f}')
        print()
