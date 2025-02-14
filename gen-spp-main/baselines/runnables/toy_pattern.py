import logging
from pathlib import Path

import numpy as np
import torch as th
from lightning.pytorch import seed_everything
from torchmetrics.classification.f_beta import F1Score

from components.data_loader import ToyLoader
from components.processing import HighlightDataset

patterns = {
    'v1': ["aba", "baa", 'abc'],
    'v2': ['abc', 'baa', 'aba'],
    'v3': ['ba', 'aa', 'bc'],
}


def get_patterns(
        variant
):
    return patterns[variant]


def get_prediction(
        text,
        p0,
        p1,
        p2
):
    if p0 in text:
        return 0
    elif p1 in text:
        return 1
    elif p2 in text:
        return 2
    else:
        raise RuntimeError(f'Should never happen: {text}')


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

    p0, p1, p2 = get_patterns(variant='v3')

    seeds = [
        2023,
        15451,
        1337,
        2001,
        2080,
    ]

    f1 = F1Score(task='multiclass', num_classes=3, average='macro')

    metrics = {}
    for seed in seeds:
        seed_everything(seed=seed)

        texts = test_data.texts
        y_true = th.tensor(test_data.labels)
        y_pred = th.tensor([get_prediction(text, p0=p0, p1=p1, p2=p2) for text in texts])

        f1.reset()
        th_f1 = float(f1(y_pred, y_true).detach().cpu().numpy())

        metrics.setdefault('f1', []).append(th_f1)

    metrics['avg_f1'] = (np.mean(metrics['f1']), np.std(metrics['f1']))

    print(f'Clf f1: {metrics["avg_f1"][0]:.4f} +/- {metrics["avg_f1"][1]:.4f}')
