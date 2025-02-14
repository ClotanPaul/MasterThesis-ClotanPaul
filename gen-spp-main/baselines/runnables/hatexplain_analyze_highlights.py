import logging
import os
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage
from lightning.pytorch import seed_everything

from components.data_loader import HatexplainLoader
from components.processing import HighlightDataset


def show_stats(
        groups,
        group_sizes,
        sparsity,
        split
):
    to_show = {
        'avg_groups': np.mean(groups),
        'top_k_groups': Counter(groups).most_common(5),
        'avg_size': np.mean(group_sizes),
        'top_k_size': Counter(group_sizes).most_common(5),
        'avg_sparsity': np.mean(sparsity),
        'top_k_sparsity': Counter(sparsity).most_common(5)
    }

    logging.info(f'{split}: {os.linesep}{to_show}')


def count_highlight_groups(
        highlight
):
    highlight = np.array(highlight) if type(highlight) != np.ndarray else highlight

    group_mask, groups = ndimage.label(highlight)
    group_sizes = []
    for g_idx in range(groups):
        group_sizes.append(np.where(group_mask == g_idx + 1)[0].shape[0])

    sparsity = [size / len(highlight) for size in group_sizes]

    return groups, np.mean(group_sizes), np.mean(sparsity)


def get_highlight_info(
        highlights
):
    groups, group_sizes, sparsity = [], [], []
    for highlight in highlights:
        # Skip empty highlights
        if np.sum(highlight) == 0:
            continue

        h_groups, h_sizes, h_sparsity = count_highlight_groups(highlight=highlight)
        groups.append(h_groups)
        group_sizes.append(h_sizes)
        sparsity.append(h_sparsity)

    return groups, group_sizes, sparsity


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    data_dir = Path(__file__).parent.parent.resolve().joinpath('data')
    seed_everything(seed=15000)

    loader = HatexplainLoader(data_dir=data_dir)
    train_df, val_df, test_df = loader.get_splits()

    train_data = HighlightDataset(texts=train_df.text.values,
                                  masks=train_df['mask'].values,
                                  labels=train_df.label.values,
                                  highlights=train_df.highlight.values,
                                  sample_ids=train_df.sample_id.values)
    groups, group_sizes, sp = get_highlight_info(train_data.highlights)
    show_stats(groups=groups, group_sizes=group_sizes, sparsity=sp, split='train')

    val_data = HighlightDataset(texts=val_df.text.values,
                                masks=val_df['mask'].values,
                                labels=val_df.label.values,
                                highlights=val_df.highlight.values,
                                sample_ids=val_df.sample_id.values)
    groups, group_sizes, sp = get_highlight_info(val_data.highlights)
    show_stats(groups=groups, group_sizes=group_sizes, sparsity=sp, split='val')

    test_data = HighlightDataset(texts=test_df.text.values,
                                 masks=test_df['mask'].values,
                                 labels=test_df.label.values,
                                 highlights=test_df.highlight.values,
                                 sample_ids=test_df.sample_id.values)
    groups, group_sizes, sp = get_highlight_info(test_data.highlights)
    show_stats(groups=groups, group_sizes=group_sizes, sparsity=sp, split='test')
