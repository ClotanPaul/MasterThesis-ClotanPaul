from pathlib import Path

import numpy as np
from components.data_loader import HatexplainLoader


def get_sparsity_level(df):
    ratios = []
    counts = []
    for highlight in df.highlight.values:
        count = np.array(highlight).sum()
        ratio = count / len(highlight)
        ratios.append(ratio)
        counts.append(count)

    print(f'Count: {np.mean(counts):.4f} +/- {np.std(counts):.4f}')
    print(f'Ratio: {np.mean(ratios):.4f} +/- {np.std(ratios):.4f}')


if __name__ == '__main__':
    data_dir = Path(__file__).parent.parent.resolve().joinpath('data')
    loader = HatexplainLoader(data_dir=data_dir)
    train_df, val_df, test_df = loader.get_splits()
    get_sparsity_level(df=train_df)
