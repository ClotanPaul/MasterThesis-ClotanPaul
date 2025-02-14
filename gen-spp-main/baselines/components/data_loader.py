import csv
import json
from pathlib import Path
from statistics import mode

import numpy as np
import pandas as pd
import torch as th
from django.utils.functional import cached_property
from torch.utils.data import Dataset


class ToyLoader:

    def __init__(
            self,
            data_dir: Path
    ):
        self.data_dir = data_dir
        self.df_dir = data_dir.joinpath('toy', 'toy_sequential_dataset_easy.pkl')

    @cached_property
    def data(
            self
    ) -> pd.DataFrame:
        df = pd.read_pickle(self.df_dir)
        df['mask'] = self.build_attention_mask(df=df)
        df['highlight'] = self.parse_highlights(df=df)
        df['sample_id'] = np.arange(df.shape[0])
        return df

    def build_attention_mask(
            self,
            df
    ):
        masks = []
        for text in df.text.values:
            masks.append([1] * len(list(text)))
        return masks


    def parse_highlights(
            self,
            df
    ):
        highlights = []
        for indexes, text in zip(df['structure_indexes'].values, df.text.values):
            highlight = np.zeros((len(list(text)),))
            highlight[indexes] = 1
            highlights.append(highlight.tolist())

        return highlights

    def get_splits(
            self
    ):
        train_count = int(self.data.shape[0] * 0.80)
        val_count = int(train_count * 0.20)

        train_df = self.data[:train_count]
        test_df = self.data[train_count:]
        val_df = train_df.sample(n=val_count)
        train_df = train_df[~train_df.index.isin(val_df.index.values)]

        return train_df, val_df, test_df


class HatexplainLoader:

    def __init__(
            self,
            data_dir: Path
    ):
        self.data_dir = data_dir.joinpath('hatexplain')

    @cached_property
    def data(
            self
    ) -> pd.DataFrame:
        train_df = pd.read_pickle(self.data_dir.joinpath('train.pkl'))
        train_df['split'] = ['train'] * train_df.shape[0]

        val_df = pd.read_pickle(self.data_dir.joinpath('val.pkl'))
        val_df['split'] = ['val'] * val_df.shape[0]

        test_df = pd.read_pickle(self.data_dir.joinpath('test.pkl'))
        test_df['split'] = ['test'] * test_df.shape[0]

        df = pd.concat((train_df, val_df, test_df))
        df = df[df['post_tokens'].map(len) <= 30]
        df.rename(columns={'post_tokens': 'text'}, inplace=True)

        df['mask'] = self.build_attention_mask(df=df)
        df['highlight'] = self.parse_highlights_majority(df=df)
        df['label'] = self.parse_labels(df=df)
        df['sample_id'] = np.arange(df.shape[0])
        return df

    def build_attention_mask(
            self,
            df
    ):
        masks = []
        for text in df.text.values:
            masks.append([1] * len(text))
        return masks

    def parse_labels(
            self,
            df
    ):
        labels = []
        for annotation in df['annotators'].values:
            votes = annotation["label"]
            votes = [x if x != 2 else 0 for x in votes]  # replace labels 2 with labels 0
            majority = mode(votes)
            labels.append(majority)

        return labels

    def parse_highlights_majority(
            self,
            df
    ):
        highlights = []
        for highlight_group, text in zip(df['rationales'].values, df['text'].values):
            total_tokens = len(text)

            if len(highlight_group) == 1:
                highlight = highlight_group[0][:total_tokens]
            elif len(highlight_group) > 1:
                n_annotators = len(highlight_group)
                min_vote = n_annotators / 2.0
                highlight_group = np.array([seq[:total_tokens] for seq in highlight_group])
                votes = np.sum(highlight_group, axis=0)
                highlight = np.where(votes > min_vote, 1.0, 0.0)
            else:
                highlight = [0] * len(text)

            highlights.append(highlight)

        return highlights

    def get_splits(
            self
    ):
        train_df = self.data[self.data.split == 'train']
        val_df = self.data[self.data.split == 'val']
        test_df = self.data[self.data.split == 'test']

        return train_df, val_df, test_df

