import logging
from pathlib import Path

import lightning as L
import numpy as np
import wandb
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification.f_beta import F1Score

from components.callbacks import PycharmProgressBar
from components.data_loader import ToyLoader
from components.metrics import BinaryHighlightF1Score, SelectionRate, SelectionSize
from components.model import MGRHlModel
from components.processing import HighlightDataset, OneHotEmbedderCollator, HighlightExtractor
from configurations.base import ConfigKey
from configurations.model import MGRConfig
from modeling.spp import MGR

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Config
    # -------------

    save_path = Path(__file__).parent.parent.resolve().joinpath('results', 'toy', 'mgr')
    if not save_path.exists():
        save_path.mkdir(parents=True)

    ckpt_path = save_path.joinpath('checkpoints')
    if not ckpt_path.exists():
        ckpt_path.mkdir(parents=True)

    data_dir = Path(__file__).parent.parent.resolve().joinpath('data')

    config = MGRConfig.from_config(key=ConfigKey(dataset='toy', tags={'spp', 'mgr'}))

    trainer_args = {
        'accelerator': 'auto',
        'devices': 1,
        'accumulate_grad_batches': 1,
        'max_epochs': 500,
    }

    # -------------
    seed_everything(seed=15000)

    loader = ToyLoader(data_dir=data_dir)
    train_df, val_df, test_df = loader.get_splits()

    collator = OneHotEmbedderCollator()
    collator.fit(df=train_df)

    train_data = HighlightDataset(texts=train_df.text.values,
                                  masks=train_df['mask'].values,
                                  labels=train_df.label.values,
                                  highlights=train_df.highlight.values,
                                  sample_ids=train_df.sample_id.values)
    train_data = DataLoader(train_data,
                            shuffle=True,
                            batch_size=config.batch_size,
                            collate_fn=collator)

    val_data = HighlightDataset(texts=val_df.text.values,
                                masks=val_df['mask'].values,
                                labels=val_df.label.values,
                                highlights=val_df.highlight.values,
                                sample_ids=val_df.sample_id.values)
    val_data = DataLoader(val_data,
                          batch_size=config.batch_size,
                          collate_fn=collator)

    test_data = HighlightDataset(texts=test_df.text.values,
                                 masks=test_df['mask'].values,
                                 labels=test_df.label.values,
                                 highlights=test_df.highlight.values,
                                 sample_ids=test_df.sample_id.values)
    test_data = DataLoader(test_data,
                           batch_size=config.batch_size,
                           collate_fn=collator)

    extractor = HighlightExtractor(
        text_splitter=lambda t: list(t),
        text_merger=lambda t: ''.join(t)
    )

    metrics = {}
    for seed in config.seeds:
        seed_everything(seed=seed)

        seed_ckpt_path = ckpt_path.joinpath(f'seed={seed}')
        seed_ckpt_path.mkdir(parents=True, exist_ok=True)

        model = MGR(vocab_size=collator.vocab_size,
                    embedding_dim=config.embedding_dim,
                    hidden_size=config.hidden_size,
                    classification_head=config.classification_head,
                    selection_head=config.selection_head,
                    num_generators=config.num_generators,
                    dropout_rate=config.dropout_rate,
                    embedding_matrix=collator.embedding_matrix,
                    temperature=config.temperature,
                    freeze_embeddings=config.freeze_embeddings)

        model = MGRHlModel(model=model,
                           num_classes=config.num_classes,
                           optimizer_class=config.optimizer_class,
                           optimizer_kwargs=config.optimizer_kwargs,
                           val_classification_metrics=MetricCollection({
                               'f1': F1Score(task='multiclass', average='macro', num_classes=3)}),
                           test_classification_metrics=MetricCollection({
                               'f1': F1Score(task='multiclass', average='macro', num_classes=3)}),
                           val_highlight_metrics=MetricCollection({
                               'hl_pos_f1': BinaryHighlightF1Score(pos_label=1),
                               'hl_rate': SelectionRate(),
                               'hl_size': SelectionSize()
                           }),
                           test_highlight_metrics=MetricCollection({
                               'hl_pos_f1': BinaryHighlightF1Score(pos_label=1),
                               'hl_rate': SelectionRate(),
                               'hl_size': SelectionSize()
                           }),
                           add_sparsity_loss=config.add_sparsity_loss,
                           sparsity_level=config.sparsity_level,
                           sparsity_coefficient=config.sparsity_coefficient,
                           add_continuity_loss=config.add_continuity_loss,
                           continuity_coefficient=config.continuity_coefficient,
                           classification_coefficient=config.classification_coefficient)

        should_train = not any(seed_ckpt_path.glob('*.ckpt'))

        if should_train:
            wandb_logger = WandbLogger(project='hlext-toy', name='mgr')
        else:
            wandb_logger = None

        trainer = L.Trainer(**trainer_args,
                            callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=30),
                                       ModelCheckpoint(monitor='val_loss', mode='min', dirpath=seed_ckpt_path),
                                       PycharmProgressBar()],
                            deterministic=True,
                            logger=wandb_logger
                            )

        if should_train:
            trainer.fit(model,
                        train_dataloaders=train_data,
                        val_dataloaders=val_data)
        else:
            logging.info('Found existing checkpoint! Skipping model training...')

        ckpt_filepath = seed_ckpt_path.glob('*.ckpt').__next__()

        model = MGRHlModel.load_from_checkpoint(checkpoint_path=ckpt_filepath,
                                                model=model.model,
                                                val_classification_metrics=MetricCollection({
                                                    'f1': F1Score(task='multiclass', average='macro',
                                                                  num_classes=3)}),
                                                test_classification_metrics=MetricCollection({
                                                    'f1': F1Score(task='multiclass', average='macro',
                                                                  num_classes=3)}),
                                                val_highlight_metrics=MetricCollection({
                                                    'hl_pos_f1': BinaryHighlightF1Score(pos_label=1),
                                                    'hl_rate': SelectionRate(),
                                                    'hl_size': SelectionSize()
                                                }),
                                                test_highlight_metrics=MetricCollection({
                                                    'hl_pos_f1': BinaryHighlightF1Score(pos_label=1),
                                                    'hl_rate': SelectionRate(),
                                                    'hl_size': SelectionSize()
                                                }))
        seed_everything(seed=seed)

        # Metrics
        val_metrics = trainer.validate(model=model, dataloaders=val_data)[0]
        logging.info(f'Validation metrics: {val_metrics}')

        model.enable_storing_predictions()

        test_metrics = trainer.test(model=model, dataloaders=test_data)[0]
        test_predictions = model.test_predictions
        logging.info(f'Test metrics: {test_metrics}')

        for metric_name, metric_value in val_metrics.items():
            metrics.setdefault('validation', {}).setdefault(metric_name, []).append(metric_value)
        for metric_name, metric_value in test_metrics.items():
            metrics.setdefault('test', {}).setdefault(metric_name, []).append(metric_value)

        # Predictions
        extractor(predictions=test_predictions,
                  vocabulary=collator.vocabulary,
                  save_path=save_path,
                  seed=seed)

        model.flush_predictions()
        model.disable_storing_predictions()

        wandb.finish()

    # Averaging
    for split_name in ['validation', 'test']:
        metric_names = list(metrics[split_name].keys())
        for metric_name in metric_names:
            metric_values = np.array(metrics[split_name][metric_name]).reshape(len(config.seeds), -1)
            per_seed_avg = metric_values.mean(axis=-1)
            per_seed_std = metric_values.std(axis=-1)
            avg = per_seed_avg.mean(axis=-1)
            std = per_seed_avg.std(axis=-1)
            metrics[split_name][f'per_seed_avg_{metric_name}'] = (per_seed_avg, per_seed_std)
            metrics[split_name][f'avg_{metric_name}'] = (avg, std)

    logging.info(metrics)
    np.save(save_path.joinpath('metrics.npy').as_posix(), metrics)
