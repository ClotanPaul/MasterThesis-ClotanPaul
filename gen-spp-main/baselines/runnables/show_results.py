from pathlib import Path

import numpy as np
import pandas as pd


def parse_metric(
        metric: str,
        metric_name: str
):
    if metric == '-':
        return metric

    is_percentage = metrics_info[metric_name]['is_percentage']
    mean, std = metric.split(' +/- ')
    mean = float(mean)
    std = float(std)
    parsed_metric = f'${mean * 100 if is_percentage else mean:.2f}' + '_{\pm ' + f'{std * 100 if is_percentage else std:.2f}' + '}$'
    return parsed_metric


def format_results(
        df
):
    formatted = [
        ' & '.join([parse_metric(metric=value, metric_name=key) for key, value in row.to_dict().items() if key not in ['dataset', 'model']])
        for _, row in df.iterrows()]
    df['Formatted'] = formatted
    return df


if __name__ == '__main__':
    results_path = Path(__file__).parent.parent.resolve().joinpath('results')
    results = {}
    metric_names = [
        'f1',
        'hl_pos_f1',
        'hl_rate',
        'hl_size',
        'SP',
        'CT',
    ]
    metrics_info = {
        'f1': {'is_percentage': True},
        'hl_pos_f1': {'is_percentage': True},
        'hl_size': {'is_percentage': False},
        'hl_rate': {'is_percentage': True},
        'SP': {'is_percentage': False},
        'CT': {'is_percentage': False},
    }

    for results_file in results_path.rglob('metrics.npy'):
        dataset_name = results_file.parent.parent.name
        test_name = results_file.parent.name

        metrics = np.load(results_file.as_posix(), allow_pickle=True).item()

        results.setdefault('dataset', []).append(dataset_name)
        results.setdefault('model', []).append(test_name)
        for metric_name in metric_names:
            if f'avg_test_{metric_name}' in metrics['test']:
                metric_value = metrics['test'][f'avg_test_{metric_name}']
                parsed_metric_value = f'{metric_value[0]:.4f} +/- {metric_value[1]:.4f}'
                results.setdefault(f'{metric_name}', []).append(parsed_metric_value)
            else:
                results.setdefault(f'{metric_name}', []).append('-')

    df = pd.DataFrame.from_dict(results)
    df = format_results(df)
    print(df.to_string())
