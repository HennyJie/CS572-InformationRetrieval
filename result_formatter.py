import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import shutil

folder = 'semi_performance_results'
output_folder = Path('format_result')

mdcg = ['NDCG@1', 'NDCG@2', 'NDCG@3', 'NDCG@4', 'NDCG@5',
        'NDCG@6', 'NDCG@7', 'NDCG@8', 'NDCG@9', 'NDCG@10']
p = ['P@1', 'P@2', 'P@3', 'P@4', 'P@5',
     'P@6', 'P@7', 'P@8', 'P@9', 'P@10', 'MAP']
index = ["Fold1", "Fold2", "Fold3", "Fold4", "Fold5", "Avg  "]
datasets = ['MQ2007semi', 'MQ2008semi']
algorithms = ['Mart', 'LambdaMart']
dataset_type = ['test', 'validation', 'training']

file_list = os.listdir(folder)
df_list = []
for dataset in datasets:
    for algorithm in algorithms:
        df_list += [f"{dataset}.{algorithm}.{d}" for d in dataset_type]
df_list += [c+'.new' for c in df_list]


dfs = dict((c, pd.DataFrame(columns=mdcg+p, index=index)) for c in df_list)

for _, df in dfs.items():
    df.index.name = 'Para'

for file_path in file_list:

    accs = {}
    paras = file_path.split('.')

    dataset, fold, algorithm, metric = paras[:4]
    row_index = int(re.findall('\d', fold)[0])

    baseline = paras[4] != 'Predict'
    with open(folder/Path(file_path), 'r') as f:
        for line in f.readlines():

            for c in dataset_type:
                pattern = re.compile(rf"{metric}\s+on {c} data:\s+(\d+\.\d+)")
                acc = pattern.findall(line)
                if acc:
                    dfs[f"{dataset}.{algorithm}."+c +
                        ('' if baseline else '.new')].loc[fold, metric] = float(acc[0])
for _, df in dfs.items():
    for col in mdcg + p:
        m = f"{df[col].mean():.5f}"
        if m == 'nan':
            m = ''
        df.loc['Avg  ', col] = m


if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.mkdir(output_folder)

for dataset in datasets:
    for algorithm in algorithms:
        for new in [True, False]:
            file_name = f"{dataset}.{algorithm}" + ('' if new else '.new')
            with open(output_folder/f'{file_name}.csv', 'a') as f:
                f.write(f'Algorithm: {algorithm}\nDataset:   {dataset}\n\n')

                for dt in dataset_type:
                    df_name = f"{dataset}.{algorithm}.{dt}" + \
                        ('' if new else '.new')
                    if dt == 'test':
                        dt = 'testing'
                    f.write(f'Performance on {dt} set\n')
                    dfs[df_name][mdcg].to_csv(f, sep='\t', float_format='%.4f')
                    f.write('\n')
                    tmp = dfs[df_name][p]
                    tmp.columns = [n+"   " for n in tmp.columns]
                    tmp.to_csv(f, sep='\t', float_format='%.4f')
                    f.write('\n')
