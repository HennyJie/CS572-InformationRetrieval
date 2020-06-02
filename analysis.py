import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import shutil

folder = 'analysis'


data = defaultdict(list)

file_list = os.listdir(folder)

for file_path in file_list:
    paras = file_path.split('.')
    dataset, fold, algorithm, metric = paras[:4]
    if fold != 'Fold1':
        continue

    with open(folder/Path(file_path), 'r') as f:
        for line in f.readlines():
            acc = re.findall(r'\((\S+\d+\.\d+)\%\)', line)
            if len(acc) != 0:
                data[f'{dataset}_{algorithm}_{metric}'].append(
                    float(acc[0]))
print(data)
for key, v in data.items():
    data[key] = sum(v)/len(v)
    print(f'{key}\t{data[key]}')
