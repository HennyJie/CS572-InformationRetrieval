import pandas as pd

for dataset in ['MQ2007semi', 'MQ2008semi']:
    for i in range(1, 6):
        folder_path = f'/home/xuankan/Documents/CS572-InformationRetrieval/{dataset}/Fold{i}'

        data = pd.read_csv(
            folder_path + '/train.txt', header=None, sep='\s+')

        filter_bool = data.iloc[:, 0] == -1

        labeled = data[~filter_bool]

        labeled.to_csv(folder_path + '/train_labeled.txt',
                       sep=' ', header=False, index=False)

        print(labeled.describe())

        unlabel = data[filter_bool]

        print(unlabel.describe())
        unlabel.to_csv(folder_path + '/train_unlabel.txt',
                       sep=' ', header=False, index=False)
