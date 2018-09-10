import pandas as pd

from redditscore.models import fasttext_mod

fasttext_model = fasttext_mod.FastTextModel(loss='softmax', lr=0.1, thread=16)

param_grid = {'epoch': [1, 3, 5, 10], 'wordNgrams': [1, 2], 'dim': [100], 'bucket': [
    2000000], 'minCount': [1, 10, 25, 50], 't': [0.0001, 0.001, 0.01]}

train_file = '/scratch/en919/FINAL_DATA/data_train_grouped.txt'
test_file = '/scratch/en919/FINAL_DATA/data_test_grouped.txt'

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

X_train = df_train['clean_version']
y_train = df_train['subgroup']
X_test = df_test['clean_version']
y_test = df_test['subgroup']

fasttext_model.tune_params(X_train, y_train, cv=5,
                           param_grid=param_grid, scoring='accuracy')
