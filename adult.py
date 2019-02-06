import os
import pandas as pd
import numpy as np
import random

column_names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                "hours_per_week", "country", "target"]

path = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(path, 'data', 'adult.data')
test_path = os.path.join(path, 'data', 'adult.test')

original_train = pd.read_csv(train_path,
                             names = column_names,
                             sep = r'\s*,\s*',
                             engine = 'python',
                             na_values = '?',
                             true_values = ['>50K'],
                             false_values = ['<=50K'])
original_test = pd.read_csv(test_path,
                            names = column_names,
                            comment = '|',
                            sep = r'\s*,\s*',
                            engine = 'python',
                            na_values = '?',
                            true_values = ['>50K.'],
                            false_values = ['<=50K.'])

original = pd.concat([original_train, original_test], ignore_index=True)

weights = original['fnlwgt']

del original['fnlwgt']
del original["education"]

binary = pd.get_dummies(original)
labels = binary["target"]
binary = binary[binary.columns.difference(["target"])]
