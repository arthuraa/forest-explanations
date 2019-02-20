import os
import pandas as pd
import numpy as np
import random

path = os.path.dirname(os.path.realpath(__file__))
path_train = os.path.join(path, 'data', 'gisette_train.data')
path_train_labels = os.path.join(path, 'data', 'gisette_train.labels')
path_valid = os.path.join(path, 'data', 'gisette_valid.data')
path_valid_labels = os.path.join(path, 'data', 'gisette_valid.labels')

original = pd.read_csv(path_train, header = None, sep = r'\s+')
target = pd.read_csv(path_train_labels, header = None)[0]
validation = pd.read_csv(path_valid, header = None, sep = r'\s+')
validation_target = pd.read_csv(path_valid_labels, header = None)[0]
