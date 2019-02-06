import os
import pandas as pd
import numpy as np
import random

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, 'data', 'physics_train.data')

original = pd.read_csv(path, header = None, sep = r'\s+')
del original[0]
target = original[1]
del original[1]
