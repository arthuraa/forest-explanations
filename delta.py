import os
import pandas as pd
import numpy as np
import random

column_names = ["roll_rate", "pitch_rate", "curr_pitch", "curr_roll",
                "diff_roll_rate", "target"]

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, 'data', 'delta_ailerons.data')

original = pd.read_csv(path, names = column_names, sep = ' ')
target = original['target']
del original['target']
