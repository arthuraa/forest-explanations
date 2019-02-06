import os
import pandas as pd
import numpy as np
import random

column_names = ["fLength",
                "fWidth",
                "fSize",
                "fConc",
                "fConc1",
                "fAsym",
                "fM3Long",
                "fM3Trans",
                "fAlpha",
                "fDist",
                "target"]

descriptions = [
    "major axis of ellipse [mm]",
    "minor axis of ellipse [mm]",
    "10-log of sum of content of all pixels [in #phot]",
    "ratio of sum of two highest pixels over fSize  [ratio]",
    "ratio of highest pixel over fSize  [ratio]",
    "distance from highest pixel to center, projected onto major axis [mm]",
    "3rd root of third moment along major axis  [mm]",
    "3rd root of third moment along minor axis  [mm]",
    "angle of major axis with vector to origin [deg]",
    "distance from origin to center of ellipse [mm]",
    "gamma (signal), hadron (background)"
]

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, 'data', 'magic.data')

original = pd.read_csv(path, names = column_names, sep = ',')
target = original['target']
del original['target']
