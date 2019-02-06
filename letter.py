import os
import pandas as pd
import numpy as np
import random

column_names = [
    "target",
    "x-box",
    "y-box",
    "width",
    "high",
    "onpix",
    "x-bar",
    "y-bar",
    "x2bar",
    "y2bar",
    "xybar",
    "x2ybr",
    "xy2br",
    "x-ege",
    "xegvy",
    "y-ege",
    "yegvx"
]

descriptions = [
        "capital letter	(26 values from A to Z)",
        "horizontal position of box	(integer)",
        "vertical position of box	(integer)",
        "width of box			(integer)",
        "height of box			(integer)",
        "total # on pixels		(integer)",
        "mean x of on pixels in box	(integer)",
        "mean y of on pixels in box	(integer)",
        "mean x variance                (integer)",
        "mean y variance		(integer)",
        "mean x y correlation		(integer)",
        "mean of x * x * y		(integer)",
        "mean of x * y * y		(integer)",
        "mean edge count left to right	(integer)",
        "correlation of x-ege with y	(integer)",
        "mean edge count bottom to top	(integer)",
        "correlation of y-ege with x	(integer)"
]

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, 'data', 'letter.data')

original = pd.read_csv(path, names = column_names, sep = ',')
target = original['target']
del original['target']
