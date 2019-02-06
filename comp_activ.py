import os
import pandas as pd
import numpy as np
import random

column_names = ["time",
                "lread",
                "lwrite",
                "scall",
                "sread",
                "swrite",
                "fork",
                "exec",
                "rchar",
                "wchar",
                "pgout",
                "ppgout",
                "pgfree",
                "pgscan",
                "atch",
                "pgin",
                "ppgin",
                "pflt",
                "vflt",
                "runqsz",
                "runocc",
                "freemem",
                "freeswap",
                "usr",
                "sys",
                "wio",
                "idle"]

path = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(path, 'data', 'comp_activ.data')
original = pd.read_csv(path, names = column_names, sep = ' ')
