import warnings
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost
from xgboost import XGBClassifier
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss
from category_encoders.one_hot import OneHotEncoder as CEOneHotEncoder
from multiprocessing import Pool
from itertools import product
import pickle

import adult
import communities
import gisette
import letter
import physics
import magic
import spambase

import boosting_compression.stochastic as stochastic

warnings.filterwarnings('ignore', category = FutureWarning)

data = {}

def enc(X):
    e = CEOneHotEncoder(use_cat_names = True, handle_unknown = 'ignore').fit(X)
    return e.transform(X)

adult_features = adult.original_train
adult_features = adult_features.assign(male = adult_features['sex'] == 'Male')
adult_target = adult_features['target']
data['adult'] = (enc(adult_features.drop(['target', 'sex', 'fnlwgt'], axis=1, inplace=False)),
                 adult_target)
data['communities'] = (communities.data.drop(communities.uninformative, axis=1),
                       communities.target >= 0.5)
data['gisette'] = (gisette.original, gisette.target)
data['letter'] = (letter.original, letter.target)
data['physics'] = (physics.original, physics.target)
data['magic'] = (magic.original, magic.target)
data['spambase'] = (spambase.original, spambase.target)

n_estimators = 100
n_trials = 4
n_processes = 5
n_steps = 2
depths = [1, 2]
population_size = 100
pool_size = 50
pooling_options = [False, True]

def task(arg):
    ((name, (X, y)), depth, pooling, i) = arg
    exp_id = "%s, depth %d, pooling %d, trial %d" % (name, depth, pooling, i)
    print(time.strftime('%X'), "Beginning", exp_id)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    model = XGBClassifier(max_depth = depth, n_estimators = n_estimators)
    model = model.fit(X_train, y_train)
    print(time.strftime('%X'), "Trained boosting model", exp_id)
    predicates = stochastic.compress(exp_id, X_train, X_test, y_train, y_test, model, n_steps, pooling,
                                     population_size, pool_size)
    original_score = model.score(X_test, y_test)
    X_proj_train = np.concatenate([stochastic.apply_pred(X_train,pred).values.reshape(-1,1) for pred in predicates], axis = 1)
    X_proj_test = np.concatenate([stochastic.apply_pred(X_test,pred).values.reshape(-1,1) for pred in predicates], axis = 1)
    logreg = LogisticRegression(solver = 'lbfgs', multi_class = 'auto').fit(X_proj_train, y_train)
    simplified_score = logreg.score(X_proj_test, y_test)
    print(time.strftime('%X'), "Finished %s, score %.3f -> %.3f" % (exp_id, original_score, simplified_score))
    return (name, depth, pooling, i, original_score, simplified_score, model, predicates)

# pool = Pool(n_processes)

# results = pool.map(task, product(data.items(), depths, pooling_options, range(n_trials)))

data = {'gisette': data['gisette']}

results = [task(arg) for arg in product(data.items(), depths, [True, False], range(n_trials))]

pickle.dump(results, open('results/boosting_compression', 'wb'))
