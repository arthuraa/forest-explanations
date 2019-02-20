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

n_estimators = 2000
n_trials = 4
n_processes = 5
n_steps = 7
depths = [1, 2]
population_size = 2000
pool_size = 1000
fresh_size = population_size - pool_size

def traverse_aux(tree, feature_type, predicates, path):
    if 'leaf' in tree:
        predicates.append(path)
    else:
        feature = tree['split']
        if feature_type == np.dtype('int'):
            feature = int(feature)
        threshold = tree['split_condition'] if 'split_condition' in tree else 1
        traverse_aux(tree['children'][0], feature_type, predicates,
                     path + ((feature, threshold, True),))
        traverse_aux(tree['children'][1], feature_type, predicates,
                     path + ((feature, threshold, False),))

def traverse(trees, feature_type):
    predicates = []
    for tree in trees:
        traverse_aux(tree, feature_type, predicates, ())
    return predicates

def apply_pred(X, pred):
    mask = np.ones(len(X), dtype = bool)
    for feature, threshold, direction in pred:
        if direction:
            mask = mask & (X[feature] < threshold)
        else:
            mask = mask & (X[feature] >= threshold)
    return mask

def sample(l, size):
    idx = np.random.choice(range(len(l)), size, replace = False)
    return [l[i] for i in idx]

def simplify(exp_id, X_train, X_test, y_train, y_test, model, steps, pooling):
    dump = model.get_booster().get_dump(dump_format = 'json')
    trees = [json.loads(tree) for tree in dump]
    predicates = traverse(trees, X_train.columns.dtype)
    selected = []
    X_proj_train = np.zeros((len(X_train),0))
    X_proj_test = np.zeros((len(X_test),0))
    if pooling:
        preds = sample(predicates, pool_size)
    for step in range(steps):
        if pooling:
            fresh = sample(predicates, fresh_size)
            preds = preds + fresh
        else:
            preds = sample(predicates, population_size)
        res = []
        best_pred = None
        best_score = None
        best_loss = None
        best_X_proj_train = None
        best_logreg = None
        for i, pred in enumerate(preds):
            mask = apply_pred(X_train, pred).values.reshape(-1, 1)
            X_proj_train_aug = np.concatenate((X_proj_train, mask), axis = 1)
            X_train_train, X_train_test, y_train_train, y_train_test = \
              train_test_split(X_proj_train_aug, y_train, test_size = 0.1)
            logreg = LogisticRegression(solver = 'lbfgs', multi_class = 'auto')
            logreg = logreg.fit(X_train_train, y_train_train)
            y_pred = logreg.predict_proba(X_train_test)
            loss   = log_loss(y_train_test, y_pred)
            res.append((loss, pred))
            if best_loss == None or loss < best_loss:
                best_pred = pred
                best_loss = loss
                best_score = logreg.score(X_train_test, y_train_test)
                best_X_proj_train = X_proj_train_aug
                best_logreg = logreg
        res.sort()
        losses = np.array([r[0] for r in res])

        pool = [r[1] for r in res[:pool_size]]
        print(time.strftime('%X'), "Exp. %s, Step %d" % (exp_id, step))
        print("Added predicate", best_pred)
        print("Loss %.4f (mean %.4f, sd %.4f)" % (best_loss, losses.mean(), losses.std()))
        print("Score %.4f" % best_score)
        X_proj_train = best_X_proj_train
        mask = apply_pred(X_test, best_pred).values.reshape(-1,1)
        X_proj_test = np.concatenate((X_proj_test, mask), axis = 1)
        print("Test score %.4f" % best_logreg.score(X_proj_test, y_test))
        selected.append(best_pred)
    return selected

def task(arg):
    ((name, (X, y)), depth, pooling, i) = arg
    exp_id = "%s, depth %d, pooling %d, trial %d" % (name, depth, pooling, i)
    print(time.strftime('%X'), "Beginning", exp_id)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    model = XGBClassifier(max_depth = depth, n_estimators = n_estimators)
    model = model.fit(X_train, y_train)
    print(time.strftime('%X'), "Trained boosting model", exp_id)
    predicates = simplify(exp_id, X_train, X_test, y_train, y_test, model, n_steps, pooling)
    original_score = model.score(X_test, y_test)
    X_proj_train = np.concatenate([apply_pred(X_train,pred).values.reshape(-1,1) for pred in predicates], axis = 1)
    X_proj_test = np.concatenate([apply_pred(X_test,pred).values.reshape(-1,1) for pred in predicates], axis = 1)
    logreg = LogisticRegression(solver = 'lbfgs', multi_class = 'auto').fit(X_proj_train, y_train)
    simplified_score = logreg.score(X_proj_test, y_test)
    print(time.strftime('%X'), "Finished %s, score %.3f -> %.3f" % (exp_id, original_score, simplified_score))
    return (name, depth, pooling, i, original_score, simplified_score, model, predicates)

# pool = Pool(n_processes)

# results = pool.map(task, product(data.items(), depths, [True, False], range(n_trials)))

data = {'gisette': data['gisette']}

results = [task(arg) for arg in product(data.items(), depths, [True, False], range(n_trials))]

pickle.dump(results, open('results/boosting_compression', 'wb'))
