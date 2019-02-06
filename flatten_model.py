from multiprocessing import Pool
import pickle
import adult
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from category_encoders.one_hot import OneHotEncoder as CEOneHotEncoder
from xgboost import XGBClassifier

n_estimators = 6000
max_depth = 2

def traverse_tree(tree, tests):
    feature = tree['split']
    threshold = tree['split_condition'] if 'split_condition' in tree else 1
    children = tree['children']
    thresholds = tests[feature] if feature in tests else set([])
    thresholds.add(threshold)
    tests[feature] = thresholds
    if 'children' in children[0]:
        tests = traverse_tree(children[0], tests)
    if 'children' in children[1]:
        tests = traverse_tree(children[1], tests)
    return tests

def boost_predicates(model):
    trees = [json.loads(t) for t in model._Booster.get_dump(dump_format = 'json')]
    tests = {}
    for tree in trees:
        tests = traverse_tree(tree, tests)
    return tests

def encode_points(predicates, X):
    X_proj = {}

    for feature, thresholds in predicates.items():
        thresholds = list(thresholds)
        thresholds.sort()
        if len(thresholds) > 1:
            for i in range(len(thresholds)):
                key = feature + ("_%02d_%.1f" % (i, thresholds[i]))
                mask = X[feature] >= thresholds[i]
                X_proj[key] = mask
        else:
            X_proj[feature] = X[feature] >= thresholds[0]

    return pd.DataFrame(X_proj)

def process(i):
    file_name = 'adult-%dest-%dmax_depth.%d.model' % (n_estimators, max_depth, i)
    result = pickle.load(open(file_name, 'rb'))
    model = result['model']
    X_train, y_train = result['train']
    X_test, y_test = result['test']
    predicates = boost_predicates(model)
    X_train_enc = encode_points(predicates, X_train)
    X_test_enc = encode_points(predicates, X_test)
    lin_model = LogisticRegression().fit(X_train_enc, y_train)
    result_file_name = 'flatten-%dest-%dmax_depth.%d.model' % (n_estimators, max_depth, i)
    result = {'model': lin_model,
              'train_score': lin_model.score(X_train_enc, y_train),
              'test_score': lin_model.score(X_test_enc, y_test)}
    pickle.dump(result, open(result_file_name, 'wb'))

p = Pool(3)
p.map(process, range(20))
