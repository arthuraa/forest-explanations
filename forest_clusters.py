from __future__ import print_function

import os
import pandas as pd
import numpy as np
import random
from scipy import sparse

from sklearn import linear_model, model_selection, metrics, tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import communities

import cgi
from subprocess import call

def buckets(X):
    return X.round(1).value_counts(normalize = True, dropna = False).sort_index()

def total_variation(X1, X2):
    return X1.subtract(X2, fill_value = 0.0).abs().sum() / 2

# TODO Allow to mix categorical and non-categorical features
#
# Maybe this wouldn't be a problem if we allowed arbitrary encoding functions
# instead of just one hot...
class Forest():
    def __init__(self, n_estimators=10, categorical_features = []):
        self.encoder = OneHotEncoder(categorical_features = categorical_features)
        self.forest = RandomForestClassifier(n_estimators = n_estimators)

    def fit(self, X, y):
        self.encoder.fit(X)
        self.forest.fit(self.encoder.transform(X), y)
        return self

    def predict(self, X):
        return self.forest.predict(self.encoder.transform(X))

    def votes(self, X):
        # FIXME There is probably a more clever way of doing this
        X_enc = self.encoder.transform(X)
        predictions = [t.predict(X_enc) for t in self.forest.estimators_]
        votes = pd.DataFrame.from_dict(dict(zip(range(len(self.forest.estimators_)), predictions)))
        return votes.transpose().sum()

    def score(self, X, y):
        return self.forest.score(self.encoder.transform(X), y)

    def apply(self, X):
        return self.forest.apply(X)
# TODO
# - Allow subset of columns?
# - Do something about categorical features
class ForestClusters():
    def __init__(self, model, n_clusters = 20):
        self.model = model
        self.encoder = OneHotEncoder()
        self.kmeans = KMeans(n_clusters = n_clusters)

    def fit(self, X):
        leaves = self.model.apply(X)
        self.encoder.fit(leaves)
        self.kmeans.fit(self.encoder.transform(leaves))

    def predict(self, X):
        leaves = self.model.apply(X)
        leaves = self.encoder.transform(leaves)
        return self.kmeans.predict(leaves)

    def transform(self, X):
        leaves = self.model.apply(X)
        leaves = self.encoder.transform(leaves)
        return self.kmeans.transform(leaves)

    def predict_transform(self, X):
        leaves = self.model.apply(X)
        leaves = self.encoder.transform(leaves)
        return self.kmeans.predict(leaves), self.kmeans.transform(leaves)

    def summarize(self, X, y, path="."):
        clusters, distances = self.predict_transform(X)
        votes = self.model.votes(X)
        plt.figure(1)
        votes.hist()
        plt.savefig(os.path.join(path, "votes.png"))
        plt.clf()

        with open(os.path.join(path, "clusters.org"), "w") as f:
            print("* Feature Distributions\n", file = f)

            for col in X.columns:
                print("** %s" % col, file = f)
                print(buckets(X[col]), file = f)

            print("* Clusters\n", file = f)

            for i in range(self.kmeans.n_clusters):
                idx = clusters == i
                cluster_votes = votes[idx]
                mean_votes = cluster_votes.mean()
                dists_to_centroid = pd.DataFrame(distances[idx])[i]
                mean_dist_to_centroid = dists_to_centroid.mean()
                cluster = X[idx]

                # Save histogram of weights in cluster
                cluster_votes.hist()
                plt.savefig(os.path.join(path, "cluster-votes-%02d.png" % i))
                plt.clf()

                # Represent cluster as small decision tree
                t = tree.DecisionTreeClassifier(max_depth = 6,
                                                max_leaf_nodes = 7, criterion = 'gini')
                t = t.fit(X, idx)
                out_file = os.path.join(path, "cluster-tree-%02d" % i)
                tree.export_graphviz(t, out_file = out_file + ".dot",
                                     feature_names = [cgi.escape(s) for s in X.columns],
                                     filled = True,
                                     rounded = True,
                                     special_characters = True)
                with open(out_file + ".png", "w") as dot_file:
                    call(["dot", "-Tpng", out_file + ".dot"], stdout=dot_file)
                call(["rm", out_file + ".dot"])

                acc = self.model.score(X[idx], y[idx])

                # Print some cluster statistics and samples
                print("** Cluster %02d, size = %d, mean votes = %.03f, acc = %.03f, dist to center = %.03f\n" % \
                      (i, len(cluster), mean_votes, acc, mean_dist_to_centroid), file = f)
                print("*** Statistics\n", file = f)
                tvs = pd.DataFrame([(col, total_variation(buckets(X[col]),
                                                          buckets(X[idx][col])))
                                    for col in X.columns])
                tvs = tvs.sort_values(1, ascending = False)
                for i in range(len(tvs)):
                    print("**** %s: %.04f\n%s\n" % (tvs.iloc[i, 0], tvs.iloc[i, 1], communities.description[tvs.iloc[i,0]]), file = f)
                    print(buckets(X[tvs.iloc[i, 0]][idx]), file = f)
                    print("\nvs.\n", file = f)
                    print(buckets(X[tvs.iloc[i, 0]]), file = f)
                print("\n*** Samples\n", file = f)
                s = cluster.sample(n = min(5, len(cluster)))
                for p in range(len(s)):
                    with pd.option_context('display.max_rows', None):
                        print(s.iloc[p], file = f)
                        print("Weight = %.00f\n" % votes[X.index.get_loc(s.iloc[p].name)], file = f)
