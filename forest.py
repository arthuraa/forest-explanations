from __future__ import print_function

import pandas as pd
import numpy as np
import random
from scipy import sparse

from sklearn import ensemble, linear_model, preprocessing, model_selection, metrics, tree
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import adult
import communities

import cgi
from subprocess import call

original_data = adult.original
X = adult.binary
y = adult.labels
indices = range(0, len(X))

X_train, X_test, y_train, y_test, indices_train, indices_test = model_selection.train_test_split(X, y, indices, train_size=0.70)

forest = ensemble.RandomForestClassifier(n_estimators=30)
forest = forest.fit(X_train, y_train)
print("Random forest test accuracy: %f" % forest.score(X_test, y_test))

predictions = [t.predict(X) for t in forest.estimators_]
votes = pd.DataFrame.from_dict(dict(zip(range(len(forest.estimators_)), predictions)))
score = votes.transpose().sum()
plt.figure(1)
score.hist()
plt.savefig("results/score.png")
plt.clf()

# Cluster points by the votes cast by each tree in the forest
def by_votes():
    kmeans = KMeans(n_clusters = 10).fit(votes)
    clusters = kmeans.predict(votes)
    votes_dist = pd.DataFrame(kmeans.transform(votes))

    with open("results/vote-clusters.org", "w") as f:
        print("* Clusters", "\n", file = f)

        for i in range(10):
            idx = clusters == i
            cluster_scores = score[idx]
            mean_score = cluster_scores.mean()
            dists_to_centroid = votes_dist[idx][i]
            mean_dist_to_centroid = dists_to_centroid.mean()
            cluster_scores.hist()
            plt.savefig("results/vote-cluster-%d-scores.png" % i)
            plt.clf()
            print("** Cluster", i, ", mean score =", mean_score, ", dist to center =", mean_dist_to_centroid, "\n", file = f)
            s = original_data[idx].sample(n = 5)
            for p in range(len(s)):
                print(s.iloc[p], file = f)
                print("Weight =", score[s.iloc[p].name], "\n", file = f)

    return kmeans

# Cluster points by the activations of each node in each tree.  Since this is a
# very high dimensional space, we only cluster based on a sample of 1000 points.
def by_all_nodes():
    train_sample = X.sample(n = 1000)
    activations_train, n_nodes_train = forest.decision_path(train_sample)
    activations, n_nodes = forest.decision_path(X)

    kmeans = KMeans(n_clusters = 10).fit(activations_train)
    clusters = kmeans.predict(activations)
    activations_dist = kmeans.transform(activations)

    with open("results/all-node-clusters.org", "w") as f:
        print("* Clusters", "\n", file = f)

        for i in range(10):
            idx = clusters == i
            cluster_scores = score[idx]
            mean_score = cluster_scores.mean()
            dists_to_centroid = pd.DataFrame(activations_dist[idx])[i]
            mean_dist_to_centroid = dists_to_centroid.mean()
            cluster_scores.hist()
            plt.savefig("results/all-node-cluster-%d-scores.png" % i)
            plt.clf()
            print("** Cluster", i, ", mean score =", mean_score, ", dist to center =", mean_dist_to_centroid, "\n", file = f)
            s = original_data[idx].sample(n = 5)
            for p in range(len(s)):
                print(s.iloc[p], file = f)
                print("Weight =", score[s.iloc[p].name], "\n", file = f)

    return kmeans

def to_leaf_space(X):
    leaf_nodes = pd.DataFrame(forest.apply(X))
    r    = range(len(X))
    ones = [1 for i in r]
    return sparse.hstack([sparse.csr_matrix((ones, (r, leaf_nodes[j])))
                          for j in range(len(forest.estimators_))])


def total_variation(X, idx, col):
    counts_X    = X[col].value_counts(normalize = True)
    counts_Xsub = X[idx][col].value_counts(normalize = True)
    return counts_X.subtract(counts_Xsub, fill_value = 0.0).abs().sum() / 2

def by_leaf_nodes():
    to_leaf_space = preprocessing.OneHotEncoder().fit(forest.apply(X))
    activations = to_leaf_space.transform(forest.apply(X))
    X_train = X.sample(n = 1000)

    n_clusters = 50

    kmeans = KMeans(n_clusters = n_clusters).fit(to_leaf_space.transform(forest.apply(X_train)))
    clusters = kmeans.predict(activations)
    activations_dist = kmeans.transform(activations)

    with open("results/leaf-node-clusters.org", "w") as f:
        print("* Clusters", "\n", file = f)

        for i in range(n_clusters):
            idx = clusters == i
            cluster_scores = score[idx]
            mean_score = cluster_scores.mean()
            dists_to_centroid = pd.DataFrame(activations_dist[idx])[i]
            mean_dist_to_centroid = dists_to_centroid.mean()
            cluster = original_data[idx]

            # Save histogram of weights in cluster
            cluster_scores.hist()
            plt.savefig("results/leaf-node-cluster-scores-%02d.png" % i)
            plt.clf()

            # Represent cluster as small decision tree
            t = tree.DecisionTreeClassifier(max_depth = 6,
                                            max_leaf_nodes = 7, criterion = 'gini')
            t = t.fit(X, idx)
            out_file = "results/leaf-node-cluster-tree-%02d" % i
            tree.export_graphviz(t, out_file = out_file + ".dot",
                                 feature_names = [cgi.escape(s) for s in X.columns],
                                 filled = True,
                                 rounded = True,
                                 special_characters = True)
            with open(out_file + ".png", "w") as dot_file:
                call(["dot", "-Tpng", out_file + ".dot"], stdout=dot_file)
            call(["rm", out_file + ".dot"])

            # Print some cluster statistics and samples
            print("** Cluster %02d, size = %d, mean score = %.03f, dist to center = %.03f\n" % \
                  (i, len(cluster), mean_score, mean_dist_to_centroid), file = f)
            print("*** Statistics\n", file = f)
            for col in original_data.columns:
                print("%s: %.04f" % (col, total_variation(original_data, idx, col)), file = f)
            print("\n*** Samples\n", file = f)
            s = cluster.sample(n = min(5, len(cluster)))
            for p in range(len(s)):
                print(s.iloc[p], file = f)
                print("Weight = %.00f\n" % score[s.iloc[p].name], file = f)

    return activations, clusters, kmeans

activations, clusters, kmeans = by_leaf_nodes()
