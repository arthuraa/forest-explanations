import pandas as pd
import numpy as np
import random

from sklearn import ensemble, linear_model, preprocessing, cross_validation, metrics, tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import adult as adult

original_data = adult.original
X = adult.binary
y = adult.labels
indices = range(0, len(X))

X_train, X_test, y_train, y_test, indices_train, indices_test = cross_validation.train_test_split(X, y, indices, train_size=0.70)

forest = ensemble.RandomForestClassifier(n_estimators=30)
forest = forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print "Random forest F1 score: %f" % metrics.f1_score(y_test, y_pred)

predictions = [tree.predict(X) for tree in forest.estimators_]
votes = pd.DataFrame.from_items(zip(range(len(forest.estimators_)), predictions))
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
        print >> f, "* Clusters", "\n"

        for i in range(10):
            idx = clusters == i
            cluster_scores = score[idx]
            mean_score = cluster_scores.mean()
            dists_to_centroid = votes_dist[idx][i]
            mean_dist_to_centroid = dists_to_centroid.mean()
            cluster_scores.hist()
            plt.savefig("results/vote-cluster-%d-scores.png" % i)
            plt.clf()
            print >> f, "** Cluster", i, ", mean score =", mean_score, ", dist to center =", mean_dist_to_centroid, "\n"
            s = original_data[idx].sample(n = 5)
            for p in range(len(s)):
                print >> f, s.iloc[p]
                print >> f, "Weight =", score[s.iloc[p].name], "\n"

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
        print >> f, "* Clusters", "\n"

        for i in range(10):
            idx = clusters == i
            cluster_scores = score[idx]
            mean_score = cluster_scores.mean()
            dists_to_centroid = pd.DataFrame(activations_dist[idx])[i]
            mean_dist_to_centroid = dists_to_centroid.mean()
            cluster_scores.hist()
            plt.savefig("results/all-node-cluster-%d-scores.png" % i)
            plt.clf()
            print >> f, "** Cluster", i, ", mean score =", mean_score, ", dist to center =", mean_dist_to_centroid, "\n"
            s = original_data[idx].sample(n = 5)
            for p in range(len(s)):
                print >> f, s.iloc[p]
                print >> f, "Weight =", score[s.iloc[p].name], "\n"

    return kmeans

def by_leaf_nodes():
    node_paths = forest.apply(X)
    X_for_paths = X.sample(n = 20)

    with open("results/path-neighborhoods.org", "w") as f:
        print >> f, "* Clusters", "\n"

        for i in range(len(X_for_paths)):
            # FIXME: Apparently, this is too slow
            dists = pd.Series([sum(node_paths[i] != node_paths[j]) for j in range(len(X))])
            close = original_data[dists <= 26]
            if (len(close) <= 10):
                s = close
            else:
                s = close.sample(n = 10)
            print >> f, "** Cluster", i, ", mean dist = ", dists.mean(), ", sd = ", dists.std(), ", # < 27 = ", len(close), "\n"
            for p in range(len(s)):
                print >> f, s.iloc[p,:], "\n"
                print >> f, "Weight =", score[s.iloc[p].name], "\n"
