import pandas as pd
import numpy as np
import random
from scipy import sparse

from sklearn import ensemble, linear_model, preprocessing, model_selection, metrics, tree
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import adult as adult

import graphviz

import cgi
from subprocess import call

original_data = adult.original
X = adult.binary
y = adult.labels
indices = range(0, len(X))

X_train, X_test, y_train, y_test, indices_train, indices_test = model_selection.train_test_split(X, y, indices, train_size=0.70)

forest = ensemble.RandomForestClassifier(n_estimators=30)
forest = forest.fit(X_train, y_train)
print "Random forest test accuracy: %f" % forest.score(X_test, y_test)

predictions = [t.predict(X) for t in forest.estimators_]
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

def to_leaf_space(X):
    leaf_nodes = pd.DataFrame(forest.apply(X))
    r    = range(len(X))
    ones = [1 for i in r]
    return sparse.hstack([sparse.csr_matrix((ones, (r, leaf_nodes[j])))
                          for j in range(len(forest.estimators_))])


def by_leaf_nodes():
    activations = to_leaf_space(X).tocsr()
    train_indices = pd.DataFrame(range(len(X))).sample(n = 1000)[0].tolist()
    activations_train = activations[train_indices,:]

    n_clusters = 50

    kmeans = KMeans(n_clusters = n_clusters).fit(activations_train)
    clusters = kmeans.predict(activations)
    activations_dist = kmeans.transform(activations)

    with open("results/leaf-node-clusters.org", "w") as f:
        print >> f, "* Clusters", "\n"

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
            t = tree.DecisionTreeClassifier(max_depth = 4)
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
            print >> f, "** Cluster %02d, size = %d, mean score = %.03f, dist to center = %.03f\n" % \
                (i, len(cluster), mean_score, mean_dist_to_centroid)
            s = cluster.sample(n = min(5, len(cluster)))
            for p in range(len(s)):
                print >> f, s.iloc[p]
                print >> f, "Weight = %.00f\n" % score[s.iloc[p].name]

    return activations, clusters, kmeans

activations, clusters, kmeans = by_leaf_nodes()
