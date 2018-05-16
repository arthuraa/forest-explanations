import pandas as pd
import numpy as np
import random
# import graphviz

from sklearn import ensemble, linear_model, preprocessing, cross_validation, metrics, tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

import adult as adult

original_data = adult.original
binary_data = adult.binary

def proj(model, X):
    train_preds = [tree.predict(X) for tree in model.estimators_]
    return pd.DataFrame.from_items(zip(range(0,len(model.estimators_)),train_preds))

X = binary_data[binary_data.columns.difference(["Target"])]
y = binary_data["Target"]
indices = range(0, len(X))

X_train, X_test, y_train, y_test, indices_train, indices_test = cross_validation.train_test_split(X, y, indices, train_size=0.70)

forest = ensemble.RandomForestClassifier(n_estimators=30)
forest = forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
print "Random forest F1 score: %f" % metrics.f1_score(y_test, y_pred)

outcomes = proj(forest, X)

weights = outcomes.transpose().sum()

X_p  = 2 * outcomes - 1

# Assumes X is in the {-1, 1} space
def distances_outcome(X):
    return (- (X_p.dot(X.transpose()) - len(forest.estimators_)) / 2)

def neighbors(idx, ceiling):
    return distances_outcome(X_p.iloc[idx]) <= ceiling

plt.figure(1)

weights.hist()

plt.savefig("weights.png")

plt.clf()

distances_outcome(X_p.sample(n = 1000)).mean().hist()

plt.savefig("distances_outcome.png")

simple_kmeans = KMeans(n_clusters = 10).fit(outcomes)
simple_clusters = simple_kmeans.predict(outcomes)
simple_outcomes_dist = pd.DataFrame(simple_kmeans.transform(outcomes))

with open("simple-clusters.org", "w") as f:
    print >> f, "* Clusters", "\n"

    for i in range(10):
        idx = simple_clusters == i
        cluster_weights = weights[idx]
        mean_weight = cluster_weights.mean()
        dists_to_centroid = simple_outcomes_dist[idx][i]
        mean_dist_to_centroid = dists_to_centroid.mean()
        plt.clf()
        cluster_weights.hist()
        plt.savefig("simple-cluster-%d-weights.png" % i)
        print >> f, "** Cluster", i, ", weight =", mean_weight, ", dist to center =", mean_dist_to_centroid, "\n"
        s = original_data[idx].sample(n = 5)
        for p in range(len(s)):
            print >> f, s.iloc[p]
            print >> f, "Weight =", weights[s.iloc[p].name], "\n"

X_for_nodes = X.sample(n = 1000)
node_outcomes_train, n_nodes = forest.decision_path(X_for_nodes)
node_outcomes, nodes_n_nodes_ptr = forest.decision_path(X)

node_kmeans = KMeans(n_clusters = 10).fit(node_outcomes_train)
node_clusters = node_kmeans.predict(node_outcomes)
node_outcomes_dist = node_kmeans.transform(node_outcomes)

with open("node-clusters.org", "w") as f:
    print >> f, "* Clusters", "\n"

    for i in range(10):
        idx = node_clusters == i
        cluster_weights = weights[idx]
        mean_weight = cluster_weights.mean()
        dists_to_centroid = pd.DataFrame(node_outcomes_dist[idx])[i]
        mean_dist_to_centroid = dists_to_centroid.mean()
        plt.clf()
        cluster_weights.hist()
        plt.savefig("node-cluster-%d-weights.png" % i)
        print >> f, "** Cluster", i, ", weight =", mean_weight, ", dist to center =", mean_dist_to_centroid, "\n"
        s = original_data[idx].sample(n = 5)
        for p in range(len(s)):
            print >> f, s.iloc[p]
            print >> f, "Weight =", weights[s.iloc[p].name], "\n"

node_paths = forest.apply(X)
X_for_paths = X.sample(n = 20)

with open("path-neighborhoods.org", "w") as f:
    print >> f, "* Clusters", "\n"

    for i in range(len(X_for_paths)):
        dists = pd.Series([sum(node_paths[i] != node_paths[j]) for j in range(len(X))])
        close = original_data[dists <= 26]
        print >> f, "** Cluster", i, ", mean dist = ", dists.mean(), ", sd = ", dists.std(), ", # < 27 = ", len(close), "\n"
        for p in range(len(close)):
            print >> f, close.iloc[p,:], "\n"
            print >> f, "Weight =", weights[close.iloc[p].name], "\n"






# reduced_data = PCA(n_components=2).fit_transform(X_p)
# kmeans = KMeans(init='k-means++', n_clusters=30, n_init=10)
# kmeans.fit(reduced_data)

# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')

# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.savefig("kmeans.png")
