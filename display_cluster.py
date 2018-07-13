from __future__ import print_function

import matplotlib.pyplot as plt
from IPython.display import HTML
import ipywidgets as widgets
from scipy.stats import wasserstein_distance

from collections import OrderedDict
import pandas as pd
import numpy as np
import communities
import sklearn
from forest_clusters import ForestClusters, buckets, total_variation, votes

def normalized_wasserstein(u, v):
    return wasserstein_distance(u, v) / u.std()

def distance(u, v, ignore_nan=True):
    if len(u) == 0 or len(v) == 0:
        return np.nan
    elif np.issubdtype(u.dtype, np.number):
        u_nanidx = np.isnan(u)
        v_nanidx = np.isnan(v)
        u_probnan = u_nanidx.mean()
        v_probnan = v_nanidx.mean()
        diff_nan = abs(u_probnan - v_probnan)
        u_nonnan = u[~u_nanidx]
        v_nonnan = v[~v_nanidx]
        if len(u_nonnan) == 0 or len(v_nonnan) == 0:
            if ignore_nan:
                return np.nan
            else:
                return diff_nan
        elif ignore_nan:
            return normalized_wasserstein(u_nonnan, v_nonnan)
        else:
            return diff_nan + normalized_wasserstein(u_nonnan, v_nonnan)
    else:
        u = u.value_counts(normalize = True, dropna = False).sort_index()
        v = v.value_counts(normalize = True, dropna = False).sort_index()
        return total_variation(u, v)

class CompressedTree:
    def __init__(self, tree):
        classes = -np.ones(tree.node_count, dtype=np.int64)
        compressed = np.zeros(tree.node_count, dtype=np.int64)

        def compress(ancestor, node):
            left = tree.children_left[node]
            right = tree.children_right[node]
            if (left == right):
                # We are at a leaf
                compressed[node] = ancestor
            else:
                compress(ancestor, left)
                compress(ancestor, right)

        def mark(node):
            left = tree.children_left[node]
            right = tree.children_right[node]
            if left != right:
                # We are in an internal node
                mark(left)
                mark(right)
                if classes[left] == classes[right]:
                    classes[node] = classes[left]
                else:
                    classes[node] = -1
                    if classes[left] != -1:
                        compress(left, left)
                    if classes[right] != -1:
                        compress(right, right)
            else:
                # We are at a leaf
                classes[node] = np.argmax(tree.value[node, 0])

        mark(0)
        self.classes = classes
        self.compressed = compressed

class CompressedForest:
    def __init__(self, forest):
        self.forest = forest
        self.compressed = [CompressedTree(tree.tree_)
                           for tree in forest.estimators_]

    def apply(self, X):
        leaves = self.forest.apply(X)
        return np.array([[self.compressed[t].compressed[leaves[i, t]]
                          for t in range(self.forest.n_estimators)]
                         for i in range(len(X))])

    def score(self, X, y):
        return self.forest.score(X, y)

    def predict(self, X):
        return self.forest.predict(X)

class ForestPath:
    def __init__(self, model, tree_idx, feature_names, node):
        tree = model.estimators_[tree_idx].tree_
        path = []
        first_node = node
        direction = None
        while node != 0:
            feature = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            path = [(node, feature, threshold, direction)] + path
            right_idx = np.where(tree.children_right == node)[0]
            left_idx  = np.where(tree.children_left  == node)[0]
            if len(right_idx) != 0:
                node = right_idx[0]
                direction = 'right'
            else: # The node must be a left child
                node = left_idx[0]
                direction = 'left'
        feature = feature_names[tree.feature[0]]
        threshold = tree.threshold[0]
        path = [(0, feature, threshold, direction)] + path
        self.model = model
        self.first_node = first_node
        self.tree = tree_idx
        self.path = path

    def __repr__(self):
        res = "Tree %d, path %d\n" % (self.tree, self.first_node)
        for node, feature, threshold, direction in self.path[:-1]:
            if direction == 'left':
                res = res + ("%s < %0.3f\n" % (feature, threshold))
            else:
                res = res + ("%s >= %0.3f\n" % (feature, threshold))
        return res

class InfluentialPaths:
    """Compute the most influential paths for each cluster"""
    # FIXME: Extract feature names from X.columns?
    def __init__(self, model, leaves, feature_names, X, clusters, n_clusters):
        path_scores = []
        for c in range(n_clusters):
            c_scores = pd.Series([])
            leaves_cluster = pd.DataFrame(leaves[clusters == c, :])
            for tree in range(model.n_estimators):
                leaves_cluster_tree = leaves_cluster[tree].apply(lambda path: (tree, path))
                c_scores = c_scores.append(leaves_cluster_tree.value_counts(normalize = True))
            path_scores = path_scores + [c_scores.sort_values(ascending = False)]
        self.model = model
        self.feature_names = feature_names
        self.points = X
        self.clusters = clusters
        self.n_clusters = n_clusters
        self.scores = path_scores

    def __getitem__(self, pair):
        """Get the rank-th most influential path associated with cluster"""
        cluster, rank = pair
        tree, node = self.scores[cluster].index[rank]
        proportion = self.scores[cluster].iloc[rank]
        return ForestPath(self.model, tree, self.feature_names, node), proportion

def display_cluster(model, X, X_test, y_test, leaves_test, clusters_test, n_clusters,
                    encode_features=None, present_features=None,
                    feature_description=None):
    orig_X = X
    orig_X_test = X_test
    if present_features:
        orig_X = present_features(orig_X)
        orig_X_test = present_features(orig_X_test)
    if encode_features:
        X = encode_features(X)
        X_test = encode_features(X_test)
    vs = votes(model, X)
    vs_test = votes(model, X_test)

    influential_paths = InfluentialPaths(model, leaves_test, X_test.columns, X_test, clusters_test, n_clusters)

    def cluster_desc(c):
        orig_X_cluster = orig_X_test[clusters_test == c]
        X_cluster = X_test[clusters_test == c]
        y_cluster = y_test[clusters_test == c]
        vs_cluster = vs_test[clusters_test == c]
        distances = [{'Column': col,
                      'Distance': distance(orig_X[col], orig_X_cluster[col])}
                     for col in orig_X.columns]
        distances = pd.DataFrame(distances).sort_values('Distance', ascending = False)
        distances = distances.set_index('Column')
        return {'orig_X': orig_X_cluster,
                'X': X_cluster,
                'y': y_cluster,
                'votes': vs_cluster,
                'Size': len(X_cluster),
                'Accuracy': model.score(X_cluster, y_cluster) if len(X_cluster) != 0 else '--',
                'distances': distances}

    cluster_descs = [cluster_desc(c) for c in range(n_clusters)]

    with pd.option_context('display.max_rows', None):
        display(pd.DataFrame([OrderedDict([('Size', d['Size']),
                                           ('Accuracy', d['Accuracy']),
                                           ('Mean Votes', d['votes'].mean() / model.n_estimators)])
                              for d in cluster_descs]))

    cluster_widget = widgets.BoundedIntText(value = 0,
                                            min = 0,
                                            max = n_clusters,
                                            step = 1,
                                            description = 'Cluster')

    column_widget = widgets.Dropdown(options = orig_X.columns,
                                     description = 'Column')

    rank_widget = widgets.IntText(value = 0, description = 'Path rank')

    def do_display(cluster, curr_col, rank):
        desc = cluster_descs[cluster]
        orig_X_cluster = desc['orig_X']
        X_cluster = desc['X']
        y_cluster = desc['y']
        vs_cluster = desc['votes']
        if feature_description:
            print(feature_description[curr_col])
        print('Cluster accuracy: %.03f' % model.score(X_cluster, y_cluster))
        print('Cluster size: %d/%d' % (len(X_cluster), len(X_test)))
        print('Cluster mean votes: %.03f' % (vs_cluster.mean() / model.n_estimators))
        distances = desc['distances']
        out1 = widgets.Output()
        with out1:
            with pd.option_context('display.max_rows', None):
                display(distances)
        fig, axes = plt.subplots(figsize = (5, 5))
        axes.set_title('Distribution of votes')
        axes.grid()
        labels = ['Cluster %d' % cluster, 'General']
        axes.hist([vs_cluster, vs], density = True, label = labels)
        axes.legend()
        out2 = widgets.Output()
        with out2:
            plt.show(fig)
        ws = [out1, out2]
        if curr_col:
            plt.ioff()
            fig, axes = plt.subplots(figsize = (5, 5))
            distr_cluster = orig_X_cluster[curr_col]
            distr         = orig_X[curr_col]
            distance      = distances.loc[curr_col]
            axes.set_title('Distribution of column %s\nDistance %f' %
                           (curr_col, distance))
            axes.grid()
            axes.hist([distr_cluster, distr], density = True, bins = 20, label = labels)
            if np.issubdtype(distr.dtype, np.number):
                plt.xticks(rotation = 'horizontal')
            else:
                plt.xticks(rotation = 'vertical')
            axes.legend()
            plt.ion()
            out3 = widgets.Output()
            with out3:
                plt.show(fig)
            ws.append(out3)
        display(widgets.HBox(ws, layout = {'height': '700px'}))
        display(influential_paths.scores[cluster].head(10))
        display(influential_paths[cluster, rank])

    w = widgets.interactive(do_display, cluster = cluster_widget, curr_col = column_widget, rank = rank_widget)
    display(w)
