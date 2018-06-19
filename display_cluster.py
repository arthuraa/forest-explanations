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

def distance(u, v, ignore_nan=True):
    if len(u) == 0 or len(v) == 0:
        return np.nan
    elif u.dtype.name == 'float64' or u.dtype.name == 'int64':
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
            return wasserstein_distance(u_nonnan, v_nonnan)
        else:
            return diff_nan + wasserstein_distance(u_nonnan, v_nonnan)
    else:
        u = u.value_counts(normalize = True, dropna = False).sort_index()
        v = v.value_counts(normalize = True, dropna = False).sort_index()
        return total_variation(u, v)

def display_cluster(clusters, X, X_test, y_test,
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
    vs = votes(clusters.model, X)
    vs_test = votes(clusters.model, X_test)

    clusters_gen, distances_gen = clusters.predict_transform(X)
    clusters_test, distances_test = clusters.predict_transform(X_test)

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
                'Accuracy': clusters.model.score(X_cluster, y_cluster) if len(X_cluster) != 0 else '--',
                'distances': distances}

    cluster_descs = [cluster_desc(c) for c in range(clusters.kmeans.n_clusters)]

    with pd.option_context('display.max_rows', None):
        display(pd.DataFrame([OrderedDict([('Size', d['Size']),
                                           ('Accuracy', d['Accuracy']),
                                           ('Mean Votes', d['votes'].mean())])
                              for d in cluster_descs]))

    cluster_widget = widgets.BoundedIntText(value = 0,
                                            min = 0,
                                            max = clusters.kmeans.n_clusters,
                                            step = 1,
                                            description = 'Cluster')

    column_widget = widgets.Dropdown(options = orig_X.columns,
                                     description = 'Column')

    def do_display(cluster, curr_col):
        desc = cluster_descs[cluster]
        orig_X_cluster = desc['orig_X']
        X_cluster = desc['X']
        y_cluster = desc['y']
        vs_cluster = desc['votes']
        if feature_description:
            print(feature_description[curr_col])
        print('Cluster accuracy: %.03f' % clusters.model.score(X_cluster, y_cluster))
        print('Cluster size: %d/%d' % (len(X_cluster), len(X_test)))
        print('Cluster mean votes: %.03f' % vs_cluster.mean())
        distances = desc['distances']
        out1 = widgets.Output()
        with out1:
            with pd.option_context('display.max_rows', None):
                display(distances)
        plt.ioff()
        fig, axes = plt.subplots(figsize = (5, 5))
        axes.set_title('Distribution of votes')
        axes.grid()
        labels = ['Cluster %d' % cluster, 'General']
        axes.hist([vs_cluster, vs], density = True, label = labels)
        axes.legend()
        out2 = widgets.Output()
        with out2:
            display(fig)
        ws = [out1, out2]
        plt.ion()
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
            axes.legend()
            plt.ion()
            out3 = widgets.Output()
            with out3:
                display(fig)
            ws.append(out3)
        display(widgets.HBox(ws, layout = {'height': '700px'}))

    w = widgets.interactive(do_display, cluster = cluster_widget, curr_col = column_widget)
    display(w)
