from __future__ import print_function

import matplotlib.pyplot as plt
from IPython.display import HTML
import ipywidgets as widgets

from collections import OrderedDict
import pandas as pd
import numpy as np
import communities
import sklearn
from forest_clusters import Forest, ForestClusters, buckets, total_variation

def display_cluster(clusters, X, X_test, y_test, extract):
    orig_X = X
    orig_X_test = X_test
    X = extract(X)
    X_test = extract(X_test)
    votes = clusters.model.votes(X)
    votes_test = clusters.model.votes(X_test)

    clusters_gen, distances_gen = clusters.predict_transform(X)
    clusters_test, distances_test = clusters.predict_transform(X_test)

    def cluster_desc(c):
        orig_X_cluster = orig_X_test[clusters_test == c]
        X_cluster = X_test[clusters_test == c]
        y_cluster = y_test[clusters_test == c]
        votes_cluster = votes_test[clusters_test == c]
        return {'orig_X': orig_X_cluster,
                'X': X_cluster,
                'y': y_cluster,
                'votes': votes_cluster,
                'Size': len(X_cluster),
                'Accuracy': clusters.model.score(X_cluster, y_cluster)}

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
        votes_cluster = desc['votes']
        print(communities.description[curr_col])
        print('Cluster accuracy: %.03f' % clusters.model.score(X_cluster, y_cluster))
        print('Cluster size: %d/%d' % (len(X_cluster), len(X_test)))
        print('Cluster mean votes: %.03f' % votes_cluster.mean())
        tvs = [{'Column': col,
                'Distance': total_variation(buckets(orig_X[col]),
                                            buckets(orig_X_cluster[col]))}
               for col in orig_X.columns]
        tvs = pd.DataFrame(tvs).sort_values('Distance', ascending = False)
        out1 = widgets.Output()
        with out1:
            with pd.option_context('display.max_rows', None):
                display(HTML(tvs.to_html(index = False)))
        plt.ioff()
        fig, axes = plt.subplots(figsize = (5, 5))
        axes.set_title('Distribution of votes')
        axes.grid()
        labels = ['Cluster %d' % cluster, 'General']
        axes.hist([votes_cluster, votes], density = True, label = labels)
        axes.legend()
        out2 = widgets.Output()
        with out2:
            display(fig)
        ws = [out1, out2]
        plt.ion()
        if curr_col:
            plt.ioff()
            fig, axes = plt.subplots(figsize = (5, 5))
            dist_cluster = orig_X_cluster[curr_col]
            dist         = orig_X[curr_col]
            tv = total_variation(buckets(dist_cluster), buckets(dist))
            axes.set_title('Distribution of column %s\nTotal variation %f' % (curr_col, tv))
            axes.grid()
            axes.hist([dist_cluster, dist], density = True, bins = 20, label = labels)
            axes.legend()
            plt.ion()
            out3 = widgets.Output()
            with out3:
                display(fig)
            ws.append(out3)
        display(widgets.HBox(ws, layout = {'height': '700px'}))

    w = widgets.interactive(do_display, cluster = cluster_widget, curr_col = column_widget)
    display(w)
