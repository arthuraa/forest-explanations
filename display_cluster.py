from __future__ import print_function

import matplotlib.pyplot as plt
from IPython.display import HTML
import ipywidgets as widgets

import pandas as pd
import numpy as np
import communities
import sklearn
from forest_clusters import Forest, ForestClusters, buckets, total_variation

def display_cluster(clusters, X, X_test, y_test):
    votes = clusters.model.votes(X)
    votes_test = clusters.model.votes(X_test)

    clusters_gen, distances_gen = clusters.predict_transform(X)
    clusters_test, distances_test = clusters.predict_transform(X_test)

    cluster_widget = widgets.BoundedIntText(value = 0,
                                            min = 0,
                                            max = clusters.kmeans.n_clusters,
                                            step = 1,
                                            description = 'Cluster')

    column_widget = widgets.Dropdown(options = X.columns,
                                     description = 'Column')

    def do_display(cluster, curr_col):
        tvs = [{'Column': col,
                'Distance': total_variation(buckets(X[col]),
                                            buckets(X_test[clusters_test == cluster][col]))}
               for col in X.columns]
        tvs = pd.DataFrame(tvs).sort_values('Distance', ascending = False)
        out1 = widgets.Output()
        with out1:
            with pd.option_context('display.max_rows', None):
                display(HTML(tvs.to_html(index = False)))
        plt.ioff()
        fig, axes = plt.subplots(figsize = (5, 5))
        votes_cluster = votes_test[clusters_test == cluster]
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
            dist_cluster = X_test[clusters_test == cluster][curr_col]
            dist         = X[curr_col]
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
