import numpy as np

from sklearn import cluster, metrics
from sklearn.metrics import euclidean_distances, silhouette_samples, silhouette_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


from matplotlib.mlab import griddata
import matplotlib.pyplot as plt

from numpy import linspace
from numpy import meshgrid
from numpy.random import uniform, seed

from mpl_toolkits.basemap import Basemap, shiftgrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


title_sz = 27
axis_sz = 22
tick_sz = 21

from matplotlib.ticker import MaxNLocator
from matplotlib import cm, rcParams
rcParams.update({'font.size': tick_sz-4}) # Increase font-size
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
from matplotlib import gridspec

import pylab as pl

from itertools import islice
import sys, pickle, time, copy, re

from chem_ocean import Build_Map as bm
from chem_ocean import Plot_Raw2 as pr

from flask import flash
#kwargs: models

def test_clustering3(_x, _y, _data, _xLab, _yLab, N_CLUSTERS, _latLon_params, _basemp, models):
    pred_dict = {}
    np.random.seed(0)

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    
    plot_num = 1
    X = _data
    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    
    # Compute distances
    # create clustering estimators
    alg_list = []
    for model in [models]:
        
        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        
        if model in ['ward', 'agglomerative_clustering']:
            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(X, n_neighbors=10)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
            if model == 'ward':
                ward = cluster.AgglomerativeClustering(n_clusters=N_CLUSTERS,
                                linkage='ward', connectivity=connectivity)
                alg_list.append(('ward', ward))
            if model == 'agglomerative_clustering':
                average_linkage = cluster.AgglomerativeClustering(linkage="average",
                                        affinity="cityblock", n_clusters=N_CLUSTERS,
                                        connectivity=connectivity)
                alg_list.append(('agglomerative_clustering', average_linkage))
        if model == 'mini_batch_kmeans':
            two_means = cluster.MiniBatchKMeans(n_clusters=N_CLUSTERS)
            alg_list.append(('mini_batch_kmeans', two_means))
        if model == 'spectral_clustering':
            spectral = cluster.SpectralClustering(n_clusters=N_CLUSTERS,
                                              eigen_solver='arpack',
                                              affinity="nearest_neighbors")
            alg_list.append(('spectral_clustering', spectral))
        if model == 'dbscan':
            dbscan = cluster.DBSCAN(eps=.2)
            alg_list.append(('dbscan', dbscan))
        if model == 'affinity_propagation':
            affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200)
            alg_list.append(('affinity_propagation', affinity_propagation))

    print(alg_list)
    models = {}
    for name, algorithm in alg_list:
        models[name] = {}
        # predict cluster memberships
        models[name]['start'] = time.time()
        algorithm.fit(X)
        models[name]['end'] = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
            models[name]['y_pred']= y_pred
        else:
            y_pred = algorithm.predict(X)
            models[name]['y_pred']= y_pred

        models[name]['sil_score'] = metrics.silhouette_score(X, y_pred, metric='euclidean')
        models[name]['sample_sil_vals'] = silhouette_samples(X, y_pred)
        
        #models[name]['model']= algorithm
        models[name]['N_CLUSTERS']= N_CLUSTERS

    return models



def subplot_labels(_ax, _xLab, _yLab, xlabelpad, ylabelpad, tick_sz, axis_sz, _title, text_coord, text_note, text_note_sz, text_note_align):
    _ax.set_ylabel(_yLab, fontsize=axis_sz-3, labelpad = ylabelpad)
    _ax.set_xlabel(_xLab, fontsize=axis_sz-3, labelpad = xlabelpad)

    # set tick size
    xtickNames = _ax.get_xticklabels()
    ytickNames = _ax.get_yticklabels()

    plt.setp(ytickNames, rotation=0, fontsize=tick_sz-3)
    plt.setp(xtickNames, rotation=0, fontsize=tick_sz-3)

    _ax.set_title(_title, size=axis_sz)

    _ax.text(text_coord[0], text_coord[1],text_note, transform=plt.gca().transAxes, size=text_note_sz, horizontalalignment=text_note_align)


'''
Generates plots:
one (1) constant-depth slice ("plans") model output with sil coef plot
one (1) constant-lat or lon slice ("section") model output with sil coef plot

constant-depth slices ("plans") of multiple models (no sil coef plot)
constant-lat or lon slices ("section") of multiple models (no sil coef plot)
'''

def plot_model_output(_x, _y, _xLab, _yLab, minLat, maxLat, minLon, maxLon, _latLon_params, model_d, name, _sliceType, sil = "no"):
    pred_dict = {}
    np.random.seed(0)

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    
    plot_num = 1
    NROWS = 1
    NCOLS = 1#len(models)
    pos_num = 100+10*NCOLS+1
    
    if sil == 'no':
        if _sliceType == 'plan':
            fig = plt.figure(figsize=(17*NROWS, 17*NCOLS), facecolor='w')
        else:
            fig = plt.figure(figsize=(14*NROWS, 7*NCOLS), facecolor='w')
            
    # for name, model_d in models.items():
    y_pred = model_d['y_pred']
    N_CLUSTERS = model_d['N_CLUSTERS']
    score = model_d['sil_score']
    t0 = model_d['start']
    t1 = model_d['end']
    sample_silhouette_values = model_d['sample_sil_vals']
    _colors = cm.spectral(y_pred.astype(float) / N_CLUSTERS)
        # print(_colors[1:10])

        
    if sil == 'yes':
        if _sliceType == "plan":
            figSize = (12, 8)
        else:
            figSize = (16, 7)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figSize, facecolor='w')
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

        ax2 = plt.subplot(gs[1])
        ax2.set_xlim([-0.1, 1])
        ax2.set_ylim([0, len(sample_silhouette_values) + (N_CLUSTERS + 1) * 10])

        # Compute the silhouette scores for each sample
        sample_silhouette_values = model_d['sample_sil_vals']

        y_lower = 10
        for i in range(N_CLUSTERS):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[y_pred == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / N_CLUSTERS)
            ax2.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax2.text(-0.1, y_lower + 0.45 * size_cluster_i, str(i), size=axis_sz)

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        # The vertical line for average silhoutte score of all the values
        ax2.axvline(x=score, color="red", linestyle="--", linewidth=3)

        ax2.set_yticks([])  # Clear the yaxis labels / ticks
        ax2.set_xticks([ 0, .3, .6, .9])

        subplot_labels(ax2, "Silhouette Coefficient", "Cluster label", 0, 0, 
                       tick_sz, axis_sz, '', [.87, .90], 'avg.=\n'+ '%.2f' % score, tick_sz, 'center')

        _ylabpad = 35
        if _sliceType == 'plan':
            lbls_in = 'lb'
            _ylabpad = 35

            pos_num = 121

            _colors = cm.spectral(y_pred.astype(float) / N_CLUSTERS)
            pred_dict[name] = [_x, _y, colors[y_pred].tolist()] 

            _basemap, fig, ax = bm.build_map('y', 'merc', minLat, maxLat, minLon, maxLon, 'c', fig, plt.subplot(gs[0]), pos_num, lbls_in)
            x,y = _basemap(_x, _y)
            
            for im in range(len(x)):
                _basemap.plot(x[im], y[im], color = _colors[im], marker = 'o', markersize=9, alpha = .3)    
            
            plt.tight_layout(pad=3, w_pad=4., h_pad=3.0)
        
        else:
            # plot section 
            ax = pr.make_section(plt.subplot(gs[0]), _colors, _x,_y, _latLon_params, 'model')

    else:
        pred_dict[name] = [_x, _y, colors[y_pred].tolist()]   
        lbls_in = 'lb'
        if _sliceType == 'plan':
            if plot_num == 1:
                _ylabpad = 35
            else:
                _yLab = ''
                _ylabpad = 20
                lbls_in = 'rb' if plot_num == NCOLS else 'b'
                    
            ax = plt.subplot(pos_num)
            _basemap, fig, ax = bm.build_map('y', 'merc', minLat, maxLat, minLon, maxLon, 'c', fig, ax, pos_num, lbls_in)

            x,y = _basemap(_x, _y)
            for im in range(len(x)):
                _basemap.plot(x[im], y[im], color = _colors[im], marker = 'o', markersize=10, alpha = .3)  
            
            plt.tight_layout(pad=3, w_pad=4., h_pad=3.0)

        else:
            ax = pr.make_section(plt.subplot(pos_num), _colors, _x,_y, _latLon_params, 'model')
                
        subplot_labels(ax, _xLab, _yLab, 25, _ylabpad, tick_sz, axis_sz, name+ ', ' + '%.2f' % score,  [.99, .01], ('%.2fs' % (t1 - t0)).lstrip('0'), 12, 'right')

        plot_num += 1
        pos_num += 1
            
    return pred_dict, fig    