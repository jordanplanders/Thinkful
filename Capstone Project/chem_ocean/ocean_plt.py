import chem_ocean.ocean_data as oc_data

from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
title_sz = 27
axis_sz = 22
tick_sz = 21
rcParams.update({'font.size': tick_sz-4}) # Increase font-size
# from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
from matplotlib import gridspec

from mpl_toolkits.basemap import Basemap, shiftgrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
from numpy import linspace
from numpy import meshgrid

import pylab as pl
from collections import defaultdict


import chem_ocean.Build_Map as bm
from chem_ocean.ocean_plt_util import make_section, subplot_labels




def plotRaw(minLat, maxLat, minLon, maxLon, _in_var_names, _sliceType, **kwargs):
    units_dict = {'temperature': '$^\circ$C', 'oxygen': 'ml/l', 'aou': 'ml/l', 'longitude': '$^\circ$E', 'salinity': '(psu)', 'nitrate': '$\mu$mol/l', 'depth': 'm', 'phosphate': '$\mu$mol/l', 'latitude': '$^\circ$N', 'oxygen_saturation': '%'}

    if _sliceType == 'column':
        _x, _y, _d, _feat_data, _basemap, _xLab, _yLab, _latLon_params = oc_data.get_column([minLat, maxLat], [minLon, maxLon], _in_var_names)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 7), facecolor='w')
        
        ax.scatter(_feat_data, _d,  color='black')
        make_section(ax, False, _feat_data, _d, (minLon, minLat), 'column')
        ax.set_xlabel(fig.get_axes()[-1].get_ylabel(), fontsize=axis_sz-3, labelpad = 0)
    
        # set tick parameters
        xtickNames = ax.get_xticklabels()
        ytickNames = ax.get_yticklabels()
    
        for names in [ytickNames, xtickNames]:
            plt.setp(names, rotation=0, fontsize=tick_sz-4)

        #ax.invert_yaxis()
        ax_out = (ax)
    else:
        if _sliceType == 'plan':
            _x, _y, _d, _feat_data, _basemap, _xLab, _yLab, _latLon_params = oc_data.get_plan([minLat, maxLat], [minLon, maxLon], _in_var_names, kwargs['depth'])
                        
            if 'add_profile' in kwargs:   
                fig = plt.figure(figsize=(7+6, 7))
                gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.4) 
                ax = plt.subplot(gs[0])
                ax1 = fig.add_subplot(gs[1])
                plt.sca(ax)
                ax_out = (ax, ax1)
                #ax = fig.add_subplot(221)
            else:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7), facecolor='w')
                ax_out = (ax)
                
            _basemap, fig, ax = bm.build_map('y', 'merc', minLat, maxLat, minLon, maxLon, 'c', fig, ax, 111, 'lb')
            x_coord,y_coord = _basemap(_x, _y)
            cbar_pad = .22
            #ax.tight_layout(pad=3, w_pad=4., h_pad=3.0)
            ylabpad = 35
            xlabpad = 25
            #ax.tight_layout(pad=0.4, w_pad=1., h_pad=1.0)

        else:
            if _sliceType == 'NS_section': 
                _x, _y, _d, _feat_data, _basemap, _xLab, _yLab, _latLon_params = oc_data.get_section('NS_section', minLon, [minLat, maxLat], _in_var_names)
            elif _sliceType == 'EW_section': 
                _x, _y, _d, _feat_data, _basemap, _xLab, _yLab, _latLon_params = oc_data.get_section('EW_section', minLat, [minLon, maxLon], _in_var_names)

            x_dim = np.floor((max(_x)-min(_x))/140 * 12) +6

            y_dim = np.floor((max(_y)-min(_y))/5500 * 7)
            
            fig = plt.figure(figsize=(x_dim, y_dim))
            if 'add_profile' in kwargs:   
                gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.4) 
                ax = plt.subplot(gs[0])
                ax1 = fig.add_subplot(gs[1], sharey=ax)
                plt.sca(ax)
                ax_out = (ax, ax1)
                #ax = fig.add_subplot(221)
            else:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(x_dim, y_dim), facecolor='w')
                ax_out = (ax)
            x_coord = _x
            y_coord = _y
            ylabpad = 0
            xlabpad = 0

        # define grid.
        xi = np.linspace(min(x_coord), max(x_coord),300)
        yi = np.linspace(min(y_coord), max(y_coord),300)

        # grid the data.
        zi = griddata(x_coord,y_coord,_feat_data,xi,yi,interp='linear')

        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = plt.contour(xi,yi,zi,10,linewidths=0.5,colors='k')
        CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.rainbow, 
                        vmax=abs(zi).max(), vmin=abs(zi).min(), alpha = .7)
        
        if 'plot_pt' in kwargs:
            #lon_pt, lat_pt = _basemap(kwargs['plot_pt'][1], kwargs['plot_pt'][0])
            _basemap.scatter(kwargs['plot_pt'][1], kwargs['plot_pt'][0], latlon = True, color = 'k')
            _basemap.etopo(scale= .5, alpha = .4)
                    
        # label axes
        ax.set_ylabel(_yLab, fontsize=axis_sz-3, labelpad = ylabpad)
        ax.set_xlabel(_xLab, fontsize=axis_sz-3, labelpad = xlabpad)

        # reverse axis of y is depth
        if not _basemap:
            ax = make_section(plt.gca(), None, _x,_y, _latLon_params, 'raw')  
            cbar_pad = .22

        # create colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=cbar_pad)
        cbar = plt.colorbar(CS, cax=cax)
        cbar_label = _in_var_names[0] + ' (' + units_dict[_in_var_names[0]] + ')'
        cbar.ax.set_ylabel(cbar_label, fontsize=axis_sz) 
        

        # set tick size
        xtickNames = ax.get_xticklabels()
        ytickNames = ax.get_yticklabels()
        cbartickNames = cbar.ax.get_yticklabels()

        for names in [ytickNames, xtickNames, cbartickNames]:
            plt.setp(names, rotation=0, fontsize=tick_sz-4)

    # plt.show()
    return fig, ax_out



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
            fig = plt.figure(figsize=(7*NROWS, 7*NCOLS), facecolor='w')
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
            ax = make_section(plt.subplot(gs[0]), _colors, _x,_y, _latLon_params, 'model')

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
            ax = make_section(plt.subplot(pos_num), _colors, _x,_y, _latLon_params, 'model')
        
        _ylabpad = 35        
        subplot_labels(ax, _xLab, _yLab, 25, _ylabpad, tick_sz, axis_sz, name+ ', ' + '%.2f' % score,  [.99, .01], ('%.2fs' % (t1 - t0)).lstrip('0'), 12, 'right')

        plot_num += 1
        pos_num += 1
            
    return pred_dict, fig 