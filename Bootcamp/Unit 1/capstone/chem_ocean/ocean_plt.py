import chem_ocean.ocean_data as oc_data

from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
title_sz = 27
axis_sz = 18
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


class rawPlotter():
    '''
    Generates plots (single and double): section, section+section, section+column, plan, plan+column, column of tracer data.  All require a data object, the tracer name, a preexisting fig, an ax object (of that fig).  The add_plan also requires the user specifies a depth. add_section and add_plan both include kwargs: share_limits (True applies general limits for the tracer, False applies limits of that particular dataset, a list uses the specified bounds as the tracer limits), and colorbar ('y' or no mention of 'colorbar' in kwargs prompt a colorbar, 'n' causes colorbar to be omitted)
    support methods include: make_colorbar, make_contours, make_insetmap
    '''
    
    gs_d = {'section':3, 'plan':2, 'column':1, 'other': 1}
    units_dict = {'temperature': '$^\circ$C', 'oxygen': 'ml/l', 'aou': 'ml/l', 'longitude': '$^\circ$E', 'salinity': '(psu)', 'nitrate': '$\mu$mol/l', 'depth': 'm', 'phosphate': '$\mu$mol/l', 'latitude': '$^\circ$N', 'oxygen_saturation': '%'}
    limits_dict = {'temperature': [0, 28], 'oxygen': [0, 10], 'aou': [0,7], 'longitude': [-179, 179], 'salinity': [32, 36.4], 'nitrate': [0,48], 'depth': [0,6000], 'phosphate':[0,4], 'latitude': [-90,90]}
    
    def __init__(self, plotlist, tracerlist):
        self.plotlist = plotlist
        self.tracerlist = tracerlist
    
    def make_insetmap(self, ax, _colors, _x, _y, _lonLat_params, model_raw):
        corner = 4
#         if model_raw == 'model':
#             for im in range(len(_x)):
#                 ax.plot(_x[im], _y[im], c = _colors[im], marker = 'o', markersize=15, alpha = .1)
        if model_raw == 'column':
            corner = 3

        ax.invert_yaxis()

        axin = inset_axes(ax, width="35%", height="35%", loc=corner)
        inmap = Basemap(projection='ortho', lon_0=np.mean(_lonLat_params[0]), lat_0=0, ax=axin, anchor='NE')
        inmap.fillcontinents()
        inmap.drawcoastlines(color='k')
        if model_raw == 'column':  
            inmap.scatter(_lonLat_params[0], _lonLat_params[1], color ='b' , latlon=True)  
        else:
            inmap.plot(_lonLat_params[0], _lonLat_params[1], '-b', linewidth=2 , latlon=True) 
        return ax

        
    def make_contours(self, x_coord, y_coord, _feat_data, ax, share_limits):
        # define grid.
        xi = np.linspace(min(x_coord), max(x_coord),300)
        yi = np.linspace(min(y_coord), max(y_coord),300)

        # grid the data.
        zi = griddata(x_coord,y_coord,_feat_data,xi,yi,interp='linear')

        # contour the gridded data, plotting dots at the nonuniform data points.
        CS = ax.contour(xi,yi,zi,10,linewidths=0.5,colors='k')
#                         vmax=abs(zi).max(), vmin=abs(zi).min(), alpha = .7)

        if type(share_limits) == list:
            CS = ax.contourf(xi,yi,zi,15,cmap=plt.cm.rainbow, 
                            vmax=share_limits[1], vmin=share_limits[0], alpha = .7)
        else:
            if share_limits == True:
                print(self.limits_dict[self.tracerlist[0]][0],self.limits_dict[self.tracerlist[0]][1])
                CS = ax.contourf(xi,yi,zi,15,cmap=plt.cm.rainbow, 
                        vmax=self.limits_dict[self.tracerlist[0]][1], vmin=self.limits_dict[self.tracerlist[0]][0], alpha = .7)
            else:
                CS = ax.contourf(xi,yi,zi,15,cmap=plt.cm.rainbow, 
                                    vmax=abs(zi).max(), vmin=abs(zi).min(), alpha = .7)
    
        return CS, ax
    
    def make_colorbar(self, ax, CS, tracer, cbar_pad):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=cbar_pad)
        cbar = plt.colorbar(CS, cax=cax)
        cbar_label = tracer + ' (' + self.units_dict[tracer] + ')'
        cbar.ax.set_ylabel(cbar_label, fontsize=axis_sz)
        return ax, cbar
    
    
    def make(self, **kwargs):
        cols = len(self.tracerlist)
        rows = len(self.plotlist)
        print(self.plotlist)
        w_ratios = [self.gs_d[self.plotlist[ik]] for ik in range(len(self.plotlist))]
        fig = plt.figure(figsize=(10+6, 8))
        if self.plotlist == ['section', 'section'] or self.plotlist == ['plan', 'plan']:
            gs = gridspec.GridSpec(cols, rows, width_ratios=w_ratios, wspace=0.1) 
        else:
            gs = gridspec.GridSpec(cols, rows, width_ratios=w_ratios, wspace=0.7) 
        
        ax_out = []

        for ik in range(rows*cols):
            ax_out.append(fig.add_subplot(gs[ik]))

        return fig, ax_out

    def add_column(self, fig, ax, data, tracer):#_x, _y, _d, _feat_data, _xLab, _yLab, _lonLat_params):
        if self.plotlist == ['column']:
            fig.set_figheight(7)
            fig.set_figwidth(6)
            
        ax.scatter(data._feat_data, data._d,  color='black')
        ax = self.make_insetmap(ax, False, data._feat_data, data._d, data._lonLat_params, 'column')
        ax.set_xlabel(fig.get_axes()[-1].get_ylabel(), fontsize=axis_sz-3, labelpad = 0)
    
        # set tick parameters
        xtickNames = ax.get_xticklabels()
        ytickNames = ax.get_yticklabels()
    
        for names in [ytickNames, xtickNames]:
            plt.setp(names, rotation=0, fontsize=tick_sz-4)
            
        return fig, ax
    
    def add_plan(self, fig, ax, data, tracer, depth, **kwargs):#minLat, maxLat, minLon, maxLon, tracer, depth):
        maxLon, minLon = max( data._x), min( data._x)
        maxLat, minLat = max( data._y), min( data._y)
        figwidth = np.ceil((maxLon-minLon)/360 * 12)
        print(figwidth)
        if 'column' in self.plotlist:
            fig.set_figwidth(6+figwidth)
        if self.plotlist == ['plan', 'plan']:
            fig.set_figwidth(figwidth*2)
            fig.set_figheight(8)
        else:
            fig.set_figwidth(figwidth)

        _basemap, fig, ax = bm.build_map('y', 'merc', minLat, maxLat, minLon, maxLon, 'c', fig, ax, 'lb')
        x_coord,y_coord = _basemap(data._x, data._y)
        cbar_pad = .22
        #ax.tight_layout(pad=3, w_pad=4., h_pad=3.0)
        ylabpad = 35
        xlabpad = 25
        #ax.tight_layout(pad=0.4, w_pad=1., h_pad=1.0)
        
        if 'share_limits' in kwargs:
            share_limits = kwargs['share_limits']
        else:
            share_limits = False
            
        CS, ax = self.make_contours(x_coord, y_coord, data._feat_data, ax, share_limits)
        
        # label axes
        ax.set_ylabel(data._yLab, fontsize=axis_sz-3, labelpad = ylabpad)
        ax.set_xlabel(data._xLab, fontsize=axis_sz-3, labelpad = xlabpad)
        
#         ax, cbar = self.make_colorbar(ax, CS, tracer, cbar_pad)

        # set tick size
        xtickNames = ax.get_xticklabels()
        ytickNames = ax.get_yticklabels()
        ticknames = [ytickNames, xtickNames]
        
        if ('colorbar' not in kwargs) or (kwargs['colorbar'] =='y'):
            fig.subplots_adjust(right = .8)
            ax, cbar = self.make_colorbar(ax, CS, tracer, cbar_pad)
            cbartickNames = cbar.ax.get_yticklabels()
            ticknames = [ytickNames, xtickNames, cbartickNames]
        
        for names in ticknames:
            plt.setp(names, rotation=0, fontsize=tick_sz-6)
            
        return fig, ax
    
    def add_section(self, fig, ax, data, tracer, **kwargs ):
        figwidth = np.floor((max(data._x)-min(data._x))/140 * 12)
        print(figwidth)

        if 'column' in self.plotlist:
            figwidth = np.ceil((max(data._x)-min(data._x))/360 * 12)
            fig.set_figwidth(6+figwidth)
        else:
            fig.set_figwidth(figwidth)
        
        x_coord = data._x
        y_coord = data._y
        ylabpad = 0
        xlabpad = 0
        
        if 'share_limits' in kwargs:
            share_limits = kwargs['share_limits']
        else:
            share_limits = False
            
        CS, ax = self.make_contours(x_coord, y_coord, data._feat_data, ax, share_limits)
        
        ax = self.make_insetmap(ax, None, x_coord, y_coord, data._lonLat_params, 'raw')
        
        cbar_pad = .22
        if 'plot_depths' in kwargs:
            ax.scatter(data._x, data._d, color = 'w', alpha = .5)
            
        # label axes
        ax.set_ylabel(data._yLab, fontsize=axis_sz-3, labelpad = ylabpad)
        ax.set_xlabel(data._xLab, fontsize=axis_sz-3, labelpad = xlabpad)
        
        # set tick size
        xtickNames = ax.get_xticklabels()
        ytickNames = ax.get_yticklabels()
        ticknames = [ytickNames, xtickNames]
        
        if ('colorbar' not in kwargs) or (kwargs['colorbar'] =='y'):
            ax, cbar = self.make_colorbar(ax, CS, tracer, cbar_pad)
            cbartickNames = cbar.ax.get_yticklabels()
            ticknames = [ytickNames, xtickNames, cbartickNames]
        
        for names in ticknames:
            plt.setp(names, rotation=0, fontsize=tick_sz-4)
            
        return fig, ax


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