from itertools import islice
import re

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

import sys, pickle, time

# from chem_ocean.oc_db_fxns import build_oc_station_db, make_arrays, oc_data_2df
from matplotlib.backends.backend_pdf import PdfPages

from sklearn import cluster, datasets, metrics
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from chem_ocean.oc_clustering_fxns import build_cluster_data, test_clustering

from numpy.random import uniform, seed
from matplotlib.mlab import griddata
from mpl_toolkits.basemap import shiftgrid
from numpy import linspace
from numpy import meshgrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



title_sz = 27
axis_sz = 22
tick_sz = 21

import shapefile
from matplotlib import cm, rcParams
from mpl_toolkits.basemap import Basemap
rcParams.update({'font.size': 11}) # Increase font-size
import time

# @cache.cached(timeout=50, key_prefix='custom_map')
def build_map(show, proj, minLat, maxLat, minLon, maxLon, res, fig, ax, labels_in):
    t0 = time.time()
    if proj in ['cyl', 'merc', 'mill', 'cea', 'gall', 'lcc']:
        _map = Basemap(projection=proj, lat_0 = (minLat+maxLat)*.5, lon_0 = (minLon+maxLon)*.5,
            resolution = res, area_thresh = 1,
            llcrnrlon=(minLon)*1, llcrnrlat=(minLat)*1,
            urcrnrlon=(maxLon)*1, urcrnrlat=(maxLat)*1)
    if proj in ['stere']:
        _map = Basemap(projection=proj, lat_0 = (minLat+maxLat)*.5, lon_0 = (minLon+maxLon)*.5,
            resolution = res, area_thresh = 1,
            llcrnrlon=(minLon)-30, llcrnrlat=(minLat)*1,
            urcrnrlon=(maxLon)*1, urcrnrlat=(maxLat)*1)
    if proj in ['ortho', 'geos', 'nsper']:
        _map = Basemap(projection=proj, lat_0 = (minLat+maxLat)*.5, lon_0 = (minLon+maxLon)*.5,
            resolution = res, area_thresh = 1,
            llcrnry=minLat*1,urcrnry=maxLat*1)#, llcrnrx=minLon*1, urcrnrx=maxLon*1, )
    else: 
        _map = Basemap(projection=proj, lat_0 = (minLat+maxLat)*.5, lon_0 = (minLon+maxLon)*.5,
            resolution = res, area_thresh = 1, llcrnrlon=(minLon)*1, llcrnrlat=(minLat)*1,
            urcrnrlon=(maxLon)*1, urcrnrlat=(maxLat)*1, rsphere=(6378137.00,6356752.3142))

    t1 = time.time()
#     print(1, t1-t0)
    if show == 'y':
        t0 = time.time()
        _map.ax = ax
        t1 = time.time()
        _map.drawcoastlines(color='k')
        _map.drawcountries()
        _map.fillcontinents(lake_color='b',color = 'gray')
        _map.drawmapboundary(linewidth=2)
        
        # labels = [left,right,top,bottom]
        lbls = []
        for label in ['l','r','t','b']:
            if label in labels_in:
                lbls.append(1)
            else:
                lbls.append(0)
            
        _map.drawmeridians(np.arange(0, 360, 30), labels=lbls)
        _map.drawparallels(np.arange(-90, 90, 30), labels=lbls)    
        t2 = time.time()
        print(t2-t1)
#     plt.savefig('/static/temp_map.png', dpi=200)
    return _map, fig, ax