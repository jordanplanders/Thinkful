import matplotlib.pyplot as plt

import numpy as np

from mpl_toolkits.basemap import Basemap, shiftgrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

title_sz = 27
axis_sz = 22
tick_sz = 21

#from matplotlib import cm, rcParams
#rcParams.update({'font.size': tick_sz-4}) # Increase font-size
# from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm

import pylab as pl


def make_section(ax, _colors, _x, _y, _latLon_params, model_raw):
    corner = 4
    if model_raw == 'model':
        for im in range(len(_x)):
            ax.plot(_x[im], _y[im], c = _colors[im], marker = 'o', markersize=15, alpha = .1)
    elif model_raw == 'column':
        corner = 3

    ax.invert_yaxis()
    
    axin = inset_axes(ax, width="35%", height="35%", loc=corner)
    inmap = Basemap(projection='ortho', lon_0=np.mean(_latLon_params[0]), lat_0=0, ax=axin, anchor='NE')
    inmap.fillcontinents()
    inmap.drawcoastlines(color='k')
    if model_raw == 'column':  
        inmap.scatter(_latLon_params[0], _latLon_params[1], color ='b' , latlon=True)  
    else:
        inmap.plot(_latLon_params[0], _latLon_params[1], '-b', linewidth=2 , latlon=True) 
    return ax


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

