import numpy as np
import sqlalchemy
from collections import defaultdict

class dataFetcher():
    
    def __init__(self):
        self._x = None
        self._y = None
        self._d = None
        self._feat_data = None
        self.cluster_d = None
        self._xLab = None
        self._yLab = None
        self._lonLat_params = None
        
    
    def connect(self, user, password, db, host='localhost', port=5432):
        '''Returns a connection and a metadata object'''
        url  = 'postgresql://{}:{}@{}:{}/{}'
        url = url.format(user, password, host, port, db)

        # The return value of create_engine() is our connection object
        con = sqlalchemy.create_engine(url, client_encoding='utf8')

        # We then bind the connection to MetaData()
        meta = sqlalchemy.MetaData(bind=con, reflect=True)

        return con, meta, url
    
    def return_from_psql(self, query, cols, _in_var_names, _x_var, _y_var, **kwargs):
        conn, meta, url = self.connect('jlanders', '', 'odv_app_dev', host='localhost', port=5432)
        result = conn.execute(query)

        cluster_d = defaultdict(list)
        for row in result:
            for ik, col in enumerate(cols):
                cluster_d[col].append(row[ik])
        _x = np.asarray(cluster_d[_x_var])
        _y = np.asarray(cluster_d[_y_var])
        _d = np.asarray(cluster_d['depth'])

        _feat_data = np.zeros((len(_x), len(_in_var_names))) 
        for ik, name in enumerate(_in_var_names):
            _feat_data[:, ik] = np.asarray(cluster_d[name])

        mask = np.all(np.isnan(_feat_data), axis=1)
        self._x = _x[~mask]
        self._y = _y[~mask]
        self._d = _d[~mask]
        _feat_data = _feat_data[~mask]
        self._feat_data = np.array([_feat_data[ik][0] for ik in range(len(_feat_data))])
        self.cluster_d = cluster_d
        self._xLab = _x_var
        self._yLab = _y_var

    
    def get_plan(self, lat_bounds, lon_bounds, _var_names, depth):
        sum_names = ['station', 'longitude', 'latitude', 'depth'] + _var_names
        cols = ', '.join(sum_names)
        query = "select "+ cols+ " from woa13 where latitude> {} AND latitude< {} AND longitude>{} and longitude<{} and depth={};".format(str(lat_bounds[0]), str(lat_bounds[1]), str(lon_bounds[0]), str(lon_bounds[1]), str(depth))

        _x_var = 'longitude'
        _y_var = 'latitude'

        self.return_from_psql(query, sum_names, _var_names, _x_var, _y_var)
        self._lonLat_params = None
        self._xLab = _x_var.title()+' (deg)'
        self._yLab = _y_var.title()+' (deg)'

    
    def get_section(self, traj_type, line, limits, _var_names):
        sum_names = ['station', 'longitude', 'latitude', 'depth'] + _var_names
        cols = ', '.join(sum_names)
        traj_bounds = (line-2.5, line+2.5)

        if traj_type == 'EW_section':
            query = "select "+ cols+ " from woa13 where latitude> {} AND latitude< {} AND longitude>{} and longitude<{};".format(str(traj_bounds[0]), str(traj_bounds[1]), str(limits[0]), str(limits[1]))

            _x_var = 'longitude'
            _y_var = 'depth'

            self.return_from_psql(query, sum_names, _var_names, _x_var, _y_var)
            self._yLab = 'Depth (m))'  
            self._xLab = 'Longitude (deg) along '+ r'%s $^\circ$' % abs(line)+ ['S' if lat<0 else 'N' for lat in [line]][0]
            self._lonLat_params = (limits, (line, line))

        if traj_type == 'NS_section':
            query = "select "+ cols+ " from woa13 where latitude> {} AND latitude< {} AND longitude>{} and longitude<{};".format(str(limits[0]), str(limits[1]), str(traj_bounds[0]), str(traj_bounds[1]))

            _x_var = 'latitude'
            _y_var = 'depth'

            self.return_from_psql(query, sum_names, _var_names, _x_var, _y_var)
            self._yLab = 'Depth (m)'  
            self._xLab = 'Latitude (deg) along '+ r'%s $^\circ$' % abs(line)+ ['W' if lon<0 else 'E' for lon in [line]][0]
            self._lonLat_params = ((line, line), limits)

    
    def get_column(self, lat_bounds, lon_bounds, _var_names):
        sum_names = ['station', 'longitude', 'latitude', 'depth'] + _var_names
        cols = ', '.join(sum_names)
        query = "select "+ cols+ " from woa13 where latitude> {} AND latitude< {} AND longitude>{} and longitude<{} order by depth;".format(str(lat_bounds[0]), str(lat_bounds[1]), str(lon_bounds[0]), str(lon_bounds[1]))
        
        _x_var = 'longitude'
        _y_var = 'latitude'

        self.return_from_psql(query, sum_names, _var_names, _x_var, _y_var)
        self._lonLat_params = [lon_bounds, lat_bounds]#None
        self._xLab = _x_var.title()+' (deg)'
        self._yLab = _y_var.title()+' (deg)'


class water_column():
    
    def __init__(self, dataset, traj_type, **kwargs):
        self.traj_type = traj_type
        self.char_values = []
        self.char_points = []
        self._ax_smooth = []
        self._d = dataset._d
        self._x = dataset._x
        self._y = dataset._y
        self._feat_data = dataset._feat_data
        self.ordered_by = 'd'
        ax = self._d
        
        if 'depth' in kwargs: #creates a horizontal water column at some depth 
            self._feat_data = dataset._feat_data[dataset._d == kwargs['depth']]
            self._x = dataset._x[dataset._d == kwargs['depth']]
            self._y = dataset._y[dataset._d == kwargs['depth']]
            self._d = dataset._d[dataset._d == kwargs['depth']]
            
            self._y = self._y[np.argsort(self._x[:])]
            self._d = self._d[np.argsort(self._x[:])]
            self._feat_data = self._feat_data[np.argsort(self._x[:])]
            self._x = self._x[np.argsort(self._x[:])]
            self.ordered_by = 'x'
            ax = self._x
    
        _ax = []
        _feat_data = []
        for ik in range(len(ax)):
            if ax[ik] not in _ax:
                _ax.append(ax[ik])
                #change this to pull the median value if there are multiple values for a given depth
                _feat_data.append(np.mean(self._feat_data[ax == ax[ik]]))#ik])
                
        self._ax_avgd = np.array(_ax)
        self._feat_data_avgd = np.array(_feat_data)
        self._feat_data_smooth = []
        self._feat_data_smooth_d1 = []
        self.extrema_d1 = []
        self.extrema_d2 = []
        self.mixing_labels = []
        self.mixing_ratios = []
        
    def smooth_data(self):

        spl = UnivariateSpline(self._ax_avgd, self._feat_data_avgd)
        if self.traj_type == 'traj':
            spl.set_smoothing_factor(0.01)
        else:
            spl.set_smoothing_factor(0.001)
            
        numpts = len(self._ax_avgd)/2
        self._ax_smooth = np.linspace(min(self._ax_avgd), max(self._ax_avgd), num=numpts, endpoint=True)
        self._feat_data_smooth = spl(self._ax_smooth)

        self._feat_data_smooth_d1 = np.gradient(self._feat_data_smooth,self._ax_smooth)
                
    def get_mixing_labels(self, mode):
        ax = self._ax_avgd
        
        ik = 0
        colors = [('r', 'b'), ('b', 'y'), ('y', 'm'), ('m', 'g'), ('g', 'c'),('c','k')]
        color = []
        depth = ax[0]
        
        # two endmember mixing model
        if mode == 'two_endmember':
            mixing_ratios = [np.zeros(2) for ik in range(len(ax))]
            u_char = min( self._feat_data_avgd) 
            l_char = max(self._feat_data_avgd)
            for ik in range(len(ax)):
                mixing_ratios[ik][0] = max(min((self._feat_data_avgd[ik]-l_char)/(u_char-l_char), 1), 0)
                mixing_ratios[ik][1] = 1-mixing_ratios[ik][0]
                if mixing_ratios[ik][0] >= .5:
                    color.append(colors[0][0])
                else:
                    color.append(colors[0][1])
                ik+=1
                
        self.mixing_labels = color
        self.mixing_ratios = mixing_ratios
        
        
'''
def connect(user, password, db, host='localhost', port=5432):
    url  = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, db)

    # The return value of create_engine() is our connection object
    con = sqlalchemy.create_engine(url, client_encoding='utf8')

    # We then bind the connection to MetaData()
    meta = sqlalchemy.MetaData(bind=con, reflect=True)

    return con, meta, url


#This is a special version to accomodate a z_var
def return_from_psql(query, cols, _in_var_names, _x_var, _y_var, **kwargs):
    conn, meta, url = connect('jlanders', '', 'odv_app_dev', host='localhost', port=5432)
    result = conn.execute(query)
    #print(query)
    
    cluster_d = defaultdict(list)
    for row in result:
        for ik, col in enumerate(cols):
            cluster_d[col].append(row[ik])
    _x = np.asarray(cluster_d[_x_var])
    _y = np.asarray(cluster_d[_y_var])
    _d = np.asarray(cluster_d['depth'])
    
    _feat_data = np.zeros((len(cluster_d['station']), len(_in_var_names))) 
    for ik, name in enumerate(_in_var_names):
        _feat_data[:, ik] = np.asarray(cluster_d[name])

    mask = np.all(np.isnan(_feat_data), axis=1)
    _x = _x[~mask]
    _y = _y[~mask]
    _d = _d[~mask]
    _feat_data = _feat_data[~mask]
    _feat_data = [_feat_data[ik][0] for ik in range(len(_feat_data))]
    
    return cluster_d, _feat_data, _x, _y, _d


def get_plan(lat_bounds, lon_bounds, _var_names, depth):
    sum_names = ['station', 'longitude', 'latitude', 'depth'] + _var_names
    cols = ', '.join(sum_names)
    query = "select "+ cols+ " from woa13 where latitude> {} AND latitude< {} AND longitude>{} and longitude<{} and depth={};".format(str(lat_bounds[0]), str(lat_bounds[1]), str(lon_bounds[0]), str(lon_bounds[1]), str(depth))
    
    _x_var = 'longitude'
    _y_var = 'latitude'

    cluster_d, _feat_data, _x, _y, _d = return_from_psql(query, sum_names, _var_names, _x_var, _y_var)
    _basemap = True
    _lonLat_params = None
    
    return _x, _y, _d, _feat_data, _basemap, _x_var.title()+' (deg)', _y_var.title()+' (deg)', _lonLat_params
    
def get_section(traj_type, line, limits, _var_names):
    sum_names = ['station', 'longitude', 'latitude', 'depth'] + _var_names
    cols = ', '.join(sum_names)
    traj_bounds = (line-1.5, line+1.5)
    
    if traj_type == 'EW_section':
        query = "select "+ cols+ " from woa13 where latitude> {} AND latitude< {} AND longitude>{} and longitude<{};".format(str(traj_bounds[0]), str(traj_bounds[1]), str(limits[0]), str(limits[1]))
        
        _x_var = 'longitude'
        _y_var = 'depth'
        
        cluster_d, _feat_data, _x, _y, _d = return_from_psql(query, sum_names, _var_names, _x_var, _y_var)

        _yLab = 'Depth (m))'  
        _xLab = 'Longitude (deg) along '+ r'%s $^\circ$' % abs(line)+ ['S' if lat<0 else 'N' for lat in [line]][0]
        _basemap = False
        _lonLat_params = (limits, (line, line))
        
    if traj_type == 'NS_section':
        query = "select "+ cols+ " from woa13 where latitude> {} AND latitude< {} AND longitude>{} and longitude<{};".format(str(limits[0]), str(limits[1]), str(traj_bounds[0]), str(traj_bounds[1]))
        
        _x_var = 'latitude'
        _y_var = 'depth'
        
        cluster_d, _feat_data, _x, _y, _d = return_from_psql(query, sum_names, _var_names, _x_var, _y_var)

        _yLab = 'Depth (m)'  
        _xLab = 'Latitude (deg) along '+ r'%s $^\circ$' % abs(line)+ ['W' if lon<0 else 'E' for lon in [line]][0]
        _basemap = False
        _lonLat_params = ((line, line), limits)

    return _x, _y, _d, _feat_data, _basemap, _xLab, _yLab, _lonLat_params

    
def get_column(lat_bounds, lon_bounds, _var_names):
    sum_names = ['station', 'longitude', 'latitude', 'depth'] + _var_names
    cols = ', '.join(sum_names)
    query = "select "+ cols+ " from woa13 where latitude> {} AND latitude< {} AND longitude>{} and longitude<{};".format(str(lat_bounds[0]), str(lat_bounds[1]), str(lon_bounds[0]), str(lon_bounds[1]))
    
    _x_var = 'longitude'
    _y_var = 'latitude'

    cluster_d, _feat_data, _x, _y, _d = return_from_psql(query, sum_names, _var_names, _x_var, _y_var)
    _basemap = True
    _lonLat_params = [lon_bounds, lat_bounds]#None
    
    return _x, _y, _d, _feat_data, _basemap, _x_var.title()+' (deg)', _y_var.title()+' (deg)', _lonLat_params
   '''