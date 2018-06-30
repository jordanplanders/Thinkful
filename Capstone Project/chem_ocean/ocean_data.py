import numpy as np
import sqlalchemy
from collections import defaultdict



def connect(user, password, db, host='localhost', port=5432):
    '''Returns a connection and a metadata object'''
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
    _latLon_params = None
    
    return _x, _y, _d, _feat_data, _basemap, _x_var.title()+' (deg)', _y_var.title()+' (deg)', _latLon_params
    
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
        _latLon_params = (limits, (line, line))
        
    if traj_type == 'NS_section':
        query = "select "+ cols+ " from woa13 where latitude> {} AND latitude< {} AND longitude>{} and longitude<{};".format(str(limits[0]), str(limits[1]), str(traj_bounds[0]), str(traj_bounds[1]))
        
        _x_var = 'latitude'
        _y_var = 'depth'
        
        cluster_d, _feat_data, _x, _y, _d = return_from_psql(query, sum_names, _var_names, _x_var, _y_var)

        _yLab = 'Depth (m)'  
        _xLab = 'Latitude (deg) along '+ r'%s $^\circ$' % abs(line)+ ['W' if lon<0 else 'E' for lon in [line]][0]
        _basemap = False
        _latLon_params = ((line, line), limits)

    return _x, _y, _d, _feat_data, _basemap, _xLab, _yLab, _latLon_params

    
def get_column(lat_bounds, lon_bounds, _var_names):
    sum_names = ['station', 'longitude', 'latitude', 'depth'] + _var_names
    cols = ', '.join(sum_names)
    query = "select "+ cols+ " from woa13 where latitude> {} AND latitude< {} AND longitude>{} and longitude<{};".format(str(lat_bounds[0]), str(lat_bounds[1]), str(lon_bounds[0]), str(lon_bounds[1]))
    
    _x_var = 'longitude'
    _y_var = 'latitude'

    cluster_d, _feat_data, _x, _y, _d = return_from_psql(query, sum_names, _var_names, _x_var, _y_var)
    _basemap = True
    _latLon_params = None
    
    return _x, _y, _d, _feat_data, _basemap, _x_var.title()+' (deg)', _y_var.title()+' (deg)', _latLon_params
    