import pandas as pd 
from itertools import islice
import re
#from StringIO import StringIO
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import sys, pickle, time

def conv360to180_lons(_lon):
    try:
        for ik in xrange(len(_lon)):
            if _lon[ik] > 180:
                _lon[ik] = _lon[ik]-360
            else: 
                _lon[ik] = _lon[ik]
    except:
        if _lon > 180:
            _lon = _lon-360
        else: 
            _lon = _lon
    return _lon

def find_data_start(_file_path):
	    still_looking = 1
	    counter = 0
	    extra = 1
	    for still_looking in xrange(1,8):
	        with open(_file_path) as mess_file:
	            head = list(islice(mess_file,(still_looking-1)*10,still_looking*10))
	            mess_file.close()
	        for line in head:
	            if '\\' in line[-5:]:
	                extra +=1
	            if line[0].isalpha():
	                index = (still_looking-1)*10 + counter #+ extra
	                print(line)
	                return index
	            counter =+ 1

def oc_data_2df(_file_path, **kwargs):
    first_line = find_data_start(_file_path)
    _nrows = None
    _ncols = None 
    if kwargs:
        if '_cols' in kwargs:
            col_dict = {'Cruise': 'Cruise', 
                'Station': 'Station',
                'Longitude':'Longitude [degrees_east]',
                'Latitude': 'Latitude [degrees_north]',
                'Depth':'Depth [m]',
                'Temperature':'Temperature [degC]',
                'Salinity':'Salinity [psu]',
                'Oxygen':'Oxygen [ml/l]',
                'Oxygen Saturation':'Oxygen Saturation [%]',
                'AOU':'AOU [ml/l]',
                'Phosphate':'Phosphate [~$m~#mol/l]',
                'Nitrate':'Nitrate [~$m~#mol/l]',
                'Silicate':'Silicate [~$m~#mol/l]',
                'Mean Temperature': 'Mean Temperature [degC]',
                'Mean Salinity':'Mean Salinity [psu]',
                'Mean Oxygen':'Mean Oxygen [ml/l]',
                'Mean Oxygen Saturation': 'Mean Oxygen Saturation [%]',
                'Mean AOU': 'Mean AOU [ml/l]',
                'Mean Phosphate':'Mean Phosphate [~$m~#mol/l]',
                'Mean Nitrate': 'Mean Nitrate [~$m~#mol/l]',
                'Mean Silicate':'Mean Silicate [~$m~#mol/l]'}
            _cols = [col_dict[col] for col in kwargs['_cols']]
        if '_nrows' in kwargs:
            _nrows = kwargs['_nrows']
        _df = pd.read_csv(_file_path, skiprows=first_line, delimiter="\t",usecols=_cols, nrows = _nrows)
        pass
    '''
    else:
        _df = pd.read_csv(_file_path, skiprows=first_line, delimiter="\t", nrows = _nrows)
        pass 
    '''    
    _unit_dict = dict()
    try:
        for name in _df.columns:
            string = str(name)
            if 'QF' in string:
                _df = _df.drop(name,1) 
            if 'yyyy-mm-ddThh' in string:
                _df = _df.drop(name,1)
            if 'Type' in string:
                _df = _df.drop(name,1)
            if '[' in string:
                split_ind = string.find('[')
                header = string[0:split_ind-1]
                unit = string[split_ind+1:-1]
                _unit_dict[header] = unit
                _df=_df.rename(columns = {name:unicode(header, "utf-8")})
            if ':' in string:
                split_ind = string.find(':')
                header = string[0:split_ind]
                #unit_dict[header] = string[split_ind+1:-1]
                _df=_df.rename(columns = {name:unicode(header, "utf-8")})
    except:
        pass
        # print "columns already dropped"
    
    _df['Station'] = _df['Station'].fillna(method = 'ffill')
    for col in _df.columns:
        if col in ['Latitude', 'Longitude']:
            _df[col] = _df[col].replace('%','',regex=True).astype('float').fillna(method = 'ffill')
        if col in ['Depth','Temperature', 'Salinity', 'Oxygen', 'Oxygen Saturation', 'AOU', 'Phosphate', 'Nitrate', 'Silicate']:
            _df[col] = _df[col].replace('%','',regex=True).astype('float')
     
    _df = _df[_df.Station != 'rschlitz@GSYSM234-1']
    _df['Longitude'] = _df['Longitude'].map(conv360to180_lons)
    return _df, _unit_dict

def make_arrays(_flat_records, _flat_record_names):
    temp = []
    for col in xrange(len(_flat_records[0])):
        temp.append(np.array([_flat_records[im][col] for im in xrange(len(_flat_records))]).reshape(len(_flat_records),1).astype(float))
    _flat_records = np.concatenate(temp, axis = 1)
    
    _struct_array = np.zeros(_flat_records.shape[0], dtype={'names':_flat_record_names, 'formats': [float for ik in xrange(len(_flat_record_names))]})
    for ip in xrange(len(_struct_array)):
        _struct_array[ip]= tuple(_flat_records[ip])
        
    min_max_print(_flat_record_names,_flat_records)
    return _flat_records, _struct_array

def build_oc_station_db(_df):
    grouped = _df.groupby('Station')
    station_flat = []
    oc_np_db = dict()
    #station_data_sumary = dict()
    #station_dict = dict()

    counter = 0 
    for group in grouped:
            count = group[1]['Longitude'].size
            min_depth = min(group[1]['Depth'])
            max_depth = max(group[1]['Depth'])
            lon = min(group[1]['Longitude'])
            lat = min(group[1]['Latitude'])
            station_flat.append([group[0],lon, lat, count, min_depth, max_depth])
            oc_np_db[group[0]] = [np.array([lon]), np.array([lat])] +[group[1][_df.columns[ik]].as_matrix() for ik in xrange(3,len(_df.columns))]
            '''
            var_cts_vect = [group[1][_df.columns[ik]].size for ik in xrange(4,len(_df.columns))]
            # check if all stations have all variable sampled at all depths
            for im in xrange(len(var_cts_vect)):
                if var_cts_vect[im] != var_cts_vect[0]: ctr =1
                else: ctr = 0
            counter += ctr    
            station_data_sumary[group[0]] = (lon, lat, count, (min_depth, max_depth), var_cts_vect)
            '''
            #station_dict[group[0]] = (lon, lat, count, (min_depth, max_depth))

    oc_np_db_names = list(enumerate(_df.columns[1:]))
    print(oc_np_db_names)
    return station_flat, oc_np_db, oc_np_db_names

def min_max_print (name_vect, flat_array):
    for col in xrange(1,len(flat_array[0])):
        print(name_vect[col], len(flat_array[:,col]), np.min(flat_array[:,col].astype(float)), np.max(flat_array[:,col]))