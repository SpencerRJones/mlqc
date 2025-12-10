'''
Utility functions for operations that work on Level 1C data files.
'''

import numpy as np
import xarray as xr
from src import sensor_info
import glob


########################################################################

def read_gmi_l1c(file):

    '''
    Reads L1C file and orients data for making Tb predictions.

    Tb array will be set up as follows:
        Tbs = [m x n], where m is the number of samples and n is the
              number of channels (features)
        1-2:   10V and H
        3-4:   19V and H
        5:     24V
        6-7:   37V and H
        8-9:   89V and H
        10-11: 166V and H
        12:    183+-3 V
        13:    183+-7 V
    '''

    tb_dim = sensor_info.nfeatures
    qual_dim = sensor_info.nscangroups

    with xr.open_dataset(file, group='S1', decode_timedelta=False) as f:
        lat = f.Latitude.values[:,25:-25]
        lon = f.Longitude.values[:,25:-25]

    nscans, npixs = lat.shape
    Tbs = np.zeros([nscans,npixs,tb_dim], dtype=np.float32)
    qual = np.zeros([nscans,npixs,qual_dim], dtype=np.int32)

    with xr.open_dataset(file, group='S1', decode_timedelta=False) as f:
        qual[:,:,0]  = f.Quality[:,25:-25].values
        Tbs[:,:,0:2] = f.Tc[:,25:-25,0:2].values #10V and H
        Tbs[:,:,2:4] = f.Tc[:,25:-25,2:4].values #19V and H
        Tbs[:,:,4]   = f.Tc[:,25:-25,4].values   #23V
        Tbs[:,:,5:7] = f.Tc[:,25:-25,5:7].values #37V and H
        Tbs[:,:,7:9]  = f.Tc[:,25:-25,7:].values #89V and H

    with xr.open_dataset(file, group='S2', decode_timedelta=False) as f:
        qual[:,:,1] = f.Quality[:,25:-25].values
        Tbs[:,:,9:11] = f.Tc.values[:,25:-25,0:2] #166V and H
        Tbs[:,:,11:]  = f.Tc.values[:,25:-25,2:] #183+-3 and 183+-7 V

    with xr.open_dataset(file, group='S1/ScanTime', decode_timedelta=False) as f:
        scantime = (f.Year.values, f.Month.values, f.DayOfMonth.values, f.Hour.values, f.Minute.values, f.Second.values)

    #Change scan time format from L1C to datetime format for easier use
    scantime = scantime2datetime(scantime)

    data = {}

    data['lat'] = lat
    data['lon'] = lon
    data['scantime'] = scantime
    data['Tbs'] = Tbs
    data['qual'] = qual

    return data

########################################################################

def scantime2datetime(scantime):

    year, month, day, hour, minute, second = scantime

    year = year.astype('i')
    month = month.astype('i')
    day = day.astype('i')
    hour = hour.astype('i')
    minute = minute.astype('i')
    second = second.astype('i')

    datestr = [f'{year[i]}-{month[i]:02}-{day[i]:02}T{hour[i]:02}:{minute[i]:02}:{second[i]:02}'
               for i in range(year.size)]

    date_array = np.array(datestr, dtype='datetime64[s]')

    return date_array

