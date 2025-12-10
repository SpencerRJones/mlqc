'''
Contains functions for reading surface data and attaching surface
type to pixels based on their geolocation and time.
'''

import numpy as np
import xarray as xr
import glob
import paths
from .utils import array_funcs

########################################################################

def attach_gpm_sfctype(lats, lons, time, sensor, keepdims=None, ncpus=8):
    
    '''
    Attaches surface type from CSU surface files. The CSU surface maps
    are created by Paula Brown, Dept. of Atmospheric Science, Colorado
    State University.
    '''

    sfccode_path = paths.sfc_datapath

    if np.any(np.isnan(lats)):
        raise ValueError(f'Lat array contains NaNs at {np.where(np.isnan(lats))}')
    if np.any(np.isnan(lons)):
        raise ValueError(f'Lon array contains NaNs at {np.where(np.isnan(lons))}')
    if lats.shape != lons.shape:
        raise ValueError(f'Shapes of lat and lon arrays not consistent.')

    if lats.shape != time.shape:
        if lats.shape[0] == time.size: #They are matched by scan
            time = array_funcs.copy_columns(time, lats.shape[1])
        else:
            raise ValueError(f'Shapes of lat/lon arrays {lats.shape} and time array {time.shape} not consistent.')

    if lats.ndim > 1:
        dims = lats.shape
        lats = lats.flatten()
        lons = lons.flatten()
        time = time.flatten()

    if time.dtype != '<M8[s]':
        raise ValueError(f'ScanTime array must be of type np.datetime64[s] (<M8[s]), but got {scantime.dtype}')

    nearest_day = array_funcs.round_datetime(time, np.timedelta64(1, 'D'))
    unique_days = np.unique(nearest_day)

    flist = []

    for iday, day in enumerate(unique_days):
        sfcfile = glob.glob(f"{sfccode_path}/{sensor}_surfmap_{str(day)[2:7].replace('-','')}_*.nc")[0]

        flist.append(sfcfile)

    unique_files = np.unique(np.array(flist))

    with xr.open_mfdataset(unique_files, combine='nested', concat_dim='time') as f:
        slats = f.lat.values
        slons = f.lon.values
        stime = f.time.sel(time=unique_days).values
        stype = f.SurfaceType.sel(time=unique_days).values

    lat_inc = slats[1] - slats[0]
    lon_inc = slons[1] - slons[0]
    lat_indcs = ((lats + 90.) / lat_inc).round().astype(int)
    lon_indcs = ((lons + 180.) / lon_inc).round().astype(int)
    lon_indcs[np.where(lon_indcs == slons.size)] = 0
    time_indcs = np.searchsorted(stime, nearest_day)

    sfctype = stype[time_indcs, lat_indcs, lon_indcs]

    if keepdims:
        sfctype = sfctype.reshape(dims)

    return sfctype.astype('int32')

########################################################################

