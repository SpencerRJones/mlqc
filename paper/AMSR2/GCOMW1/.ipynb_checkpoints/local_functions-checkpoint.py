import numpy as np
import xarray as xr
import sensor_info
import glob
import model_class
import torch

########################################################################

def read_amsr2_l1b(file):
    '''
    Reads L1B file and orients data for making Tb predictions.

    nscans: generally around 3200
    npixs: 243 and 486 (89 GHz is double-sampled)

    Tb array will be set up as follows:
        Tbs = [m x n], where m is the number of samples and n is the
              number of channels (features)
        1-2: 6 V/H
        3-4: 7 V/H
        5-6: 10V/H
        7-8: 19V/H
        9-10: 24V/H
        11-12: 37V/H
        13-14: 89V/H
    '''

    tb_dim = sensor_info.nfeatures
    qual_dim = sensor_info.nscangroups

    with xr.open_dataset(file, group='S1', decode_timedelta=False) as f:
        lat = f.Latitude_6.values
        lon = f.Longitude_6.values

        nscans, npixs = lat.shape
        Tbs = np.zeros([nscans,npixs,tb_dim], dtype=np.float32)
        qual = np.zeros([nscans,npixs,qual_dim], dtype=np.int32)

        Tbs[:,:,0] = f.Tb_6GHz_V.values
        Tbs[:,:,1] = f.Tb_6GHz_H.values
        Tbs[:,:,2] = f.Tb_7GHz_V.values
        Tbs[:,:,3] = f.Tb_7GHz_H.values
        Tbs[:,:,4] = f.Tb_10GHz_V.values
        Tbs[:,:,5] = f.Tb_10GHz_H.values
        Tbs[:,:,6] = f.Tb_18GHz_V.values
        Tbs[:,:,7] = f.Tb_18GHz_H.values
        Tbs[:,:,8] = f.Tb_23GHz_V.values
        Tbs[:,:,9] = f.Tb_23GHz_H.values
        Tbs[:,:,10] = f.Tb_36GHz_V.values
        Tbs[:,:,11] = f.Tb_36GHz_H.values
        #Tbs[:,:,12] = f.Tb_89GHz_V_A.values[:,::2]
        #Tbs[:,:,13] = f.Tb_89GHz_H_A.values[:,::2]
        qual[:,:,0] = f.Pixel_Data_Quality_6_to_36.values[:,::2]
        #qual[:,:,1] = f.Pixel_Data_Quality_89.values[:,::2]

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

def initial_qc(data_dict):
    
    '''
    Checks for NaNs in L1C file and gets rid of them
    '''

    lat = data_dict['lat']
    lon = data_dict['lon']
    scantime = data_dict['scantime']
    Tbs = data_dict['Tbs']
    qual = data_dict['qual']

    if np.any(np.isnan(lat)):
        
        goodscans = find_nan_rows(lat, return_good=True)

        lat = lat[goodscans]
        lon = lon[goodscans]
        scantime = scantime[goodscans]
        Tbs = Tbs[goodscans]
        qual = qual[goodscans]
        
    data = {}

    data['lat'] = lat
    data['lon'] = lon
    data['scantime'] = scantime
    data['Tbs'] = Tbs
    data['qual'] = qual
    
    return data


########################################################################


def split_data_indcs(x, train=80, test=10, val=10, device=None, randomize=False):

    '''
    Creates train/test/validation split for training data. Default is 80%/10%/10%.

    Inputs:
        x       |   Array of predictors. Expects shape (nsamples, nfeatures).
        y       |   Array of predictands. Expects shape (nsamples, nfeatures).
        train   |   Percentage of data to be used for training.
        test    |   Percentage of data to be used for testing.
        val     |   Percentage of data to be used for validation.
                        -Train + test + val must equal 100.
        device (optional)    |  Either 'cuda' or 'cpu'. 
        randomize (optional) |  Whether to shuffle the data before creating
                                    the splits. This functionality is currently
                                    not supported but is intended to be 
                                    implemented in the future.
    '''

    if train + test + val != 100:
        raise ValueError(f'train + test + val must equal 100%.')

    #if x.shape[0] != y.shape[0]:
    #    raise ValueError(f'Dimensions of x {x.shape} and y {y.shape} not compatible.')

    nsamples = x.shape[0]

    ntrain = int(nsamples * (train / 100.))
    ntest  = int(nsamples * (test / 100.))
    nval   = nsamples - ntrain - ntest

    indcs = np.arange(0,nsamples)

    train_indcs = indcs[0:ntrain]
    test_indcs  = indcs[ntrain:ntrain+ntest]
    val_indcs   = indcs[ntrain+ntest:]

    return train_indcs, test_indcs, val_indcs


########################################################################


def attach_gpm_sfctype(lats, lons, time, sensor, gpmversion=None, keepdims=None, ncpus=8):

    sfccode_path = '/edata2/spencer/gpm_surf'

    sensor_list = ['GMI', 'SSMI', 'SSMIS', 'AMSR2']

    if sensor not in sensor_list:
        raise ValueError(f'Sensor {sensor} not recognized. Supported sensors include {sensor_list}')

    if sensor == 'SSMIS':
        sensor = 'SSMI'
    elif sensor == 'AMSR2':
        sensor = 'SSMI'


    if np.any(np.isnan(lats)):
        raise ValueError(f'Lat array contains NaNs at {np.where(np.isnan(lats))}')
    if np.any(np.isnan(lons)):
        raise ValueError(f'Lon array contains Nans at {np.where(np.isnan(lons))}')

    if lats.shape != lons.shape:
        raise ValueError(f'Shapes of lat and lon arrays not consistent.')

    if lats.shape != time.shape:
        if lats.shape[0] == time.size: #They are matched by scan
            time = copy_columns(time, lats.shape[1])
        else:
            raise ValueError(f'Shapes of lat/lon arrays {lats.shape} and time array {time.shape} not consistent.')

    if lats.ndim > 1:
        
        dims = lats.shape
        lats = lats.flatten()
        lons = lons.flatten()
        time = time.flatten()


    if time.dtype != '<M8[s]':
        raise ValueError(f'ScanTime array must be of type np.datetime64[s] (<M8[s]), but got {scantime.dtype}')

    nearest_day = round_datetime(time, np.timedelta64(1, 'D'))
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


def extract_columns(a, cols):

    '''
    Splits array a into two arrays by extracting specified columns.

    Inputs:
        a      | Array to extract columns from, with shape = (m, n)
        cols   | Indicies of columns to extract
    Outputs:
        b      | Array of shape = (m, ncols) that consists of desired
                 columns.
        c      | Array of shape = (m, n - ncols) that is array a without
                 columns specified.
    '''

    b = a[:,cols]
    c = delete_columns(a, cols)

    return b, c

def delete_rows(a, rows):

    '''
    Deletes specified rows from array a using numpy's logical
    indexing.

    Inputs:
        a        |  array of shape (m,n)
        rows     |  list or tuple of row indices to delete

    Outputs:
        b        |  array of shape (m,n) with rows deleted
    '''

    m, n = a.shape

    l = np.ones_like(a, dtype=np.bool)
    l[rows,:] = False
    b = a[l].reshape(-1,n)

    return b


def delete_columns(a, cols):

    '''
    Similar to delete_rows
    '''

    m, n = a.shape

    l = np.ones_like(a, dtype=np.bool)
    l[:,cols] = False
    b = a[l].reshape(m,-1)

    return b


def find_nan_rows(a, return_good=False):

    nanrows = np.unique(np.where(np.any(np.isnan(a), axis=1))[0])

    if return_good:
        return np.where(np.all(~np.isnan(a), axis=1))[0]

    return nanrows


def copy_columns(a, n, cols=None):

    '''
    Copies columns along the 2nd axis if input array is of shape
    [rows x columns]. Higher dimensions currently not supported.

    Inputs:
        a         |   Array of values. If single dimensional, columns
                        are stacked n times along the new 2nd dimension.
                        If 2D, columns are repeated along 2nd axis.
        n         |   Number of times to copy the column
        cols      |   (Optional) A subset of the columns to copy. Expects
                        a list or array with column indices.
                        NOT CURRENTLY SUPPORTED

    Outputs:
        b         |   New array.


    Examples:
        IN:
        a = np.array([1,2,3,4,5,6])
        copy_columns(a, n=2)

        OUT:
        array([[1, 1],
               [2, 2],
               [3, 3],
               [4, 4],
               [5, 5],
               [6, 6]])

        IN:
        a = np.array([[1,2,3],[4,5,6]])
        copy_columns(a, n=2)

        OUT:
        array([[1, 1, 2, 2, 3, 3],
               [4, 4, 5, 5, 6, 6]])

    '''


    if a.ndim == 1:
        b = np.repeat(a, n).reshape(-1, n)
        return b

    if a.ndim == 2:

        nrows, ncols = a.shape

        if cols is not None:
            pass

        else:  #Copy all
            b = np.repeat(a, n).reshape(nrows, -1)
            return b




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


def round_datetime(time_arr, round_unit):

    '''
    Rounds numpy datetime64 data to nearest unit specified.

    Inputs:
        time_arr   |  array of type datetime64[s]
        round_unit |  unit of time to round to. Must be a single
                        value of type timedelta64
    Outputs:
        rounded    |  array of type datetime64[s]

    Example:
        input: 2025-12-25T12:45:39, round_unit=np.timedelta64(12,'h') -->
        output: 2025-12-25T12:00:00

    '''

    dttm_seconds = time_arr.astype('int64')
    one_unit     = round_unit / np.timedelta64(1, 's')
    half_unit    = one_unit / 2

    rounded = (dttm_seconds + half_unit) // one_unit * one_unit

    return rounded.astype('datetime64[s]')

########################################################################
