'''
Creates training data for GMI channel predictor from L1C files.

L1C on-swath data is available from the NASA PPS at https://gpm.nasa.gov/data/directory.

Reads in L1C files and extracts best quality data for training.

nscans: generally around 3000
npixs:  171*
    *Note: GMI high frequency channels (>37 GHz) don't go all the way to the swath edges.
           The actual number of pixels is 221, but we cut off 25 from each side that do
           not measure the high frequencies. These have a fill value for high frequency
           channel brightness temperatures (Tbs) in the data.

Tb array is set up as follows:
        Tbs = [m x n], where m is the number of samples and n is the number of channels
              (features)
        1-2: 10V/H
        3-4: 19V/H
        5:   24V
        6-7: 37V/H
        8-9: 89V/H
        10-11: 166V/H
        12: 183+-3 V
        13: 183+-7 V
'''

import numpy as np
import xarray as xr
import glob
from tqdm import tqdm
import paths
from src import surface, sensor_info
from src.utils import L1C, array_funcs, data2xarray

datapath = paths.l1c_datapath
modelpath = paths.model_path
training_datapath = paths.training_datapath
satellite = sensor_info.satellite
sensor = sensor_info.sensor

#Get list of files
file_list = glob.glob(f'{datapath}/1C-R.{satellite}.{sensor}.*.HDF5'); file_list.sort()

#Change this line if you want to subset your data
flist = file_list[:]

#Loop through files and get good quality data.
for i, ifile in enumerate(tqdm(flist, desc="Processing Files")):

    data = L1C.read_gmi_l1c(ifile)

    lat = data['lat']
    lon = data['lon']
    scantime = data['scantime']
    Tbs = data['Tbs']
    qual = data['qual']

    #Get only good quality data and reshape:
    goodqual = np.all(qual == 0, axis=2)
    all_bad = np.all(goodqual == False)
    if all_bad:
        print('all were bad.')
        continue
    lat = lat[goodqual]
    lon = lon[goodqual]
    scantime = scantime[np.where(goodqual)[0]]
    Tbs = Tbs[goodqual]

    #Check for NaNs (shouldn't be any if all good, but I've seen some)
    nonans = array_funcs.find_nan_rows(Tbs, return_good=True)
    Tbs = Tbs[nonans]
    lat = lat[nonans]
    lon = lon[nonans]
    scantime = scantime[nonans]

    #Attach GPROF surface map data to each pixel
    sfctype = surface.attach_gpm_sfctype(lat, lon, scantime, sensor=sensor)

    npixs = lat.size

    #Output as NetCDF
    dset = data2xarray(data_vars = [lat, lon, scantime, sfctype, Tbs],
                       var_names = ['latitude', 'longitude', 'scantime', 'sfctype', 'Tbs'],
                       dims = [npixs, sensor_info.nfeatures],
                       dim_names = ['pixels', 'channels'])

    if i == 0:
        training_dataset = dset
    else:
        training_dataset = xr.concat((training_dataset, dset), dim='pixels')

training_dataset.to_netcdf(f'{training_datapath}/{satellite}_training_data.nc', engine='netcdf4')
