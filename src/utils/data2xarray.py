import numpy as np
import xarray as xr

def data2xarray(data_vars, var_names, dims, dim_names):

    '''
    
    Provides an easy interface to converting numpy data arrays
    to Xarray datasets.

    Inputs:
        data_vars       |  Tuple or list of data arrays (data1, data2, ...)
        var_names       |  Tuple or list of strings (varname1, varname2, ...)
        dims            |  Tuple or list of dimensions (dim1, dim2, ...)
                            -dimensions must be a tuple or list of integers
        dim_names       |  Tuple or list of names of the dimensions (name1, name2, ...)

    Outputs:
        Xarray.Dataset

    '''


    if len(data_vars) != len(var_names):
        raise ValueError(f'Numer of variables != number of variable names.')

    if len(dims) != len(dim_names):
        raise ValueError(f'Number of dimensions != number of dimension names.')

    var_dict = {}
    coord_dict = {}

    #Create dictionary of coordinates:
    for i,idim in enumerate(dims):
        coord_dict[f'{dim_names[i]}'] = np.arange(0,dims[i])
    
    for i,ivar in enumerate(data_vars):
        var      = data_vars[i]
        var_name = var_names[i]
        var_shape = np.shape(var)
        dim_list = []
        for j,jdim in enumerate(var_shape):
            dim_indx = np.where(jdim == np.array(dims))[0][0]
            dim_list.append(dim_names[dim_indx])

        var_dict[f'{var_names[i]}'] = (dim_list, var)


    dset = xr.Dataset(data_vars=var_dict, coords=coord_dict)

    return dset

