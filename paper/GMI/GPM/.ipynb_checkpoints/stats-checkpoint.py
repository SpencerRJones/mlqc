import numpy as np


######################################################################

def mean_squared_error(pred, true, axis=None):

    '''
    Calculates the mean-squared error of an array
    of predictions and true values. Expects shape [m x n]
    where m is the sampling dimension and n is the 
    features dimension.

    MSE = 1 / n * sum((prediction - truth)^2)

    Inputs:
        pred    |   array of prediction values of shape [m x n]
        true    |   array of truth values of shape [m x n]
        axis (optional) | if provided, the axis across which to
                             apply the operation
    Outputs:
        mse     |   mean squared error
    '''


    if axis is not None:
        n = pred.shape[axis]
    else:
        n = pred.size

    mse = np.sum((pred - true)**2, axis=axis) / n

    return mse


######################################################################


def root_mean_squared_error(pred, true, axis=None):

    '''
    Calculates the root-mean-squared error of an array
    of predictions and true values. Expects shape [m x n]
    where m is the sampling dimension and n is the 
    features dimension.

    MSE = sqrt(1 / n * sum((prediction - truth)^2))

    Inputs:
        pred    |   array of prediction values of shape [m x n]
        true    |   array of truth values of shape [m x n]
        axis (optional) | if provided, the axis across which to
                             apply the operation
    Outputs:
        rmse     |   root mean squared error
    '''

    if axis is not None:
        n = pred.shape[axis]
    else:
        n = pred.size

    rmse = np.sqrt(np.sum((pred-true)**2, axis=axis) / n)

    return rmse


######################################################################

def mean_absolute_error(pred, true, axis=None):

    '''
    Calculates the mean absolute error of an array
    of predictions and true values. Expects shape [m x n]
    where m is the sampling dimension and n is the 
    features dimension.

    MAE = 1/n * sum(|pred - true|)

    Inputs:
        pred    |   array of prediction values of shape [m x n]
        true    |   array of truth values of shape [m x n]
        axis (optional) | if provided, the axis across which to
                             apply the operation
    Outputs:
        mae    |    mean absolute error
    '''

    if axis is not None:
        n = pred.shape[axis]
    else:
        n = pred.size

    mae = np.sum(np.abs(pred - true), axis=axis) / n
    
    return mae


######################################################################

def running_mean(a, window_size, ends='drop', axis=None, method='center'):

    '''
    Calculates a running mean on an array of data.

    Inputs:
        a            |   array of data
        window_size  |   number of elements to average together, i.e.
                            length of running window
        ends (optional) | what to do with the ends of the data where the
                            averaging window reaches beyond the array edges.
                            -Accepts "drop" or "wrap".
                            -Drop (default): ignore the ends of the data and
                            do not include in averages
                            -Wrap: circle boundary condition, i.e. the data
                            at the end becomes averaged with the beginning and
                            vice versa.

        axis (optional)   | apply window along axis (not currently supported)
        method (optional) | method of applying averaging window.
                            -Accepts "center", "forward", or "backward"
                            -Default is "center"
                            -Backward: values leading up to current value are
                            averaged together
                            -Forward: values beyond the current value are
                            averaged together
                            -Center: window is split where the current value
                            is in the middle and averaged with previous and
                            succeeding values.
                            
        
    '''

    if len(a.shape) != 1: raise ValueError(f'Array must be 1D, but got shape = {a.shape}')

    a = np.tile(a, (window_size,1)).T

    if method == 'backward':
        for i in np.arange(1,window_size):
            a[:,i] = np.roll(a[:,i], i)
        a = np.mean(a, axis=1)
        if ends == 'drop':
            a = a[window_size-1:]

    elif method == 'forward':
        for i in np.arange(1,window_size):
            a[:,i] = np.roll(a[:,i], -i)
        a = np.mean(a, axis=1)
        if ends == 'drop':
            a = a[:-window_size+1]

    elif method == 'center':
        for i in np.arange(1,window_size):
            a[:,i] = np.roll(a[:,i], -i)
        a = np.roll(a, window_size/2, axis=0)
        a = np.mean(a, axis=1)
        if ends == 'drop':
            a = a[window_size//2:-window_size//2+1]

    else:
        raise ValueError(f'method must be either "forward", "backward", or "center", but got {method}.')
        
    return a

############################################################################


