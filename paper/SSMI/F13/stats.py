import numpy as np


######################################################################

mean_ocn = np.array(
    [-0.0282637,   0.09551311,  0.03859631, -0.03289242, -0.04222305,
     -0.01801941, -0.0272838,  -0.04485523,  0.10969141]
)

std_ocn  = np.array(
    [0.7372899,  0.956595,   1.5781353,  0.757869,   1.0383261,
     0.92768514, 1.4900446,  1.0026524,  1.7874864 ]
)

mean_non = np.array(
    [-0.06133469,  0.14145795,  0.04045961,  0.02687342, -0.05239937,
     -0.05282856, -0.01309643, -0.04486879,  0.02469139]
)

std_non = np.array(
    [1.1838539, 2.4209967, 1.4346677, 1.3136077, 1.5939131,
     1.4972745, 1.645995,  1.7450027, 2.1314354]
)

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


