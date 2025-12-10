'''
Commonly used operations on arrays
'''
import numpy as np


########################################################################

def find_nan_rows(a, return_good=False):

    nanrows = np.unique(np.where(np.any(np.isnan(a), axis=1))[0])

    if return_good:
        return np.where(np.all(~np.isnan(a), axis=1))[0]

    return nanrows

########################################################################


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

