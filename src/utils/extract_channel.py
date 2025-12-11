import numpy as np
from src import sensor_info

def extract_channel(Tb_array, chan):

    '''
    Use in preparing training data.

    Assumes Tb array is [m x n] where m (rows) are samples
    and n (columns) is the number of channels.

    Passing in the channel description splits the data so
    that the specified channel is its own vector y and the
    rest are kept as predictors x.

    Inputs:
        Tb_array    |  ndarray of Tbs
        chan        |  string of channel name
    Outputs:
        x           |  matrix of predictors (other channels)
        y           |  vector of predictands (the missing channel)
        
    '''
    
    chan_desc = np.array(sensor_info.channel_descriptions)


    chan_indx = np.where(chan == chan_desc)[0]

    if np.size(chan_indx) == 0:
        raise ValueError(f'Channel description must be in list {chan_desc}.')

    y = Tb_array[:,chan_indx]
    x = Tb_array[:,np.delete(np.arange(0,len(chan_desc)),chan_indx)]

    return x, y
