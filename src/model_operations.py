import numpy as np
from src import sensor_info
import paths
from src.utils import L1C, extract_channel
from src import surface
import glob
from src.classes import model_class
import torch

########################################################################

def ml_quality_control(file, error_threshold, keep_dims=True,
                       ignore_l1cflag=False):

    '''
    Main function for creating ML-based quality control flag.

    Inputs:
        file            | Input Level 1C file
        error_threshold | Number of standard deviations of prediction error
                          to be considered "bad".
        keep_dims       | (Optional) Keep original file dimensions, i.e. 
                          mlqcflag.shape == Tbs.shape
    Output:
        mlqcflag        | Quality control flag, one value for each pixel
                          and channel.
    
    Flag values:
        0  = Good (Predictions were within specified error threshold)
        -1 = Problem with Level 1C flag
        1  = Bad (Predictions were outside of specified error threshold)
    '''

    #General setup
    chans = sensor_info.channel_descriptions
    nscangroups = sensor_info.nscangroups

    #Read the file
    data = L1C.read_gmi_l1c(file)

    #Get dims before reshaping
    nscans, npixs, nchans = data['Tbs'].shape

    #Reshape for mlqc model
    Tbs      = data['Tbs'].reshape(-1, nchans)
    lat      = data['lat'].reshape(-1)
    lon      = data['lon'].reshape(-1)
    scantime = np.repeat(data['scantime'], npixs)
    qual     = data['qual'].reshape(-1,nscangroups)

    nsamples = Tbs.shape[0]

    #Create initial qc flag
    ml_qcflag = np.zeros(Tbs.shape, dtype=np.int32)
    ml_qcflag[:] = -99
    internal_qual = np.zeros(nsamples, dtype=np.byte)

    #Check for bad quality in Level 1C
    if not ignore_l1cflag:
        l1c_bad = np.any(qual != 0, axis=1)
        ml_qcflag[l1c_bad,:] = -1
        internal_qual[l1c_bad] = 1
        if np.all(ml_qcflag == -1): #If all bad, abort
            return ml_qcflag

    #Check for any remaining NaNs in L1C:
    nans = np.logical_or(np.isnan(lat), np.isnan(lon))
    internal_qual[nans] = 1
    nans = np.any(np.isnan(Tbs), axis=1)
    internal_qual[nans] = 1

    good = internal_qual == 0

    #Attach surface type to good pixels
    sfctype = np.zeros([nsamples], dtype=np.int32)
    sfctype[:] = -99
    sfctype[good] = surface.attach_gpm_sfctype(lat[good], 
                                               lon[good], 
                                               scantime[good],
                                               sensor=sensor_info.sensor)

    #Load in model tree
    model_tree = load_model_tree()

    #Predict Tbs
    Tbs_pred = np.zeros_like(Tbs)
    Tbs_pred[:] = np.nan
    Tbs_pred[good] = run_predictions(Tbs[good], 
                                     sfctype[good], 
                                     model_tree,
                                     keep_dims=False)

    #Calculate standardized prediction errors
    epsilon_hat = np.zeros_like(Tbs)
    epsilon_hat[:] = np.nan
    epsilon_hat[good] = get_error(Tbs[good], Tbs_pred[good], sfctype[good])

    #Set flag
    ml_qcflag[epsilon_hat > error_threshold] = 1
    ml_qcflag[epsilon_hat <= error_threshold] = 0

    if keep_dims:
        ml_qcflag = ml_qcflag.reshape(nscans,npixs,nchans)

    return ml_qcflag

########################################################################


def load_model_tree():

    '''
    This function loads in all models for channel prediction. Model
    structure and definitions are loaded in from model_class.py. Weights
    and biases are loaded in from .pt files located in models/.
    '''
    
    model_path = f'{paths.model_path}/{paths.model}'
    
    feature_names = sensor_info.feature_descriptions
    satellite     = sensor_info.satellite
    sensor        = sensor_info.sensor

    model_tree = {}

    #Configuration for ocean model
    input_size = sensor_info.nfeatures - 1
    hidden_size = 256
    output_size = 1

    #Load in ocean models
    for ichan in feature_names:

        model_file = glob.glob(f'{model_path}/{sensor}_{satellite}_channel_predictor_{ichan}_ocean.pt')[0]

        model_tree[f'{model_file.split("/")[-1][:-3]}'] = model_class.channel_predictor(input_size, hidden_size, output_size)
        model_tree[f'{model_file.split("/")[-1][:-3]}'].load_state_dict(torch.load(model_file, weights_only=True))
        model_tree[f'{model_file.split("/")[-1][:-3]}'].eval()

    #Configuration for nonocean model
    input_size = sensor_info.nfeatures - 1 + 1 #(n-1) channels and surface code

    #Load in nonocean model
    for ichan in feature_names:
        model_file = glob.glob(f'{model_path}/{sensor}_{satellite}_channel_predictor_{ichan}_nonocean.pt')[0]

        model_tree[f'{model_file.split("/")[-1][:-3]}'] = model_class.channel_predictor(input_size, hidden_size, output_size)
        model_tree[f'{model_file.split("/")[-1][:-3]}'].load_state_dict(torch.load(model_file, weights_only=True))
        model_tree[f'{model_file.split("/")[-1][:-3]}'].eval()

    return model_tree

########################################################################


def run_predictions(tbs, sfccodes, model_tree, keep_dims=True):

    '''
    Runs model tree on Tb array to make predictions
    '''

    if tbs.ndim == 3:
        nscans, npixs, nchans = tbs.shape
        tbs = tbs.reshape(-1, nchans)

    nsamples, nchans = tbs.shape

    sat = sensor_info.satellite
    sens = sensor_info.sensor

    #A couple of sanity checks
    if nchans != sensor_info.nfeatures:
        raise ValueError(f'Shape of obs vector {tbs.shape} not compatible with nfeatures.')
    if sfccodes.size != nsamples:
        raise ValueError(f'Number of surface codes {sfccodes.size} != number of obs vector rows {nsamples}.')

    #Separate ocean and other surface codes
    ocean = sfccodes == 1
    nonocean = ~ocean

    features = sensor_info.feature_descriptions

    tbs_pred = np.zeros_like(tbs)

    #Predict over ocean and land separately:
    for ichan, channel in enumerate(features):

        predictors, _ = extract_channel(tbs, channel)

        ocean_model = model_tree[f'{sens}_{sat}_channel_predictor_{channel}_ocean']
        nonocean_model = model_tree[f'{sens}_{sat}_channel_predictor_{channel}_nonocean']

        #Predict over ocean if there are any
        if np.sum(ocean) > 0:
            x_ocean = predictors[ocean]
            with torch.no_grad():
                tbs_pred[ocean,ichan] = ocean_model(torch.tensor(x_ocean))[:,0]
        #Predict over nonocean surfaces if there are any
        if np.sum(nonocean) > 0:
            x_nonocean = predictors[nonocean]
            x_nonocean = np.concatenate((x_nonocean, sfccodes[nonocean][:,None].astype(np.float32)), axis=1)
            with torch.no_grad():
                tbs_pred[nonocean,ichan] = nonocean_model(torch.tensor(x_nonocean))[:,0]

        if keep_dims:
            tbs_pred = tbs_pred.reshape(nscans,npixs,nchans)


    return tbs_pred

########################################################################

def get_error(y_pred, y_true, sfccodes, keep_dims=True):
    
    '''
    Calculates standardized prediction error over ocean and nonocean
    surfaces. Error is calculated as (epsilon-mu)/sigma, where
    epsilon = y_true - y_pred is the prediction error in Tb space,
    mu = the mean and sigma = standard deviation of errors calcuated
    on the test dataset from training.
    '''

    if keep_dims:
        dims = y_pred.shape

    y_pred = y_pred.reshape(-1,sensor_info.nchannels)
    y_true = y_true.reshape(-1,sensor_info.nchannels)

    epsilon        = np.full(y_pred.shape, fill_value=np.nan)
    epsilon_hat    = np.full(y_pred.shape, fill_value=np.nan)

    #A couple of checks
    assert y_pred.shape == y_true.shape, f'Shapes of y_pred and y_true not compatible. y_pred={y_pred.shape}, y_true={y_true.shape}'
    assert y_pred.shape[0] == sfccodes.size, f'Shapes of y_pred and sfccodes not compatible. y_pred and sfccodes must align on first dimension. y_pred={y_pred.shape} and sfccodes={sfccodes.shape}'


    #Read in error stats
    mean_ocn = np.load(f'{paths.model_path}/{paths.model}/err_stats/mean_ocn.npy') 
    std_ocn  = np.load(f'{paths.model_path}/{paths.model}/err_stats/std_ocn.npy')  
    mean_non = np.load(f'{paths.model_path}/{paths.model}/err_stats/mean_non.npy') 
    std_non  = np.load(f'{paths.model_path}/{paths.model}/err_stats/std_non.npy')  

    ocean    = sfccodes == 1
    nonocean = sfccodes > 1

    epsilon[:] = y_true - y_pred
    epsilon_hat[ocean] = np.abs(epsilon[ocean] - mean_ocn) / std_ocn
    epsilon_hat[nonocean] = np.abs(epsilon[nonocean] - mean_non) / std_non

    return epsilon_hat
