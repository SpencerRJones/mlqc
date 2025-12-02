import numpy as np
import sensor_info
import local_functions
import model_class
import stats
import torch
import glob


########################################################################

def ml_quality_control(file, error_threshold=7., keep_dims=True):

    '''
    Function for creating final ML-based quality control flag.
    
    Inputs:
        file            | Input Level 1C file
        error_threshold | Number of standard deviations of prediction error
                          to be considered "bad".
        keep_dims       | Keep array dimensions

    Outputs:
        ml_qcflag       | Quality control flag array, one value for each
                          pixel and channel. If keep_dims is True, output
                          array has shape (nscans, npixs, nchannels)

    Flag Values:
        0  = Good (Predictions were within specified error threshold)
        -1 = Problem with L1C quality flag
        1  = Bad (Predictions were outside of error threshold)
    '''
    
    #General setup
    chans = sensor_info.channel_descriptions
    nscangroups = sensor_info.nscangroups

    mean_ocn = np.fromfile('mean_ocn.arr', sep='', dtype=np.float32)   
    std_ocn  = np.fromfile('std_ocn.arr', sep='', dtype=np.float32)
    mean_non = np.fromfile('mean_non.arr', sep='', dtype=np.float32)
    std_non  = np.fromfile('std_non.arr', sep='', dtype=np.float32)
    
    #Read the file
    data = local_functions.read_ssmis_l1c(file)

    nscans, npixs, nchans = data['Tbs'].shape
    
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

    #Check for bad quality in L1C:
    l1c_bad = np.any(qual != 0, axis=1)
    ml_qcflag[l1c_bad,:] = -1
    internal_qual[l1c_bad] = 1

    #Check if all data is bad:
    if np.all(ml_qcflag == -1):
        return ml_qcflag

    #Check for any remaining nans from L1C:
    nans = np.logical_or(np.isnan(lat), np.isnan(lon))
    internal_qual[nans] = 1
    nans = np.any(np.isnan(Tbs), axis=1)
    internal_qual[nans] = 1
    
    good = internal_qual == 0

    # Tbs = Tbs[good]
    # lat = lat[good]
    # lon = lon[good]
    # scantime = scantime[good]
    
    #Attach surface type to good pixels
    sfctype = np.zeros([nsamples], dtype=np.int32)
    sfctype[:] = -99
    sfctype[good] = local_functions.attach_gpm_sfctype(lat[good], lon[good], scantime[good], 
                                                 sensor=sensor_info.sensor)

    #Load in model tree
    model_tree = load_model_tree()

    #Predict Tbs
    Tbs_pred = np.zeros_like(Tbs)
    Tbs_pred[:] = np.nan
    Tbs_pred[good] = run_predictions(Tbs[good], sfctype[good], model_tree, keep_dims=False)

    #Calculate prediction errors in standardized (sigma) space
    epsilon_hat    = np.zeros_like(Tbs)
    epsilon        = np.zeros_like(Tbs)
    epsilon_hat[:] = np.nan
    epsilon[:]     = np.nan

    #Get indicies corresponding to correct surface
    ocean    = sfctype == 1
    nonocean = sfctype > 1

    #Get error
    epsilon        = Tbs - Tbs_pred
    epsilon[~good] = np.nan  #Double check that all bad quality pixels are not considered

    epsilon_hat[ocean] = np.abs(epsilon[ocean] - mean_ocn) / std_ocn
    epsilon_hat[nonocean] = np.abs(epsilon[nonocean] - mean_non) / std_non

    #Find where errors exceed error threshold and set those to bad quality
    ml_qcflag[np.where(epsilon_hat > error_threshold)] = 1

    #Set everything else to good
    ml_qcflag[np.where(ml_qcflag == -99)] = 0

    if keep_dims:
        ml_qcflag.reshape(nscans,npixs,nchans)
    
    return ml_qcflag


########################################################################

def load_model_tree():

    '''
    This function loads in all models for channel prediction. Model
    structure and definitions are loaded from model_class.py. Weights
    and biases are loaded in from .pt files located in local models/
    directory.
    '''

    model_dir = f'models'

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

        model_file = glob.glob(f'{model_dir}/{sensor}_{satellite}_channel_predictor_{ichan}_ocean.pt')[0]

        model_tree[f'{model_file.split("/")[-1][:-3]}'] = model_class.channel_predictor(input_size, hidden_size, output_size)
        model_tree[f'{model_file.split("/")[-1][:-3]}'].load_state_dict(torch.load(model_file, weights_only=True))
        model_tree[f'{model_file.split("/")[-1][:-3]}'].eval()

    #Configuration for nonocean model
    input_size = sensor_info.nfeatures - 1 + 1 #(n-1 channels and surface code)

    #Load in nonocean model
    for ichan in feature_names:
        model_file = glob.glob(f'{model_dir}/{sensor}_{satellite}_channel_predictor_{ichan}_nonocean.pt')[0]

        model_tree[f'{model_file.split("/")[-1][:-3]}'] = model_class.channel_predictor(input_size, hidden_size, output_size)
        model_tree[f'{model_file.split("/")[-1][:-3]}'].load_state_dict(torch.load(model_file, weights_only=True))
        model_tree[f'{model_file.split("/")[-1][:-3]}'].eval()


    return model_tree



########################################################################

def run_predictions(tbs, sfccodes, model_tree, keep_dims=True):

    '''
    Runs model on tb array to make predictions.
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

        _, predictors = local_functions.extract_columns(tbs, ichan)

        ocean_model = model_tree[f'{sens}_{sat}_channel_predictor_{channel}_ocean']
        nonocean_model = model_tree[f'{sens}_{sat}_channel_predictor_{channel}_nonocean']

        if np.sum(ocean) > 0:

            x_ocean = predictors[ocean]

            with torch.no_grad():
                tbs_pred[ocean,ichan] = ocean_model(torch.tensor(x_ocean))[:,0]

        if np.sum(nonocean) > 0:

            x_nonocean = predictors[nonocean]
            x_nonocean = np.concatenate((x_nonocean, sfccodes[nonocean][:,None].astype(np.float32)), axis=1)

            with torch.no_grad():
                tbs_pred[nonocean,ichan] = nonocean_model(torch.tensor(x_nonocean))[:,0]

    if keep_dims:
        tbs_pred = tbs_pred.reshape(nscans,npixs,nchans)


    return tbs_pred


########################################################################



