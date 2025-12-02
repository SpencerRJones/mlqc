#!/home/spencer/.conda/envs/base2/bin/python


import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import glob
import torch
from torch import nn
import cartopy.crs as ccrs
from util_funcs.L1C import scantime2datetime
from util_funcs import data2xarray, array_funcs
import geography
from tqdm import tqdm

from dataset_class import dataset
from model_class import channel_predictor
import local_functions
import sensor_info


#################################################################
scan_position = 0
torch.set_num_threads(10)
#################################################################


#General parameters:

satellite = sensor_info.satellite
sensor = sensor_info.sensor

nchans = sensor_info.nchannels
batch_size = 1000
input_size = nchans - 1
hidden_size = 256
output_size = 1


'''
Function Definitions:
'''

chan_desc = sensor_info.channel_descriptions


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


def train_model(model, nepochs, dataloader, learning_rate=0.001, quiet=False, stage=None, validation_dataloader=None):

    nbatches = len(dataloader)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=nepochs)
    device = None

    if stage:
        print(f'Training stage: {stage}')

    loss_arr    = np.zeros([nbatches,nepochs], dtype='f')
    valloss_arr = np.zeros([nbatches,nepochs], dtype='f')
    
    for epoch in range(nepochs):
        for i, (profs, obs) in enumerate(dataloader):
            if device:
                profs, obs = profs.to(device), obs.to(device)

            if validation_dataloader and i%1000==0:
                valprofs, valobs = next(enumerate(validation_dataloader))[1]
                val_pred = model(valprofs)
                valloss  = criterion(val_pred, valobs)
                print(f'Validation Loss = {valloss.item():.3f}')

            #Forward pass:
            obs_pred = model(profs)
            loss     = criterion(obs_pred, obs)


            #Backward pass:
            optimizer.zero_grad()
            loss.backward()

            #Update neurons:
            optimizer.step()

            loss_arr[i,epoch] = loss.item()
            valloss_arr[i,epoch] = valloss.item()
            
            if not quiet:
                if i%1000 == 0:
                    print(f'Epoch={epoch+1}, batch = {i} of {nbatches}, loss={loss.item():.3f}, LR={scheduler.get_last_lr()[0]}')
        
        scheduler.step()


    return loss_arr, valloss_arr

'''

SSMI CHANNEL PREDICTION MODEL:
    1: Ocean

'''

sfc = [1]

with xr.open_dataset(f'training_data/{satellite}_training_data.nc') as f:
    
    sfctype = f.sfctype.values

    correct_sfc = np.isin(sfctype, sfc)
    sfcindcs = np.where(correct_sfc)[0]

    Tbs = f.Tbs.values[sfcindcs]
    scanpos = f.scanpos.values[sfcindcs]
    eia = f.eia.values[sfcindcs]

correct_scanpos = scanpos == scan_position

Tbs = Tbs[correct_scanpos]
eia = eia[correct_scanpos]

# if Tbs.shape[0] > 5.0e+06:
#    Tbs = Tbs[:5_000_000,:]

print(f'Training data shape: {Tbs.shape}')

#---Split data into train/test/val:
train_indcs, test_indcs, val_indcs = local_functions.split_data_indcs(Tbs)

Tbs_train = Tbs[train_indcs]
Tbs_test  = Tbs[test_indcs]
Tbs_val   = Tbs[val_indcs]

#---Shuffle before converting to tensors:
np.random.seed(40)
Tbs_train = array_funcs.shuffle_data(Tbs_train, axis=0)


'''
Predict channels: Train all models
'''



for ichan, channel in enumerate(chan_desc):
    
    print(channel)

    #---Extract channel, x = predictors, y = channel to predict
    x_train, y_train = extract_channel(Tbs_train, channel)
    x_test,  y_test  = extract_channel(Tbs_test, channel)
    x_val,   y_val   = extract_channel(Tbs_val, channel)
    
    x_train, y_train = torch.tensor(x_train), torch.tensor(y_train)
    x_test,  y_test  = torch.tensor(x_test), torch.tensor(y_test)
    x_val,   y_val   = torch.tensor(x_val), torch.tensor(y_val)


    #---Set up dataloaders:
    train_loader = torch.utils.data.DataLoader(dataset=dataset(x_train,y_train), 
                                               batch_size=batch_size, 
                                               shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset(x_test,y_test), 
                                               batch_size=None, 
                                               shuffle=False, drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset=dataset(x_val,y_val), 
                                               batch_size=None, 
                                               shuffle=False, drop_last=False)

    #---Create model:
    model = channel_predictor(input_size, hidden_size, output_size)

    #---Train_model:
    nbatches = len(train_loader)
    nepochs_stage1 = 10
    nepochs_stage2 = 20
    nepochs_stage3 = 40

    loss_stage1, valloss_stage1 = train_model(model, nepochs=nepochs_stage1, dataloader=train_loader, 
                                          learning_rate=0.01, quiet=False, stage=1, validation_dataloader=val_loader)
    loss_stage2, valloss_stage2 = train_model(model, nepochs=nepochs_stage2, dataloader=train_loader, 
                                          learning_rate=0.001, quiet=False, stage=2, validation_dataloader=val_loader)
    loss_stage3, valloss_stage3 = train_model(model, nepochs=nepochs_stage3, dataloader=train_loader, 
                                          learning_rate=0.001, quiet=False, stage=3, validation_dataloader=val_loader)

    torch.save(model.state_dict(), f'models/{sensor}_{satellite}_channel_predictor_{channel}_scanpos{scan_position}_ocean.pt')

    loss_data = data2xarray(data_vars = (loss_stage1, loss_stage2, loss_stage3, 
                                     valloss_stage1, valloss_stage2, valloss_stage3),
                        var_names = ('LossStage1','LossStage2','LossStage3',
                                     'ValidationLossStage1', 'ValidationLossStage2', 'ValidationLossStage3'),
                        dims = (nbatches,nepochs_stage1,nepochs_stage2, nepochs_stage3),
                        dim_names = ('training_batches', 'epochs_stage1', 'epochs_stage2', 'epochs_stage3'))

    loss_data.to_netcdf(f'diagnostics/loss_data_{channel}_scanpos{scan_position}ocean.nc', engine='netcdf4')

    print(f'Finished training model for channel {channel}.')