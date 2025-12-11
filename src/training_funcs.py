import numpy as np
import torch
from torch import nn


########################################################################

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


def shuffle_data(data_array, axis, return_indcs=False):

    '''

    Shuffles data along the specified axis.

    Inputs:
        data_array | ndarray of data
        axis       | axis along which to shuffle

    Outputs:
        shuffled   | ndarray of shuffled data

    '''

    n = data_array.shape[axis]

    indcs = np.arange(0,n)
    np.random.shuffle(indcs)

    if return_indcs:
        return data_array[indcs], indcs

    return data_array[indcs]
