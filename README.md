# Machine Learning Quality Control (MLQC) algorithm for passive microwave sensors.
This repository contains the example code for training and running the MLQC machine learning models for GMI.

If this code has been helpful to your research, I would kindly ask that you please cite <...>.

## Getting Started
These instructions will walk you through installing the repository 
and creating and activating the environment provided for you in the environment.yml file.
Assumes you have conda installed. If you would rather use pip, simply install the dependencies
listed in the environment.yml file.

1. Clone the repository.
```
git clone https://github.com/SpencerRJones/mlqc
```
2. Create the conda environment and install dependencies.
```
cd mlqc
conda env create -f environment.yml
```
3. Activate the environment.
```
conda activate mlqc
```
4. Install the package for import links.
```
pip install -e .
```

## Running the pretrained model
A pretrained MLQC package for GMI is provided for you in the main repository. To run it, simply navigate to the `mlqc/run/`
directory and you will find a Jupyter Notebook titled `test_mlqc_GMI_pretrained.ipynb`. It will show you how to read the
test GMI Level 1C file, load the model tree (a Python dictionary of torch.nn models), and run the neural networks to make
predictions. To run it,
```
jupyter notebook run/test_mlqc_GMI_pretrained.ipynb
```

### Training Your Own MLQC Model:
The code is structured to be (hopefully) intuitive. The `data/` folder contains:
```
data
├── l1c
│   └── 1C-R.GPM.GMI.XCAL2016-C.20220101-S002129-E015402.044559.V07A.HDF5
├── sfctype
│   └── GMI_surfmap_2201_V7.nc
└── training_data
```
with the `l1c/` subdirectory containing the Level 1C file for testing the pretrained MLQC algroithm. The `sfctype/` directory
contains the surface map for the test file day (2022-01-01). `training_data/` is empty, but it is the destination for the
output netcdf file created by `create_training_data_GMI.py`.

The `training` directory:
```
training
├── create_training_data_GMI.py
├── diagnostics
└── train_GMI.ipynb
```
contains a couple of helpful scripts. `create_training_data_GMI.py` is a simple example script that extracts good quality data 
from the Level 1C data format and attaches surface types to the pixel geolocations. It can be run on the provided test file by:
```
python create_training_data_GMI.py
```
But this is simply for illustrative purposes, since a single orbit is obviously not enough training data to produce good results.

The `train_GMI.ipynb` notebook is an example of carrying out the training loops, contained in `utils/training_funcs.py`.

`src/` contains the functions and classes necessary for training running the model.
```
src
├── classes
│   ├── dataset_class.py
│   ├── __init__.py
│   ├── model_class.py
├── __init__.py
├── model_operations.py
├── sensor_info.py
├── surface.py
├── training_funcs.py
└── utils
    ├── array_funcs.py
    ├── data2xarray.py
    ├── extract_channel.py
    ├── __init__.py
    ├── L1C.py
```




