# Machine Learning Quality Control (MLQC) algorithm for passive microwave sensors.
This repository contains the example code for training and running the MLQC machine learning models for GMI.

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
jupyter-notebook
```
and click on the file.

### General Notes on Code Structure:
The code is structured
