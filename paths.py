'''
Set paths to data and model files
'''
from pathlib import Path

root = Path(__file__).resolve().parents[0]

l1c_datapath = root / "data/l1c"
sfc_datapath = root / "data/sfctype"
training_datapath = root / "data/training_data"
model_path   = root / "models"

