import torch

class dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.predictors = x
        self.predictands = y

    def __len__(self):
        return len(self.predictors)

    def __getitem__(self,idx):
        return self.predictors[idx], self.predictands[idx]