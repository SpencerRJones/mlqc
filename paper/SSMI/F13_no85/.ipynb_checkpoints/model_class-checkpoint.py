from torch import nn


class channel_predictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(channel_predictor, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, output_size)
        self.actvn = nn.GELU()

    def forward(self, x):
        out = self.l1(x)
        out = self.actvn(out)
        out = self.l2(out)
        out = self.actvn(out)
        out = self.l3(out)
        out = self.actvn(out)
        out = self.l4(out)
        
        return out


