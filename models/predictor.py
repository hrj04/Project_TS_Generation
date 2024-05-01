import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, 
                 input_dim = 5, 
                 hidden_dim = 50, 
                 output_dim = 1, 
                 num_layers = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)  
        out = self.linear(out[:, -1, :])
        
        return out


