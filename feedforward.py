import torch
import torch.nn as nn
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.first_layer = nn.Linear(d_model, 4*d_model)
        self.second_layer = nn.Linear(4*d_model, d_model)
    def forward(self, x):
        first_output = self.first_layer(x)
        activaited = torch.relu(first_output)
        second_output = self.second_layer(activaited)
        return second_output

