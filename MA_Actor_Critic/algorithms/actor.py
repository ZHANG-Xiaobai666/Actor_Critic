import torch
import torch.nn as nn



class Actor(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(Actor, self).__init__()
        self.f1 = nn.Linear(input_dim, hid_dim)
        self.f2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.f1(x)
        x = torch.relu(x)
        x = self.f2(x)
        x = torch.softmax(x, dim=-1)
        return x
