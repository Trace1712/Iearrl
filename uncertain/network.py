import torch.nn as nn
import torch.nn.functional as F
import torch

class Calcuate_net(nn.Module):
    def __init__(self, input_shape, num_action):
        super(Calcuate_net, self).__init__()
        self.layer1 = nn.Linear(input_shape + num_action, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return x

    def get_num(self, state, action):
        temp = [state, action]
        return self.forward(torch.cat(temp, dim=-1))
