from torch.distributions import Categorical
import gym  # pip install box2d box2d-kengz --user
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class icm_critic(nn.Module):
    def __init__(self, input_shape, num_action):
        super(icm_critic, self).__init__()
        self.layer1 = nn.Linear(input_shape, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, num_action)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.softmax(x)


class icm_actor(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(icm_actor, self).__init__()
        self.layer1 = nn.Linear(input_shape, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 64)
        # self.layer4 = nn.Linear

        self.softmax = nn.Softmax(dim=-1)
        self.fc4 = nn.Linear(64, 512)

        self.pred_module1 = nn.Linear(input_shape + num_actions, 256)
        self.pred_module2 = nn.Linear(256, input_shape)

        self.invpred_module1 = nn.Linear(input_shape + input_shape, 256)
        self.invpred_module2 = nn.Linear(256, num_actions)

    # # def get_feature(self, x):
    # #     x = F.relu(self.layer1(x))
    # #     x = F.relu(self.layer2(x))
    # #     x = F.relu(self.layer3(x))
    # #     x = F.relu(self.fc4(x.view(x.size(0), -1)))
    # #     return self.softmax(x)
    # def get_feature(self, x):
    #     x = x.cpu().detach().numpy()
    #     noise = np.random.uniform(x.min(), x.max(), size=x.shape)
    #     x = x + noise
    #     return torch.tensor(x,dtype=torch.float)
    # # def get_feature_plus(self,x):

    def forward(self, x):
        feature_x = self.get_feature(x)
        return feature_x

    def get_full(self, x, x_net, a_vec):
        x_net = x_net.clone().detach().requires_grad_(True)
        pred_s_next = self.pred(x, a_vec)  # predict next state feature
        pred_a_vec = self.invpred(x, x_net)  # (inverse) predict action

        return pred_s_next, pred_a_vec, x_net

    def pred(self, feature_x, a_vec):
        temp = [feature_x, a_vec]
        temp = torch.tensor(torch.cat(temp, dim=1), dtype=torch.float).detach()
        pred_s_next = F.relu(self.pred_module1(temp))
        pred_s_next = self.pred_module2(pred_s_next)
        return pred_s_next

    def invpred(self, feature_x, feature_x_next):
        # Inverse prediction: predict action (one-hot), given current and next state features
        temp = [feature_x, feature_x_next]
        temp = torch.cat(temp, dim=-1)
        temp = self.invpred_module1(torch.tensor(temp, dtype=torch.float))
        pred_a_vec = F.relu(temp)
        pred_a_vec = self.invpred_module2(pred_a_vec)
        return F.softmax(pred_a_vec, dim=-1)


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
        temp = torch.tensor(torch.cat(temp, dim=-1),dtype=torch.float)
        return self.forward(temp)
