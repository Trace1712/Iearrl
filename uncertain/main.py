import gym
from tqdm import tqdm
# from agent import *
import argparse
from itertools import count
from calculationModel.network import *
import torch.optim as optim
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from buffer import *

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", default="HalfCheetah-v2")
parser.add_argument('--seed', default=False, type=bool)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
directory = './' + args.env_name + '/'


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


class calculation:
    def __init__(self, state_dim, action_dim, lr=1e-4):
        self.calucate = Calcuate_net(state_dim, action_dim)
        self.optimizer = optim.Adam(list(self.calucate.parameters()), lr=lr, weight_decay=1e-2)
        self.buffer = Replay_buffer()

    def update(self):
        # state = torch.tensor(state,dtype=torch.float).unsqueeze(0)
        # action = torch.tensor(action, dtype=torch.float).unsqueeze(0)
        x, y = self.buffer.sample(256)
        state = torch.FloatTensor(x).to(device)
        action = torch.FloatTensor(y).to(device)
        output = self.calucate.get_num(state, action)
        l2_loss = Variable(torch.full(output.shape, 1, dtype=torch.float), requires_grad=True)
        loss = F.mse_loss(output, l2_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def push(self, state, action):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        self.buffer.push((state, action))


def main():
    # plt.ion()

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    loss_lst = []
    epsilon_lst = []
    agent = DDPG(state_dim, action_dim, max_action)
    # agent.load()
    agent.actor.load_state_dict(torch.load(directory + 'actor.pth', map_location=torch.device('cpu')))
    calcuate = calculation(state_dim, action_dim)
    for i in range(10000):
        state = env.reset()
        for t in count():
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            calcuate.push(state, action)
            if done:
                break
        loss = calcuate.update()
        print(loss)
        loss_lst.append(loss)
        epsilon_lst.append(i)
        if i % 10 == 0:
            plt.xlabel("Iterations")
            plt.ylabel("loss")
            plt.plot(epsilon_lst, loss_lst, color='blue', linewidth=2)
            plt.savefig(directory + 'test.jpg')
            torch.save(calcuate.calucate.state_dict(), directory + 'uncertain.pth')
            plt.clf()


if __name__ == '__main__':
    main()
