import argparse
from itertools import count

import os
from icm_ import *
from tqdm import tqdm
'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str)  # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="Walker2d-v2")
parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int)  # discounted factor
parser.add_argument('--capacity', default=1000000, type=int)  # replay buffer size
parser.add_argument('--batch_size', default=100, type=int)  # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool)  # show UI or not
parser.add_argument('--log_interval', default=50, type=int)  #
parser.add_argument('--load', default=False, type=bool)  # load model
parser.add_argument('--render_interval', default=100, type=int)  # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=50000, type=int)  # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
parser.add_argument('--parameter', default=0.1, type=float)
args = parser.parse_args()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = gym.make(args.env_name)

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device)  # min value

directory = './modelnew/' + args.env_name + '/' + str(args.parameter * 10) + '/'
result_directory = "./results/" + args.env_name + '/' + str(args.parameter * 10) + '/'
os.makedirs(result_directory, exist_ok=True)
os.makedirs(directory, exist_ok=True)


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''

    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

    def forward2(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], -1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, load=True):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        if load:
            print("load actor")
            self.actor.load_state_dict(torch.load("./model/" + args.env_name + "/actor.pth"))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic = Critic(state_dim, action_dim).to(device)
        if load:
            print("load critic")
            self.critic.load_state_dict(torch.load("./model/" + args.env_name + "/critic.pth"))
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self, directory):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth', map_location=torch.device('cpu')))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth', map_location=torch.device('cpu')))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def main():
    agent = DDPG(state_dim, action_dim, max_action, load=False)

    Qloss_scale = 1  # scale for loss function of Q value, 1
    intrinsic_scale = 100  # scale for intrinsic reward, 1
    use_extrinsic = True  # whether use extrinsic rewards, if False, only intrinsic reward generated from ICM is used
    icm_agent = ICM_DQNAgent(in_channels=state_dim, action_dim=action_dim, device=device, lr=0.005,
                             forward_scale=0.8, inverse_scale=0.2, Qloss_scale=Qloss_scale,
                             intrinsic_scale=intrinsic_scale,
                             use_extrinsic=use_extrinsic, env_name=args.env_name, load=False)
    ep_r = 0
    print(args.mode)
    if args.mode == 'test':
        agent.load(directory)
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                if done or t >= args.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif args.mode == 'train':

        total_step = 0
        rewards_lst = []
        import pandas as pd
        import numpy as np
        import time as time
        col_name = ["episode", "step", "time", "reward"]
        f = pd.DataFrame(columns=col_name, data=None)
        f.to_csv("{}".format(args.env_name + ".csv"), index=False)
        for i in tqdm(range(args.max_episode)):
            total_reward = 0
            step = 0
            state = env.reset()
            adv_action = None
            start_time = time.time()
            for t in count():
                action = agent.select_action(state)
                if adv_action is not None:
                    action = action + args.parameter * adv_action

                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

                next_state, reward, done, info = env.step(action)
                if args.render and i >= args.render_interval: env.render()
                agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))

                adv_action = icm_agent.select_action(state_tensor=state, action=action, reward=reward,
                                                     next_state=next_state, done=done).detach().cpu().numpy()
                state = next_state

                if done:
                    rewards_lst.append(total_reward)
                    break

                step += 1
                total_reward += reward
            end_time = time.time()
            info = pd.read_csv("{}".format(args.env_name + ".csv"))
            num = info["reward"].tolist()
            info.loc[len(num)] = [i, int(step), end_time - start_time, total_reward]
            info.to_csv("{}".format(args.env_name + ".csv"), index=False)

            total_step += step + 1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            agent.update()
            icm_agent.update(i=i, dir=directory)

            # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            if i % args.log_interval == 0:
                agent.save()
                np.save(os.path.join(result_directory, args.env_name + ".npy"), rewards_lst)
    else:
        raise NameError("mode wrong!!!")


if __name__ == '__main__':
    main()
