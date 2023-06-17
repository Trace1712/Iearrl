import numpy as np
import torch
import json
from main import DDPG
import gym
from itertools import count


def to_np(t):
    return t.cpu().detach().numpy()


class attack_agent:

    def __init__(self, agent, attack_type, config_path):
        self.attack_type = attack_type
        with open(config_path) as f:
            config = json.load(f)
        self.attack_type = attack_type
        self.attack_epsilon = config['attack_params']['eps']
        self.attack_iteration = 10
        self.attack_alpha = config['attack_params']['eps'] / config['attack_params']['iteration']
        self.sarsa_action_ratio = config['attack_params']['sarsa_action_ratio']
        self.state_mean = torch.tensor(config['data_config']['state_mean'], dtype=torch.float32)
        self.state_std = torch.tensor(config['data_config']['state_std'], dtype=torch.float32)
        self.action_mean = torch.tensor(config['data_config']['action_mean'], dtype=torch.float32)
        self.action_std = torch.tensor(config['data_config']['action_std'], dtype=torch.float32)
        self.agent = agent

    def eval_step(self, state):
        if self.attack_type == "action":
            state = self.attack_action(state)
        elif self.attack_type == "random":
            state = self.attack_random(state)
        elif self.attack_type == 'critic':
            state = self.attack_critic(state)
        elif self.attack_type == 'sarsa':
            state = self.attack_critic(state)
        elif self.attack_type == 'sarsa_action':
            state = self.attack_critic_action(state)
        elif self.attack_type == 'no':
            state = state
        return state
    def attack_random(self, state):
        dtype = state.dtype
        state = self.normalize(torch.tensor(state))  # convert to tensor
        noise = np.random.uniform(-self.attack_epsilon, self.attack_epsilon, state.data.shape).astype(dtype)
        state = torch.tensor(noise) + state
        state = self.denormalize(state)
        return to_np(state)

    def attack_action(self, state):
        dtype = state.dtype
        state = torch.tensor(state,dtype=torch.float32)
        gt_action = self.agent.actor(state).clone().detach()
        gt_action = self.action_normalize(gt_action)
        criterion = torch.nn.MSELoss()
        ori_state = self.normalize(state.clone().detach())
        # self.attack_epsilon = 0.1
        # random start ("alpha" is the per-step perturbation size)
        noise = np.random.uniform(-self.attack_alpha, self.attack_alpha, state.data.shape).astype(dtype)

        state = torch.tensor(noise) + ori_state  # normalized
        state = self.denormalize(state)

        for _ in range(self.attack_iteration):
            # state.requires_grad = True
            state = torch.tensor(state,dtype=torch.float32,requires_grad=True)
            action = self.agent.actor(state)
            action = self.action_normalize(action)

            loss = -criterion(action, gt_action)
            self.agent.actor_optimizer.zero_grad()
            loss.backward()
            adv_state = self.normalize(state) - self.attack_alpha * state.grad.sign()
            state = self.denormalize(
                torch.min(torch.max(adv_state, ori_state - self.attack_epsilon), ori_state + self.attack_epsilon))
        return to_np(state)

    def attack_critic(self, state, attack_epsilon=None, attack_iteration=None, attack_stepsize=None):
        # Backward compatibility, use values read in config file
        attack_epsilon = self.attack_epsilon if attack_epsilon is None else attack_epsilon
        attack_stepsize = self.attack_alpha if attack_stepsize is None else attack_stepsize
        attack_iteration = self.attack_iteration if attack_iteration is None else attack_iteration
        dtype = state.dtype
        state = torch.tensor(state, requires_grad=True)  # convert to tensor
        # ori_state = self.normalize(state.data)
        ori_state_tensor = torch.tensor(state.clone().detach(),dtype=torch.float32)
        ori_state = self.normalize(state.clone().detach())
        # random start
        noise = np.random.uniform(-attack_stepsize, attack_stepsize, state.data.shape).astype(dtype)

        state = torch.tensor(state,dtype=torch.float32) + ori_state  # normalized
        state = self.denormalize(state)

        # self.attack_epsilon = 0.1
        state_ub = ori_state + attack_epsilon
        state_lb = ori_state - attack_epsilon
        for _ in range(attack_iteration):
            state = torch.tensor(state,dtype=torch.float32, requires_grad=True)
            action = self.agent.actor(state)
            qval = self.agent.critic.forward2(ori_state_tensor, action)
            loss = torch.mean(qval)
            loss.backward()
            adv_state = self.normalize(state) - attack_stepsize * state.grad.sign()
            # adv_state = self.normalize(state) + 0.01 * state.grad.sign()
            state = self.denormalize(torch.min(torch.max(adv_state, state_lb), state_ub))
            # state =  torch.max(torch.min(adv_state, self.state_max), self.state_min)
        self.agent.critic_optimizer.zero_grad()
        self.agent.actor_optimizer.zero_grad()

        return to_np(state)

    def attack_critic_action(self, state):
        dtype = state.dtype
        state = torch.tensor(state,dtype=torch.float32)
        ori_state_tensor = torch.tensor(state.clone().detach())
        gt_action = self.agent.actor(state).clone().detach()
        gt_action = self.action_normalize(gt_action)

        criterion = torch.nn.MSELoss()
        ori_state = self.normalize(state.clone().detach())
        # random start
        noise = np.random.uniform(-self.attack_alpha, self.attack_alpha, state.data.shape).astype(dtype)

        state = torch.tensor(noise) + ori_state  # normalized
        state = self.denormalize(state)
        for _ in range(self.attack_iteration):
            # state.requires_grad = True

            state = torch.tensor(state, requires_grad=True,dtype=torch.float32)
            action = self.agent.actor(state)
            # mse loss
            qval = self.agent.critic.forward2(ori_state_tensor, action)
            loss1 = torch.mean(qval)

            loss2 = -self.sarsa_action_ratio * criterion(self.action_normalize(action), gt_action)
            if self.sarsa_action_ratio != 1:
                loss = (1 - self.sarsa_action_ratio) * loss1 + loss2
            else:
                loss = loss1 + loss2

            self.agent.actor.zero_grad()
            self.agent.critic.zero_grad()
            loss.backward()
            adv_state = self.normalize(state) - self.attack_alpha * state.grad.sign()
            state = self.denormalize(
                torch.min(torch.max(adv_state, ori_state - self.attack_epsilon), ori_state + self.attack_epsilon))
        return to_np(state)

    # Normalization are currently only used for attack.
    def normalize(self, state):
        state = (state - self.state_mean) / self.state_std
        return state

    def denormalize(self, state):
        state = state * self.state_std + self.state_mean
        return state

    def action_normalize(self, action):
        action = action / self.action_std
        return action

    def action_denormalize(self, action):
        action = action * self.action_std
        return action

