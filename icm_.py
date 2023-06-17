from model import *
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import gym
import os


class Memory_Buffer(object):
    def __init__(self, memory_size=1000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size:  # buffer not full
            self.buffer.append(data)
        else:  # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def size(self):
        return len(self.buffer)


class ICM_DQNAgent:
    def __init__(self, in_channels=1, action_dim=1, device=None, memory_size=10000, epsilon=1, lr=1e-4,
                 forward_scale=0.8, inverse_scale=0.2, Qloss_scale=0.1, intrinsic_scale=1, use_extrinsic=True,
                 env_name=None,load=False):
        self.epsilon = epsilon
        self.action_dim = action_dim
        # param for ICM
        self.forward_scale = forward_scale  # scale for loss function of forward prediction arpl_main, 0.8
        self.inverse_scale = inverse_scale  # scale for loss function of inverse prediction arpl_main, 0.2
        self.Qloss_scale = Qloss_scale  # scale for loss function of Q value, 1
        self.intrinsic_scale = intrinsic_scale  # scale for intrinsic reward, 1
        self.use_extrinsic = use_extrinsic  # whether use extrinsic rewards, if False, only intrinsic reward generated from ICM is used

        self.memory_buffer = Memory_Buffer(memory_size)
        self.icm_critic = icm_critic(in_channels, action_dim).to(device)
        if load:
            print("load dqn")
            self.icm_critic.load_state_dict(torch.load("./model/"+env_name+"/dqn.pth"))
        self.icm_critic_target = icm_critic(in_channels, action_dim)
        self.icm_critic_target.load_state_dict(self.icm_critic.state_dict())
        self.icm_actor = icm_actor(input_shape=in_channels, num_actions=action_dim).to(device)
        if load:
            print("load icm")
            self.icm_actor.load_state_dict(torch.load("./model/"+env_name+"/icm.pth"))
        self.device = device
        self.icm_actor = self.icm_actor.to(device)
        self.optimizer = optim.Adam(list(self.icm_critic.parameters()) + list(self.icm_actor.parameters()), lr=lr)

        self.update_tar_interval = 1000
        self.learning_start = 1000
        self.certin_model = Calcuate_net(in_channels, action_dim).to(device)
        self.certin_model.load_state_dict(torch.load(os.path.join("uncertain", env_name, "uncertain.pth")))

    def compute_td_loss(self, states, actions, rewards, next_states, is_done, action_plus, gamma=0.99):
        """ Compute td loss using torch operations only. Use the formula above. """
        # actions = torch.tensor(actions).long()  # shape: [batch_size]
        # action = Categorical(actions).sample()
        rewards = rewards.clone().detach().requires_grad_(True)
        # rewards = torch.tensor(rewards, dtype=torch.float)  # shape: [batch_size]
        is_done = torch.tensor(is_done).type(torch.bool)  # shape: [batch_size]

        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        is_done = is_done.to(self.device)
        states = states.to(self.device)
        next_states = next_states.to(self.device)

        # get q-values for all actions in current states
        # states = torch.from_numpy(np.array(states))
        predicted_qvalues = self.icm_critic.forward(states)

        # get ICM results
        # a_vec = F.one_hot(actions, num_classes=self.action_dim)  # convert action from int to one-hot format
        pred_s_next, pred_a_vec, feature_x_next = self.icm_actor.get_full(states, next_states, actions)
        # calculate forward prediction and inverse prediction loss
        forward_loss = F.mse_loss(pred_s_next, feature_x_next.detach(), reduction='none')
        # pred_a_vec = pred_a_vec.flatten()
        # actions = actions.flatten()
        inverse_pred_loss = F.mse_loss(pred_a_vec, actions.detach(), reduction='none')

        # calculate rewards
        intrinsic_rewards = self.intrinsic_scale * forward_loss.mean(-1)
        # intrinsic_rewards = self.intrinsic_scale * torch.exp(forward_loss.mean(-1))
        output = torch.tensor(self.certin_model.get_num(states, actions),dtype=torch.float).data.cpu().numpy()[0]
        # output = 1
        total_rewards = torch.tensor(intrinsic_rewards.data.cpu().numpy() / output,dtype=torch.float).to(device)
        if self.use_extrinsic:
            total_rewards += rewards

        # temp = states.shape[0]
        # # select q-values for chosen actions
        # action_plus = torch.tensor(action_plus).long()
        # predicted_qvalues_for_actions = predicted_qvalues[
        #     range(temp), action_plus
        # ]
        predicted_qvalues_for_actions = torch.mul(predicted_qvalues, actions)
        predicted_qvalues_for_actions = torch.mean(predicted_qvalues_for_actions, dim=1)

        # compute q-values for all actions in next states
        next_states = torch.tensor(next_states, dtype=torch.float32)
        predicted_next_qvalues = self.icm_critic.forward(next_states)  # YOUR CODE

        # compute V*(next_states) using predicted next q-values
        next_state_values = predicted_next_qvalues.max(-1)[0]  # YOUR CODE

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_qvalues_for_actions = total_rewards + gamma * next_state_values  # YOUR CODE

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done, total_rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        # loss = torch.mean((predicted_qvalues_for_actions -
        #                   target_qvalues_for_actions.detach()) ** 2)
        Q_loss = F.smooth_l1_loss(predicted_qvalues_for_actions, target_qvalues_for_actions.detach())
        loss = self.Qloss_scale * Q_loss + self.forward_scale * forward_loss.mean() + self.inverse_scale * inverse_pred_loss.mean()
        loss = torch.tensor(loss, dtype=torch.float, requires_grad=True)
        return loss, Q_loss.item(), forward_loss.mean().item(), inverse_pred_loss.mean().item(), intrinsic_rewards.mean().item()

    def sample_from_buffer(self, batch_size):
        states, actions, rewards, next_states, dones, action_plus = [], [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.memory_buffer.size() - 1)
            data = self.memory_buffer.buffer[idx]
            frame, action, reward, next_frame, done = data
            states.append(torch.tensor(frame).unsqueeze(0))
            # states.append(frame.clone().detach().requires_grad_(True))
            actions.append(torch.tensor(action).unsqueeze(0))
            action_plus.append(action.argmax().item())
            rewards.append(reward)
            next_states.append(torch.tensor(next_frame).unsqueeze(0))
            dones.append(done)
        # states = torch.tensor(states.numpy())
        # next_states = torch.tensor(next_states)
        return torch.cat(states), torch.cat(actions), rewards, torch.cat(next_states, 0), dones, action_plus

    def learn_from_experience(self, batch_size):
        if self.memory_buffer.size() > batch_size:
            states, actions, rewards_, next_states, dones, action_plus = self.sample_from_buffer(batch_size)
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(rewards_), reversed(dones)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (0.99 * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Normalizing the rewards:
            rewards = torch.tensor(rewards, dtype=torch.float32)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            td_loss, Q_loss, forward_loss, inverse_pred_loss, intrinsic_rewards = self.compute_td_loss(states, actions,
                                                                                                       rewards,
                                                                                                       next_states,
                                                                                                       dones,
                                                                                                       action_plus)
            self.optimizer.zero_grad()
            td_loss.backward()
            # for param in list(self.DQN.parameters()) + list(self.ICM.parameters()):
            #     param.grad.data.clamp_(-1, 1)
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

    def update(self, i, dir):

        if self.memory_buffer.size() >= self.learning_start:
            self.learn_from_experience(128)
        if i % self.update_tar_interval == 0:
            torch.save(self.icm_actor.state_dict(), dir + 'icm.pth')
            torch.save(self.icm_critic.state_dict(), dir + 'dqn.pth')

    def select_action(self, state_tensor, action, reward, next_state, done):
        state_tensor = torch.tensor(state_tensor, dtype=torch.float)
        self.memory_buffer.push(state_tensor, action, reward, next_state, done)
        action = self.icm_critic.forward(state_tensor)
        return action

    def load(self, dir):
        self.icm_actor.load_state_dict(torch.load(dir + 'icm.pth'))
        self.icm_critic.load_state_dict(torch.load(dir + 'dqn.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")
