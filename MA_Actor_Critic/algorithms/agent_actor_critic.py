
import torch.nn.functional as F
import torch
import torch.nn as nn
from algorithms.actor import Actor
from algorithms.critic import Critic
import numpy as np
import copy

import math

class Agent(nn.Module):
    def __init__(self, args):
        super(Agent, self).__init__()
        self.device = args.dvc
        self.num_channel = args.num_channel
        self.allowed_channel_num = args.allowed_channel_num

        self.actor_input_dim = int(args.obs_dim * args.obs_his_len)
        self.actor_net_width = args.actor_net_width
        self.actor_output_dim = int(math.comb(self.num_channel, self.allowed_channel_num))
        self.actor = Actor(self.actor_input_dim, self.actor_net_width, self.actor_output_dim)
        self.actor_lr = args.actor_lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_lr)

        self.critic_input_dim = self.actor_input_dim
        self.critic_net_width = args.critic_net_width
        self.critic_output_dim = 1
        self.critic = Critic(self.critic_input_dim, self.critic_net_width, self.critic_output_dim)
        self.critic_lr = args.critic_lr
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_lr)

        self.epsilon = args.epsilon   # exploration probability

        #self.tpdv = dict(dtype=torch.float32, device=device)


    def action_mapping(self, action): # mapping from 0-output_dim-1 to Kx1 vector where v(k)=1 indicates channel k is selected
        ones_pos = []
        cur_pos = 0
        for remaining_ones in range(self.allowed_channel_num, 0, -1):
            while True:
                block_size = math.comb(self.num_channel-cur_pos-1, remaining_ones - 1)
                if action < block_size:
                    ones_pos.append(cur_pos)
                    cur_pos += 1
                    break
                else:
                    action -= block_size
                    cur_pos += 1

        action = [0] * self.num_channel
        for pos in ones_pos:
            action[pos] = 1
        return action

    def select_action(self, obs, deterministic=False):
        #obs = check(obs).to(**self.tpdv)

        # obs = pre_action + [1 for _ in range(int((self.obs_dim-2) / 2))] + pre_feedback

        obs = torch.tensor(np.array(obs), dtype=torch.float32).view(1, 1, self.actor_input_dim)

        pros = self.actor(obs)
        m = torch.distributions.Categorical(pros.squeeze())

        if deterministic:
            action = pros.argmax(dim=-1)
            return m.log_prob(action), self.action_mapping(action.item())
        else:
            if np.random.rand() < self.epsilon:
                action = torch.randint(0, self.actor_output_dim, (1, 1, 1))
                return m.log_prob(action), self.action_mapping(action.item())
            else:
                action = m.sample()
                return m.log_prob(action), self.action_mapping(action.item())


    def eval_action(self, obs):
        obs = torch.tensor(np.array(obs), dtype=torch.float32).view(1, 1, self.actor_input_dim)
        return self.critic(obs)

    def critic_train(self, td_error):
        critic_loss = td_error.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def actor_train(self, actor_loss):
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()



    def save(self, save_dir):
        torch.save(self.actor.state_dict(), save_dir)

    def load(self, load_dir):
        self.actor.load_state_dict(torch.load(load_dir, weights_only=True,  map_location=self.device))
        self.actor.eval()

