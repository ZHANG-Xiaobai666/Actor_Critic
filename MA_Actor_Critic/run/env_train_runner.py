
import time
import numpy as np
import torch
from algorithms.agent_actor_critic import Agent
import os

class EnvRunner():
    def __init__(self, args, env):
        #self.max_train_times = args.max_train_times
        self.episodes = args.episode_num
        self.episode_length = args.episode_length
        #self.reward_type = args.reward_type
        self.env = env

        self.num_agent = args.num_agent
        self.num_channel = args.num_channel
        self.gamma  = args.gamma

        self.save_dir = args.save_dir

        self.agents = []
        for _ in range(self.num_agent):
            agent = Agent(args)
            self.agents.append(agent)

        self.td_errors = []


    def run(self):

        start = time.time()

        for episode in range(self.episodes):

            self.env.reset()
            obs = self.env.get_obs_his()

            for step in range(self.episode_length):

                log_pros, actions = self.collect(obs)             # collect actions and corresponding probs
                obs_next = self.env.step(actions, step)

                self.cal_td_error(obs, obs_next)
                self.train_critic()
                self.train_actor(log_pros)
                obs = obs_next

                if step % 100 == 0:
                    print(f"Iteration: {step+1} / {self.episode_length}")
                    print(f"Throughput {self.env.get_short_term_throughput(step)}")

            self.save_model()
            #.update_par(step, self.episode_length)


        #print(f"Iteration: {train_time+1} / {self.max_train_times}")
        #print(f"Throughput {self.env.get_throughput()}")


        end = time.time()
    def collect(self, obs):
        actions = []
        probs = []
        for agent in range(int(self.num_agent)):
            prob, action = self.agents[agent].select_action(obs[agent])
            actions.append(action)
            probs.append(prob)
        return probs, actions
    def cal_td_error(self, obs, obs_next):
        self.td_errors = []
        for agent in range(self.num_agent):
            v_s = self.agents[agent].eval_action(obs[agent])
            v_s_next = self.agents[agent].eval_action(obs_next[agent]).detach()
            td_error = sum(obs_next[agent][0]) + self.gamma*v_s_next - v_s
            self.td_errors.append(td_error)
    def train_critic(self):
        for agent in range(self.num_agent):
            self.agents[agent].critic_train(self.td_errors[agent])


    def train_actor(self, log_pros):
        for agent in range(self.num_agent):
            td_error = self.td_errors[agent].detach()
            self.agents[agent].actor_train(-log_pros[agent] * td_error)



    def save_model(self):
        for agent in range(self.num_agent):
            self.agents[agent].save(os.path.join(self.save_dir, "agent" + str(agent) + ".pt"))

    def load_model(self):
        for agent in range(self.num_agent):
            self.agents[agent].load(os.path.join(self.save_dir, "agent" + str(agent) + ".pt"))

    def update_par(self, episode, episodes):
        for agent in range(self.num_agent):
            self.agents[agent].lr_decay(episode, episodes)
            self.agents[agent].update_alpha(episode, episodes)
            self.agents[agent].update_beta(episode, episodes)

    def lstm_state_reset(self):
        for agent in range(self.num_agent):
            self.agents[agent].lstm_ret()