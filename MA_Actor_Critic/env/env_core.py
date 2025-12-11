import random
import numpy as np

class EnvCore:

    def __init__(self, args, arr_pro, state_trans_pro):
        self.episode_length = args.episode_length # Number of time slot per train/execute
        self.num_agent = args.num_agent       # Number of nodes
        self.arr_pro = arr_pro              # aggregate arrival probability
        self.state_trans_pro = state_trans_pro # the probability of channel state transition
        self.num_channel = args.num_channel   # Number of channels

        self.channel_state = [1 for _ in range(self.num_channel)]    #  all channels are good
        self.channel_feedback = [0 for _ in range(self.num_channel)]  # 1 if ACK 0 Otherwise

        self.queue_length = [0 for _ in range(self.num_agent)]   # backlogged packets
        self.generate_time = [[] for _ in range(self.num_agent)] # For derivation of delay

        self.num_successful_packets = [0 for i in range(self.num_agent)] # successful transmitted packets
        #self.mean_delay = [0 for _ in range(self.num_agent)]             # mean queueing delay of data packets
        self.throughput = 0                                             # network throughput sum(
                                                                        # num_successful_packets)/episode_length)
        self.obs_his_len = args.obs_his_len
        self.obs_obs_dim = args.obs_dim
        self.obs = [[np.zeros(self.obs_obs_dim) for _ in range(self.obs_his_len)] for _ in range(self.num_agent)]

    def reset(self):
        self.channel_state = [1 for _ in range(self.num_channel)]  # 0:idle, 1:successful 2:collision;

        self.queue_length = [0 for _ in range(self.num_agent)]    #  backlogged packets
        #self.generate_time = [[] for _ in range(self.num_agent)]  #  queue buffer


        self.obs = [[np.zeros(self.obs_obs_dim) for _ in range(self.obs_his_len)] for _ in range(self.num_agent)]

        self.num_successful_packets = [0 for _ in range(self.num_agent)]
        #self.mean_delay = [0 for _ in range(self.num_agent)]
        self.throughput = 0

    def step(self, actions, time):  #actions NXK

        random_numbers = np.array([random.random() for _ in range(self.num_channel)])
        index = np.where(random_numbers < self.state_trans_pro)[0]  # channel state transition
        for idx in index:
            self.channel_state[idx] = 1 if self.channel_state[idx] == 0 else 1


        random_numbers = np.array([random.random() for _ in range(self.num_agent)])
        index = np.where(random_numbers < self.arr_pro)[0]  # packet generation
        for idx in index:
            #self.generate_time[idx].append(time)
            self.queue_length[idx] += 1

        actions = np.array(actions)

        for idx in range(self.num_channel):      # get feedback for each channel
            if self.channel_state[idx] == 0:
                self.channel_feedback[idx] = -1
            else:
                x = np.where(actions[:, idx] == 1)[0]
                if len(x) == 1:                          # successful if only one node transmits
                    self.channel_feedback[idx] = 1
                    self.num_successful_packets[x[0]] += 1
                    #self.mean_delay[x[0]] += (self.generate_time[x[0]][0] - time + 1)/self.num_successful_packets[x[0]]
                    #del self.generate_time[x[0]][0]
                    self.queue_length[x[0]] -= 1
                elif len(x) > 1:                   # collision
                    self.channel_feedback[idx] = 1/len(x)
                else:                            # idle
                    self.channel_feedback[idx] = 0


        # update obs history
        for agent in range(self.num_agent):
            new_obs = actions[agent] * np.array(self.channel_feedback)
            self.obs[agent] = [new_obs] + self.obs[agent][:-1]



        if time == self.episode_length-1:
            self.throughput = sum(self.num_successful_packets)/self.episode_length

        return self.obs

    def get_throughput(self):
        return self.throughput

    def get_sum_success(self):
        return self.num_successful_packets

    def get_short_term_throughput(self, time):
        return sum(self.num_successful_packets)/(time + 1)

    def get_obs_his(self):
        return self.obs