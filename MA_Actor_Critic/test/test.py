import numpy as np
import time
import argparse
import os

from env.env_core import EnvCore
from run.env_test_runner import EnvRunner
'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
#add_argument('--max_train_times', type=int, default=int(1e4), help='maximum training times')
parser.add_argument('--episode_num', type=int, default=int(5), help='num of episodes per train')
parser.add_argument('--episode_length', type=int, default=int(1e5), help='number of time slots per episode')
#parser.add_argument('--deep_copy_per_episode', type=int, default=int(5), help='num of train times to deepcopy eva DQN')


parser.add_argument('--epsilon', type=float, default=0.1, help='exploration probability')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')

parser.add_argument('--allowed_channel_num', type=int, default=int(1), help='the number of channels allowed to be selected per slot')
parser.add_argument('--obs_his_len', type=int, default=int(16), help='the length of the kept obs for input')

parser.add_argument('--actor_net_width', type=int, default=int(200), help='Hidden net width of actor network')
parser.add_argument('--critic_net_width', type=int, default=int(200), help='Hidden net width of critic network')

parser.add_argument('--actor_lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--critic_lr', type=float, default=0.0005, help='Learning rate')




args = parser.parse_args()

def make_env(args, arr_pro, state_trans_pro):
    return EnvCore(args, arr_pro, state_trans_pro)

def main():


    num_agent = int(3)
    num_channel = int(3)
    arr_pro = 1
    state_trans_pro = 0

    args.num_agent = num_agent
    args.num_channel = num_channel
    args.action_dim = args.num_channel
    args.obs_dim = args.num_channel

    root_path = os.path.dirname(os.getcwd())
    args.save_dir = os.path.join(root_path, "results")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # args.save_dir = os.path.join(args.save_dir, args.reward_type)
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, "N" + str(args.num_agent) + "K" + str(args.num_channel))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    env = make_env(args, arr_pro, state_trans_pro)
    runner = EnvRunner(args, env)
    runner.run()





if __name__ == "__main__":
    main()
