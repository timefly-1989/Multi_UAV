import sys
sys.path.append(".")
from env import Env
import numpy as np
import torch as th
from bicnet import BiCNet
import argparse, datetime

MAX_EPISODES = 20000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 10000

def train(args):
    uav_num_working = 2
    charging_num = 2
    uav_num_waiting = charging_num
    user_goal_num = 2
    uav_num = uav_num_working+uav_num_waiting

    n_states = 7
    n_actions = 2
    n_agents = uav_num

    env = Env(uav_num_working, uav_num_waiting, charging_num, user_goal_num)
    bicnet = BiCNet(n_states, n_actions, n_agents, args)
    bicnet.load_model()

    var = 3  # control exploration
    for i in range(MAX_EPISODES):
        obs = env.reset()
        print(bicnet.choose_action(obs, noisy=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', default=1e10, type=int)
    parser.add_argument('--episode_length', default=50, type=int)
    parser.add_argument('--memory_length', default=int(1e5), type=int)
    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--a_lr', default=0.0001, type=float)
    parser.add_argument('--c_lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--render_flag', default=False, type=bool)
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=10000, type=int)
    parser.add_argument('--tensorboard', default=True, action="store_true")
    parser.add_argument("--save_interval", default=5000, type=int)
    parser.add_argument("--model_episode", default=240000, type=int)
    parser.add_argument('--episode_before_train', default=1000, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    args = parser.parse_args()
    train(args)