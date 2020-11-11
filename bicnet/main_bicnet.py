import sys
from env import Env
import numpy as np
import torch
from bicnet import BiCNet
import argparse, datetime

def train(args):
    uav_num_working = 2
    charging_num = 2
    uav_num_waiting = charging_num
    user_goal_num = 2
    uav_num = uav_num_working+uav_num_waiting

    state_dim = 7
    action_dim = 2
    torch.manual_seed(args.seed)

    env = Env(uav_num_working, uav_num_waiting, charging_num, user_goal_num)
    bicnet = BiCNet(state_dim, action_dim, uav_num, args)
    bicnet.load_model()
    episode = 0
    total_step = 0
    while episode < args.max_episodes:
        state = env.reset()
        episode += 1
        step = 0
        accum_reward = 0
        while True:
            action = bicnet.choose_action(state, noisy=True)
            next_state, reward, done = env.step(action)
            env.render()
            step += 1
            total_step += 1
            reward = np.array(reward)
            bicnet.memory(state, action, reward, next_state, done)
            state = next_state
            if args.episode_length < step or (True in done):
                c_loss, a_loss = bicnet.update(episode)
                print("[Episode %05d] reward %6.4f" % (episode, accum_reward))
                if c_loss and a_loss:
                    print(" a_loss %3.2f c_loss %3.2f" % (a_loss, c_loss), end='')
                if episode % args.save_interval == 0:
                    bicnet.save_model(episode)
                break
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episodes', default=1e10, type=int)
    parser.add_argument('--mode', default="eval", type=str, help="train/eval")
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