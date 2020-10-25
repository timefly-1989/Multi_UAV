from env import Env
import numpy as np
import torch as th
from DDPG import DDPG

MAX_EPISODES = 20000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 10000
n_chargings = 2
n_goals = 2

env = Env(n_chargings, n_goals)
action_dim = n_chargings * 2
state_dim = 6 * n_chargings + 4 * n_chargings * n_goals 
ddpg = DDPG(action_dim, state_dim)
def train():
    var = 3  # control exploration
    for i in range(MAX_EPISODES):
        obs = env.reset()
        total_reward = 0.0
        for j in range(MAX_EP_STEPS):
            env.render()
            action = ddpg.choose_action(obs)
            obs_, reward, done = env.step(action.numpy())
            ddpg.store_transition(obs, action, reward, obs_)
            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()
            obs = obs_
            total_reward += reward
        print('Episode: %d, reward = %f' % (i, total_reward))

train()