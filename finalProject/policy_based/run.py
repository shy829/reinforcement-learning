import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
import pygame
import mujoco

from TD3 import TD3

# hyper parameters
seed = 2022
load_model = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Hopper-v2", help="lab env")
    args = parser.parse_args()
    
    env = gym.make(args.env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_num = env.observation_space.shape[0]
    action_num = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    td3 = TD3(state_num, action_num, max_action, args.env)
    if load_model:
        td3.load_model()
    
    td3.train()
    
    