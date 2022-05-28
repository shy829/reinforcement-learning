from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    f = layer.weight.data.size()[0]
    lim = 1./np.sqrt(f)
    return (-lim, lim)

def draw_ep_reward(list):
    plt.plot(list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward")
    plt.savefig("./result/ep_reward.png")
    plt.show()
    
def draw_ev_reward(list):
    plt.plot(list)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Evaluation Average Reward")
    plt.savefig("./result/ev_reward.png")
    plt.show()

def draw_ep_step(list):
    plt.plot(list)
    plt.xlabel("Episode")
    plt.ylabel("Step")
    plt.title("Episode steps")
    plt.savefig("./result/ep_step.png")
    plt.show()