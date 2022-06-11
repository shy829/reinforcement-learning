import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd
import gym
from tqdm import tqdm
import os
import pygame

from Net import Net
from utils import load_memory, save_log_score, save_model_params, preprocess

# set hyper parameter
learn_time = 100000
buf_size = 1000000
total_step = 5000000
lr = 0.0001
gamma = 0.99
eps = 1.5e-4

# epsilon start/end and change point
ep_s = 1
ep_e = 0.1
ep_c = 50000

n_states = 3
batch_size = 32
target_change = 20

Memory = namedtuple('Memory', ('s', 'a', 'r', 's_'))


class DDQN(nn.Module):
    def __init__(self, env_name):
        super(DDQN, self).__init__()
        self.env_name = env_name
        env = gym.make(env_name)
        self.env = env.unwrapped
        self.n_actions = self.env.action_space.n
        self.eval_net, self.target_net = Net(self.n_actions, lr, eps), \
                                         Net(self.n_actions, lr, eps)
        self.learn_cnt = 0
        self.mem_cnt = 0
        self.epsilon = ep_s
        # [s,a,r,s_]
        # the state in the atari environment is different from before, we can't use numpy to create a memory of fixed size
        # we just need to create an empty list and append into it and pop out the earliest one
        self.memory = []

    def choose_action(self, s, learn_mode=True):
        # learning mode, first 50000 steps takes epsilon = 1, which means we randomly choose actions.
        # then we decrease epsilon step by step but remains no less than 0.1
        if learn_mode:
            if self.mem_cnt >= ep_c:
                self.epsilon -= (ep_s - ep_e) / buf_size
                self.epsilon = max(self.epsilon, ep_e)
            epsilon = self.epsilon
        else:
            epsilon = 0.05

        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        if np.random.uniform() > epsilon:
            v = self.eval_net(torch.Tensor(s)).detach()
            action = torch.argmax(v).data.item()
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    # can add done
    def store_transition(self, s, a, r, s_):
        transition = [s, a, r, s_]
        if len(self.memory) >= buf_size:
            self.memory.pop(0)
        self.memory.append(transition)
        # index = self.mem_cnt % buf_size
        # self.memory[index, :] = transition
        self.mem_cnt += 1

    def Learn(self):
        # update target net
        if self.learn_cnt % target_change == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_cnt += 1

        # sample_index = np.random.choice(buf_size, batch_size)
        # b_memory = self.memory[sample_index, :]
        #
        # b_s = torch.FloatTensor(b_memory[:, :n_states])
        # b_a = torch.LongTensor(b_memory[:, n_states, n_states + 1].astype(int))
        # b_r = torch.FloatTensor(b_memory[:, n_states + 1, n_states + 2])
        # b_s_ = torch.FloatTensor(b_memory[:, -n_states])

        sample = random.sample(self.memory, batch_size)
        b_memory = Memory(*zip(*sample))
        b_s = torch.tensor(np.array(b_memory.s, dtype=np.float32), dtype=torch.float32)
        b_a = torch.tensor(b_memory.a).unsqueeze(1)
        b_r = torch.tensor(np.array(b_memory.r, dtype=np.float32), dtype=torch.float32).unsqueeze(1)
        b_s_ = torch.tensor(np.array(b_memory.s_, dtype=np.float32), dtype=torch.float32)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_cur = self.eval_net(b_s_).detach()
        q_tn = self.target_net(b_s_).detach()
        q_next = q_tn.gather(1, torch.max(q_cur, 1)[1].unsqueeze(1)).squeeze(1)
        q_target = b_r + gamma * q_next.reshape(batch_size, 1)
        loss = self.eval_net.mls(q_eval, q_target)
        avg_q = torch.sum(q_eval.detach()) / batch_size

        self.eval_net.opt.zero_grad()
        loss.backward()
        self.eval_net.opt.step()

        # return loss.item(), avg_q.item()

    def Train(self):
        print("Start Training!")
        p = tqdm(range(total_step), total=total_step, ncols=50, leave=False, unit='b')
        s = self.env.reset()
        # print(len(s),len(s[0]),len(s[0][0]))
        # (210, 160, 3)->(4, 84, 84)
        s = preprocess(s)
        s = np.reshape(s, (84, 84))
        s = np.stack((s, s, s, s), axis=0)
        # print(len(s), len(s[0]), len(s[0][0]))
        ddqn = DDQN(self.env_name)
        r_mean = 0
        r_max = 0
        r_sum = 0
        r_mean_max = 0
        round = 0
        save_point = 20

        for step in p:
            a = ddqn.choose_action(s, learn_mode=True)
            s_, r, done, _ = self.env.step(a)
            s_ = preprocess(s_)
            s_ = np.append(s_, s[:3, :, :], axis=0)

            ddqn.store_transition(s, a, r, s_)
            r_sum += r

            if len(ddqn.memory) > learn_time and ddqn.mem_cnt % 10:
                ddqn.Learn()

            s = s_

            if done:
                r_mean += r_sum
                r_max = max(r_max, r_sum)
                if (round + 1) % save_point == 0:
                    save_log_score(round, r_mean / save_point, r_max)
                    print("  ", round, r_mean / save_point, r_max)
                    if r_mean > r_mean_max:
                        r_mean_max = r_mean
                        save_model_params(ddqn.eval_net)
                    r_max, r_mean = 0, 0

                r_sum = 0
                round += 1
                s = self.env.reset()
                s = preprocess(s)
                s = np.reshape(s, (84, 84))
                s = np.stack((s, s, s, s), axis=0)

