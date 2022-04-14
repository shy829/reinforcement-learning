import torch
from torch import nn
import numpy as np
import pandas as pd
import gym
import os
import collections

# Hyper Parameter
BATCH_SIZE = 32
LR = 0.01
Epsilon = 0.9
Gamma = 0.9
# target net updating rate
TARGET_REPLACE_ITER = 20
MEMORY_CAPACITY = 2000
b_size = 1000

env = gym.make("MountainCar-v0")
env = env.unwrapped
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(n_states, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        self.adv = nn.Linear(16, n_actions, bias=True)
        self.val = nn.Linear(16, 1, bias=True)
        # optimizer and loss_func
        self.opt = torch.optim.Adam(self.parameters(), lr=LR)
        self.mls = nn.MSELoss()

    def forward(self, x):
        x = self.features(x)
        adv = self.adv(x)
        val = self.val(x)
        x = val + adv - adv.mean()
        return x


class Dueling_DQN(object):
    def __init__(self):
        # initialize the eval net and the target net
        self.eval_net, self.target_net = Net(), Net()
        # learn step counter
        self.learn_cnt = 0
        # memory counter
        self.mem_cnt = 0
        # [s,a,s_,r], s takes place of n_states
        self.memory = np.zeros((MEMORY_CAPACITY, n_states * 2 + 2))
        if os.path.exists('learning.csv'):
            print("open file")
            data = pd.read_csv('learning.csv')
            for index, row in data.iterrows():
                self.memory[index, 0:1] = row['s1']
                self.memory[index, 1:2] = row['s2']
                self.memory[index, 2:3] = row['a']
                self.memory[index, 3:4] = row['r']
                self.memory[index, 4:5] = row['s_1']
                self.memory[index, 5:6] = row['s_2']
        print(self.memory)

    # observe by state s, choose action by epsilon-greedy method
    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        # choose greedy action
        if np.random.uniform() < Epsilon:
            v = self.eval_net.forward(s)
            action = torch.max(v, 1)[1].data.numpy()[0]
        # choose random action
        else:
            action = env.action_space.sample()
        return action

    def store_transition(self, s, a, r, s_):
        # combine s, a, r, s_
        transition = np.hstack((s, [a, r], s_))
        index = self.mem_cnt % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.mem_cnt += 1

    def Learn(self):
        # update target net
        if self.learn_cnt % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_cnt += 1

        # choose 32 samples from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        # take s, a, r, s_ from batch memory
        b_s = torch.FloatTensor(b_memory[:, :n_states])
        b_a = torch.LongTensor(b_memory[:, n_states:n_states + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, n_states + 1:n_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -n_states:])

        # calculate eval_net and loss function
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + Gamma * q_next.max(1)[0].reshape(BATCH_SIZE, 1)
        loss = self.eval_net.mls(q_eval, q_target)

        # updates eval_net
        self.eval_net.opt.zero_grad()
        loss.backward()
        self.eval_net.opt.step()

    def Train(self, times):
        print("Start Training!")
        l_time = []
        l_index = []
        l_reward=[]

        for i in range(times):
            s = env.reset()
            # counts total reward in an episode
            ep_r = 0
            cnt = 0
            while True:
                # env.render()
                a = self.choose_action(s)
                s_, r, done, info = env.step(a)

                """ Rewrite the reward function, start from -0.5, reach the end
                point at 0.5 . If goes lefter than start point, gives a negative reward. 
                The much closer it gets to the end point, it will receive a larger reward.
                """
                if -0.4 < s_[0] < 0.5:
                    r = 10 * (s_[0] + 0.4) ** 3
                elif s_[0] >= 0.5:
                    r = 100
                elif s_[0] <= -0.4:
                    r = -0.1

                self.store_transition(s, a, r, s_)
                ep_r += r

                if self.mem_cnt > MEMORY_CAPACITY:
                    self.Learn()
                    cnt += 1

                if done:
                # if done or cnt >= 500:
                    l_time.append(cnt)
                    l_index.append(i)
                    l_reward.append(ep_r)
                    print('Episode: ', i, '| total reward: ', ep_r, "| learning times:", cnt)
                    break

                s = s_
        print(self.memory)
        s1 = np.array(self.memory[:, 0:1])
        s1 = s1.flatten()
        s2 = np.array(self.memory[:, 1:2])
        s2 = s2.flatten()
        a = np.array(self.memory[:, 2:3])
        a = a.flatten()
        r = np.array(self.memory[:, 3:4])
        r = r.flatten()
        s_1 = np.array(self.memory[:, 4:5])
        s_1 = s_1.flatten()
        s_2 = np.array(self.memory[:, 5:6])
        s_2 = s_2.flatten()
        data = pd.DataFrame({"s1": s1, "s2": s2, "a": a, "r": r, "s_1": s_1, "s_2": s_2})
        data.to_csv('learning.csv')

        return l_time,l_index,l_reward

