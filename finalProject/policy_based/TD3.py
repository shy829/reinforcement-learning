from copy import deepcopy
import signal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import pygame
from tqdm import tqdm
import mujoco_py

from Net import Actor, Critic
from ReplayBuffer import ReplayBuffer
from utils import draw_ep_reward, draw_ep_step, draw_ev_reward

# set hyper parameters
batch_size = 256
kernel_size = 20
start_train_time = 1e4
evaluate_point = 5000
lr = 3e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# noise: N(0, 0.2), clipped to(-0.5, 0.5)
# delayed update the actor and target critic net every 2 iterations
class TD3(object):
    def __init__(self, state_num, action_num, max_action, env_name, gamma=0.99,
                 tau=0.005, noise=0.2, clip=0.5, d=2) -> None:
        self.actor = Actor(state_num, action_num, max_action).to(device)
        self.actor_target = Actor(state_num, action_num, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_num, action_num).to(device)
        self.critic_target = Critic(state_num, action_num).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.state_num = state_num
        self.action_num = action_num
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.noise = noise * max_action
        self.clip = clip * max_action
        self.d = d
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        
        self.step = 0
        # use Gauss noise N(0, 0.1)
        self.action_noise = 0
        # float->int
        self.memory = ReplayBuffer(state_num, action_num, max_size=int(1e6))
        
        
    def choose_action(self, s):
        s = torch.FloatTensor(s.reshape(1, -1)).to(device)
        # self.actor.eval()
        # with torch.no_grad():
        a = self.actor(s).cpu().data.numpy().flatten()
        return a
    
    def learn(self):
        self.step+=1
        # sample N transitions from ReplayBuffer
        s, a, s_, r, done = self.memory.sample(batch_size)
        
        # select next action from actor target and add noise
        with torch.no_grad():
            # clamp makes it in range (-c, c)
            noise = (torch.rand_like(a)*self.noise).clamp(-self.clip, self.clip)
            a_ = (self.actor_target(s_) + noise).clamp(-self.max_action, self.max_action)
            
            # compute target Q value
            # y = r + gamma * min(Q1'(s_, a_), Q2'(s_, a_))
            t_Q1, t_Q2 = self.critic_target(s_, a_)
            t_Q = torch.min(t_Q1, t_Q2)
            y = r + self.gamma * t_Q * (1-done)
        
        # update ctirics θi = argminθi(N-1 Σ(y-Qθi(s, a))^2) 
        Q1, Q2 = self.critic(s, a)
        critic_loss = F.mse_loss(Q1, y) + F.mse_loss(Q2, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # delay update
        if self.step % self.d == 0:
            actor_loss = -self.critic.forward1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # update target net
            self.soft_update(self.critic, self.critic_target)
            self.soft_update(self.actor, self.actor_target)
            
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                    (1.0 - self.tau) * target_param.data)
    
    def save_model(self):
        torch.save(self.critic.state_dict(), "./model/td3_critic" + self.env_name)
        torch.save(self.critic_optimizer.state_dict(), "./model/td3_critic_optimizer" + self.env_name)
        torch.save(self.actor.state_dict(), "./model/td3_actor" + self.env_name)
        torch.save(self.actor_optimizer.state_dict(), "./model/td3_actor_optimizer" + self.env_name)
        
    def load_model(self):
        self.critic.load_state_dict(torch.load("./model/td3_critic" + self.env_name))
        self.critic_optimizer.load_state_dict(torch.load("./model/td3_critic_optimizer" + self.env_name))
        self.actor.load_state_dict(torch.load("./model/td3_actor"+ self.env_name))
        self.actor_optimizer.load_state_dict(torch.load("./model/td3_actor_optimizer" + self.env_name))
        self.critic_target = deepcopy(self.critic)
    
    # evaluate every 5000 steps
    # reports the average reward over 10 episodes with no exploration noise 
    def evaluate(self, seed=2022, episode_num=10):
        new_env = gym.make(self.env_name)
        new_env.seed(seed)
        ep_r = 0
        for i in range(episode_num):
            s = new_env.reset()
            done = False
            while not done:
                a = self.choose_action(np.array(s))
                s_, r, done, _ = new_env.step(a)
                s = s_
                ep_r += r
                
        print(f"evaluate average reward: {(ep_r/episode_num):.2f}")
        return ep_r/episode_num
                
        
    def train(self, train_step = 1000000):
        ep_r = 0
        ep_num = 0
        ep_step = 0
        
        # total steps and reward per episode
        ep_step_list = []
        ep_r_list = []
        # evaluation reward list
        ev_r_list = []
        ev = [self.evaluate()]
        
        s = self.env.reset()
        done = False
        p = tqdm(range(train_step), total=train_step, ncols=50, leave=False, unit='b')
        
        
        for step in p:
            # self.env.render()
            ep_step += 1
            # first choose randomly to train
            if step < start_train_time:
                a = self.env.action_space.sample()
            else:
                # add Gaussian noise to action
                a = (self.choose_action(np.array(s)) + np.random.normal(0, self.max_action * 0.1, size=self.action_num)).clip(-self.max_action, self.max_action)
            
            s_, r, done, _ = self.env.step(a)
            
            if ep_step >= self.env._max_episode_steps:
                done = 0
            done = float(done)
            
            self.memory.store_transition(s, a, s_, r, done)
            
            s = s_
            ep_r += r
            
            if step>=start_train_time:
                self.learn()
                
            if done:
                ep_step_list.append(ep_step)
                ep_r_list.append(ep_r)
                print("episode: ",ep_num+1, " step: ", step+1, " episode step: ", ep_step, f" reward: {ep_r:.2f}")
                
                s = self.env.reset()
                done = False
                ep_r = 0
                ep_step = 0
                ep_num += 1
                
            if (step+1)%evaluate_point==0:
                ev_r = self.evaluate()
                ev_r_list.append(ev_r)
                ev.append(ev_r)
                np.save("./result/td3_evaluate", ev)
                self.save_model()
        
        # use median filter 
        # ep_step_list = signal.medfilt(ep_step_list, kernel_size=kernel_size)
        # ep_r_list = signal.medfilt(ep_r_list, kernel_size=kernel_size)
        
        draw_ep_reward(ep_r_list)
        draw_ev_reward(ev_r_list)
        draw_ep_step(ep_step_list)
        
                
            
            
            
        
        
    