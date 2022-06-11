from urllib.parse import MAX_CACHE_SIZE
import numpy as np
import torch

# replay buffer
class ReplayBuffer(object):
    def __init__(self, state_num, action_num, max_size=int(1e6)) -> None:
        self.max_size = max_size
        self.p = 0
        self.size = 0
        
        self.s = np.zeros((max_size, state_num))
        self.a = np.zeros((max_size, action_num))
        self.s_ = np.zeros((max_size, state_num))
        self.r = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # store the (s, a, s_, r, done) into buffer
    def store_transition(self, s, a, s_, r, done):
        self.s[self.p] = s
        self.a[self.p] = a
        self.s_[self.p] = s_
        self.r[self.p] = r
        self.done[self.p] = done
        
        self.p = (self.p + 1) % self.max_size
        self.size = min(self.max_size, self.size+1)
    
    # sample batch_size
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        
        b_s = torch.FloatTensor(self.s[idx]).to(self.device)
        b_a = torch.FloatTensor(self.a[idx]).to(self.device)
        b_s_ = torch.FloatTensor(self.s_[idx]).to(self.device)
        b_r = torch.FloatTensor(self.r[idx]).to(self.device)
        b_done = torch.FloatTensor(self.done[idx]).to(self.device)
        
        return (b_s, b_a, b_s_, b_r, b_done)
        