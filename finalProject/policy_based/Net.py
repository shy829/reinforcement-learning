import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import hidden_init

class Actor(nn.Module):
    def __init__(self,state_num, action_num, max_action, layer1_unit=400, layer2_unit=300) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_num, layer1_unit)
        self.l2 = nn.Linear(layer1_unit, layer2_unit)
        self.l3 = nn.Linear(layer2_unit, action_num)
        
        self.l1.weight.data.uniform_(*hidden_init(self.l1))
        self.l2.weight.data.uniform_(*hidden_init(self.l2))
        self.l3.weight.data.uniform_(-3e-3, 3e-3)
        
        self.max_action = max_action
        
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x)) * self.max_action
    

class Critic(nn.Module):
    def __init__(self, state_num, action_num, layer1_unit=400, layer2_unit=300) -> None:
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_num, layer1_unit)
        self.l2 = nn.Linear(layer1_unit + action_num, layer2_unit)
        self.l3 = nn.Linear(layer2_unit, 1)
        self.l1.weight.data.uniform_(*hidden_init(self.l1))
        self.l2.weight.data.uniform_(*hidden_init(self.l2))
        self.l3.weight.data.uniform_(-3e-4, 3e-4)
        
        # Q2
        self.l4 = nn.Linear(state_num, layer1_unit)
        self.l5 = nn.Linear(layer1_unit + action_num, layer2_unit)
        self.l6 = nn.Linear(layer2_unit, 1)
        self.l1.weight.data.uniform_(*hidden_init(self.l4))
        self.l2.weight.data.uniform_(*hidden_init(self.l5))
        self.l3.weight.data.uniform_(-3e-4, 3e-4)
        
    def forward(self, s, a):
        # (state, action) -> Q
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
    # just make q1
    def forward1(self, s, a):
        # (state, action) -> Q
        s_a = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
        
        