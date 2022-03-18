# Temporal-Difference Learning

import numpy as np
from GridWorld import create_random_path

alpha = 0.1
gamma = 0.7
Value = np.zeros(36)
Number = np.zeros(36)


# V(S) = V(S) + alpha[R+gamma*V(S')-V(S)]
# V(S') is the value of next state
# alpha is step size defined by oneself
# gamma is discount factor
# others are similar to mc
def td0_learning(try_times):
    for i in range(try_times):
        path, reward = create_random_path()
        for i in range(len(path) - 1):
            next_v = Value[path[i + 1]]
            R = 0 if (path[i + 1] == 1 or path[i + 1] == 35) else -1
            delta = alpha * (R + gamma * next_v - Value[path[i]])
            Value[path[i]] += delta
    return Value
