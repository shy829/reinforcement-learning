import numpy as np

"""
Q is the Q(state,action), 
e is the possibility that takes the action differ from 
the best action it observes
"""
e = 0.1
mv = [[0, -1], [0, 1], [-1, 0], [1, 0]]
x_len = 12
y_len = 4


def eg_policy(x, y, Q, e):
    if np.random.uniform(0, 1) < e:
        action = random(x, y)
    else:
        action = predict(x, y, Q)
    return action


def possible_a(x, y):
    action = []
    for i in range(4):
        new_x = x + mv[i][0]
        new_y = y + mv[i][1]
        if 0 <= new_x < x_len and 0 <= new_y < y_len:
            action.append(i)
    return action


def random(x, y):
    # only takes the move that doesn't go beyond boundary
    return np.random.choice(possible_a(x, y))


def predict(x, y, Q):
    Q_all = dict()
    p_a = possible_a(x,y)
    for i in p_a:
        Q_all.update({i: Q[x][y][i]})
    mQ = []
    for k, v in Q_all.items():
        if v == max(Q_all.values()):
            mQ.append(k)
    action = np.random.choice(mQ)
    return action
