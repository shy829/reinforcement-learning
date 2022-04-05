import numpy as np
from e_greedy import predict
from Observe import observe

x_len = 12
y_len = 4


def drawpath(Q):
    action = np.array([["0", "0", "0", "0"],["0", "0", "0", "0"],["0", "0", "0", "0"]
                      ,["0", "0", "0", "0"],["0", "0", "0", "0"],["0", "0", "0", "0"]
                      ,["0", "0", "0", "0"],["0", "0", "0", "0"],["0", "0", "0", "0"]
                      ,["0", "0", "0", "0"],["0", "0", "0", "0"],["0", "0", "0", "0"]])
    x=0
    y = 3
    while True:
        a = predict(x, y, Q)
        if a == 0:
            action[x][y] = "↑"
        elif a == 1:
            action[x][y] = "↓"
        elif a == 2:
            action[x][y] = "←"
        elif a == 3:
            action[x][y] = "→"
        x_next, y_next, reward = observe(x, y, a)
        x = x_next
        y = y_next
        if x == x_len - 1 and y == y_len - 1:
            action[x][y] = "E"
            break
    for i in range(4):
        for j in range(12):
            print(action[j][i], end="\t")
        print("")
