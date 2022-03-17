import numpy as np
from DrawAction import drawAction

tmp = np.zeros((6, 6))
action = np.zeros((6, 6))

UP = 1
DOWN = 2
LEFT = 4
RIGHT = 8

# initialize action
for row in range(6):
    for column in range(6):
        action[row][column] = 15


def valueIteration(gridWorld, theta, end):
    cnt = 0
    is_convergence = False
    while not is_convergence:
        delta = 0
        for i in range(6):
            for j in range(6):
                if i in end.keys() and end[i] == j:
                    action[i][j] = 0
                    continue
                else:
                    # i--
                    gridW = gridWorld[i][j - 1] - 1 if j > 0 else gridWorld[i][j] - 1
                    # i++
                    gridE = gridWorld[i][j + 1] - 1 if j < 5 else gridWorld[i][j] - 1
                    # j--
                    gridN = gridWorld[i - 1][j] - 1 if i > 0 else gridWorld[i][j] - 1
                    # j++
                    gridS = gridWorld[i + 1][j] - 1 if i < 5 else gridWorld[i][j] - 1
                    # tmp[i][j] = 0.25 * (gridS + gridN + gridE + gridW)
                    tmp[i][j] = 0.25 * max(gridS, gridN, gridE, gridW)
                    action[i][j] = 0
                    if tmp[i][j] == 0.25*gridN:
                        action[i][j] += 1
                    if tmp[i][j] == 0.25*gridS:
                        action[i][j] += 2
                    if tmp[i][j] == 0.25*gridW:
                        action[i][j] += 4
                    if tmp[i][j] == 0.25*gridE:
                        action[i][j] += 8
                    delta = max(delta, abs(tmp[i][j] - gridWorld[i][j]))

                    # gridWorld[i][j] = tmp[i][j]
        for i in range(6):
            for j in range(6):
                gridWorld[i][j] = tmp[i][j]
        cnt += 1
        print("Value Iteration:", delta)
        print("action:", action)
        is_convergence = True if (delta < theta) else False
    print(gridWorld)
    drawAction(action)
