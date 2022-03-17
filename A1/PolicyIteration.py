import numpy as np
from DrawAction import drawAction

# tmp just stores temporary state
# action is the action that takes

action = np.zeros((6, 6))
tmp = np.zeros((6, 6))

UP = 1
DOWN = 2
LEFT = 4
RIGHT = 8

# initialize action
for row in range(6):
    for column in range(6):
        action[row][column] = 15

def policyIteration(gridWorld, theta, end):
    is_stable = False
    print("initial action:", action)
    print("initial state:",gridWorld)
    drawAction(action)
    make_it_convergence(gridWorld, theta, end)

    while not is_stable:
        is_stable = True

        for i in range(6):
            for j in range(6):
                # reach final state!
                if i in end.keys() and end[i] == j:
                    action[i][j]=0
                    continue
                else:
                    # reward is -1 for each move
                    # j-- (LEFT--3)

                    gridW = gridWorld[i][j - 1] - 1 if j > 0 else gridWorld[i][j] - 1
                    # j++ (RIGHT--4)
                    gridE = gridWorld[i][j + 1] - 1 if j < 5 else gridWorld[i][j] - 1
                    # i-- (UP--1)
                    gridN = gridWorld[i - 1][j] - 1 if i > 0 else gridWorld[i][j] - 1
                    # i++ (DOWN--2)
                    gridS = gridWorld[i + 1][j] - 1 if i < 5 else gridWorld[i][j] - 1
                    best_action = max(gridS, gridN, gridE, gridW)
                    if (action[i][j] % 16 == 1 and best_action == gridN) or (
                            action[i][j] % 16 == 2 and best_action == gridS) \
                            or (action[i][j] % 16 == 4 and best_action == gridW) or (
                            action[i][j] % 16 == 8 and best_action == gridE):
                        continue
                    is_stable = False

                    action[i][j] = 0
                    if best_action == gridN:
                        action[i][j] += 1
                    if best_action == gridS:
                        action[i][j] += 2
                    if best_action == gridW:
                        action[i][j] += 4
                    if best_action == gridE:
                        action[i][j] += 8
        print("action change:", action)
        drawAction(action)
        if not is_stable:
            make_it_convergence(gridWorld, theta, end)
        print(gridWorld)


def make_it_convergence(gridWorld, theta, end):
    cnt = 0
    is_convergence = False
    while not is_convergence:
        delta = 0
        for i in range(6):
            for j in range(6):
                if i in end.keys() and end[i] == j:
                    continue
                else:
                    # j--
                    gridW = gridWorld[i][j - 1] - 1 if j > 0 else gridWorld[i][j] - 1
                    # j++
                    gridE = gridWorld[i][j + 1] - 1 if j < 5 else gridWorld[i][j] - 1
                    # i--
                    gridN = gridWorld[i - 1][j] - 1 if i > 0 else gridWorld[i][j] - 1
                    # i++
                    gridS = gridWorld[i + 1][j] - 1 if i < 5 else gridWorld[i][j] - 1

                    tmp[i][j] = 0.25 * (gridS + gridN + gridE + gridW)
                    delta = max(delta, abs(tmp[i][j] - gridWorld[i][j]))
                    gridWorld[i][j] = tmp[i][j]
        print("Policy Iteration:", delta)
        is_convergence = True if (delta < theta) else False
    print("gridWorld:",gridWorld)