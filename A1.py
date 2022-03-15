import numpy as np
import matplotlib.pyplot as plt

gridWorld = np.zeros((6, 6))
tmp = np.zeros((6, 6))
theta = 0.01

if_convergence = False
while not if_convergence:
    delta = 0
    for i in range(6):
        for j in range(6):
            if (i == 0 and j == 1) or (i == 5 and j == 5):
                continue
            else:
                # i--
                gridW = gridWorld[i][j-1] - 1 if j > 0 else gridWorld[i][j] - 1
                # i++
                gridE = gridWorld[i][j+1] - 1 if j < 5 else gridWorld[i][j] - 1
                # j--
                gridN = gridWorld[i-1][j] - 1 if i > 0 else gridWorld[i][j] - 1
                # j++
                gridS = gridWorld[i+1][j] - 1 if i < 5 else gridWorld[i][j] - 1
                tmp[i][j] = 0.25 * (gridS + gridN + gridE + gridW)
                delta = max(delta, abs(tmp[i][j] - gridWorld[i][j]))
                gridWorld[i][j] = tmp[i][j]
    print(delta)
    if_convergence = True if (delta < theta) else False

print(gridWorld)
plt.matshow(gridWorld)
plt.show()
mat = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1]]
plt.matshow(mat)
for i in range(6):
    for j in range(6):
        west = gridWorld[i][j-1] - gridWorld[i][j] if (j > 0) else 0
        east = gridWorld[i][j+1] - gridWorld[i][j] if (j < 5) else 0
        north = gridWorld[i-1][j] - gridWorld[i][j] if (i > 0) else 0
        south = gridWorld[i+1][j] - gridWorld[i][j] if (i < 5) else 0
        go = ""
        max_n = max(west, east, south, north)
        if west == max_n:
            go += "w /"
        if east == max_n:
            go += "e /"
        if south == max_n:
            go += "s /"
        if north == max_n:
            go += "n /"
        go = go[:-1]
        plt.text(x=j, y=i, s=go)
plt.show()
