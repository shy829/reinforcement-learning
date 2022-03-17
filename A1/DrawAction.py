import matplotlib.pyplot as plt

def drawAction(action):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    for i in range(6):
        for j in range(6):
            tmp = action[i][j]
            if tmp >= 8:
                tmp -= 8
                ax.arrow(2 * j + 1, 12 - 2 * i - 1, 1, 0, length_includes_head=True, head_width=0.2, head_length=0.4)

            if tmp >= 4:
                ax.arrow(2 * j + 1, 12 - 2 * i - 1, -1, 0, length_includes_head=True, head_width=0.2, head_length=0.4)
                tmp -= 4
            if tmp >= 2:
                ax.arrow(2 * j + 1, 12 - 2 * i - 1, 0, -1, length_includes_head=True, head_width=0.2, head_length=0.4)
                tmp -= 2
            if tmp >= 1:
                ax.arrow(2 * j + 1, 12 - 2 * i - 1, 0, 1, length_includes_head=True, head_width=0.2, head_length=0.4)
                tmp -= 1
    ax.grid()
    plt.show()