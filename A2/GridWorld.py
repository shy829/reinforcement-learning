import numpy as np
from random import randint


# define move
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4


# deploy a 6*6 gridWorld in numbers 0~35 with final state 1 and 35
# start from a random state and choose random actions to go to the terminal state
# The function is to build a path that records every move, and it also calculates the rewards
def create_random_path():
    path = []
    reward = 0
    initial_pos = randint(0, 35)
    path.append(initial_pos)

    if initial_pos == 1 or initial_pos == 35:
        return path, reward

    pos = initial_pos
    while (pos != 1 and pos != 35):
        move = randint(1, 4)
        # if the move goes beyond the grid, leave it unchanged
        if move == 1:
            pos = pos - 6 if pos >= 6 else pos
        if move == 2:
            pos = pos + 6 if pos < 30 else pos
        if move == 3:
            pos = pos - 1 if pos % 6 != 0 else pos
        if move == 4:
            pos = pos + 1 if pos % 6 != 5 else pos
        # Each movement get a reward of -1 until the terminal state is reached
        if pos!=1 and pos!=35:
            reward-=1

        path.append(pos)
    return path,reward