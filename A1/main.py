# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from GridWorld import buildGrid
from ValueIteration import valueIteration
from PolicyIteration import policyIteration

policy_grid = buildGrid(6)
value_grid = buildGrid(6)

if __name__ == '__main__':
    policyIteration(policy_grid, theta=0.01, end={0: 1, 5: 5})
    valueIteration(value_grid, theta=0.00001, end={0: 1, 5: 5})
