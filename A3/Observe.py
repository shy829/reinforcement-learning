"""
x is the row number, y is the col number, a is the action
start from (0,3)
"""

# Action list
"""
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
"""
x_len = 12
y_len = 4


# move one step by action a and return the position and rewards
def observe(x, y, a):
    #  assume that it only takes the possible actions
    if a == 0:
        y -= 1
    if a == 1:
        y += 1
    if a == 2:
        x -= 1
    if a == 3:
        x += 1

    if y == 3 and (x != 0 and x != x_len - 1):
        return 0, 3, -100
    return x, y, -1
