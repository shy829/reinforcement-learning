# use Monte-Carlo Method
import numpy as np

from GridWorld import create_random_path


# V(St) = V(St) + 1/N(St)(Gt-V(St))
# V is the state value
# Gt is the final reward
# N is the number of times
# define try_times by oneself
# for first_visit: calculate the average of the first access to s
# 将所有第一次访问到 s 得到的回报求均值
def first_visit(try_times):
    Value = np.zeros(36)
    Number = np.zeros(36)
    for i in range(try_times):
        path, reward = create_random_path()

        # If we use set() to delete repeat contents,
        # we can't make sure that the remained number in set is the first one appears in the formal list
        new_path = delete_duplicate_content(path)
        for pos in range(36):
            if pos in new_path:
                Number[pos] += 1
                Value[pos] = Value[pos] + (1 / Number[pos]) * (reward - Value[pos])
    return Value


def every_visit(try_times):
    Value = np.zeros(36)
    Number = np.zeros(36)
    for i in range(try_times):
        path, reward = create_random_path()
        for pos in path:
            Number[pos] += 1
            Value[pos] = Value[pos] + (1 / Number[pos]) * (reward - Value[pos])
    return Value


def delete_duplicate_content(lst):
    new_list = []
    for i in lst:
        if i not in new_list:
            new_list.append(i)
    return new_list
