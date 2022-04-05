# Q-Learning
import numpy as np
from Observe import observe
from e_greedy import eg_policy, predict

# Action list
"""
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
"""
# respectively corresponding to UP, DOWN, LEFT and RIGHT
mv = [[0, -1], [0, 1], [-1, 0], [1, 0]]
batch_size = 20


class q_learning(object):
    """
    similar to Sarsa, just different when taking the next action.
    It only takes the prediction without random.
    """
    def __init__(self, a, x_len, y_len, reward, gamma=0.9, alpha=0.01, epsilon=0.1):
        self.a = a
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.x_len = x_len
        self.y_len = y_len
        self.reward = reward
        # state, action
        self.Q = np.zeros([x_len, y_len, 4])

    def run_episode(self):
        reward_sum = 0
        x = 0
        y = 3

        while True:
            a = eg_policy(x, y, self.Q, self.epsilon)
            next_x, next_y, reward = observe(x, y, a)
            next_a = predict(next_x, next_y, self.Q)
            reward_sum += reward
            self.Q[x][y][a] += self.alpha * (reward + self.gamma * self.Q[next_x][next_y][next_a] - self.Q[x][y][a])
            if next_x == self.x_len - 1 and next_y == self.y_len - 1:
                # print("ql", reward_sum)
                return reward_sum
            x = next_x
            y = next_y

    def task(self, times):
        rewards = np.zeros([times])
        cnt = []
        avg_rewards = []
        for i in range(times):
            total_reward = 0
            for j in range(batch_size):
                total_reward += self.run_episode()
            total_reward /= batch_size
            rewards[i] = total_reward
        for i in range(9):
            cnt.append(i)
            avg_rewards.append(rewards[i])
        for i in range(10, len(rewards) + 1):
            cnt.append(i)
            avg_rewards.append(np.mean(rewards[i - 10:i]))
        return cnt, avg_rewards
