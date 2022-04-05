# from Sarsa import Sarsa
from QL import q_learning
from Sarsa_new import Sarsa
from DrawPath import drawpath
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a = [0, 1, 2, 3]
    Cliff2 = q_learning(a, x_len=12, y_len=4, reward=0, gamma=0.99, alpha=0.5, epsilon=0.01)
    Cliff = Sarsa(a, x_len=12, y_len=4, reward=0, gamma=0.99, alpha=0.5, epsilon=0.01)
    cnt, rewards = Cliff.task(times=500)
    print("Sarsa path:")
    drawpath(Cliff.Q)
    cnt1, rewards1 = Cliff2.task(times=500)
    print("Q-Learning path:")
    drawpath(Cliff2.Q)

    plt.plot(cnt, rewards, label="Sarsa")
    plt.plot(cnt1 ,rewards1,label = "Q_learning")
    plt.ylim(-100, 0)
    plt.legend(loc="lower right")
    plt.show()
