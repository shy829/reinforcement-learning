from DQN import DQN
from DDQN import DDQN
from Dueling_DQN import Dueling_DQN
import matplotlib.pyplot as plt
import gym

if __name__ == '__main__':
    dqn = DQN()
    time1, id1, reward1 = dqn.Train(times=50)
    ddqn = DDQN()
    time2, id2, reward2 = ddqn.Train(times=50)
    dueling_dqn = Dueling_DQN()
    time3, id3, reward3 = dueling_dqn.Train(times=50)

    plt.plot(id1, time1,label="DQN")
    plt.plot(id2, time2,label="DDQN")
    plt.plot(id3, time3,label="Dueling-DQN")
    plt.legend()
    plt.show()

    plt.plot(id1, reward1, label="DQN")
    plt.plot(id2, reward2, label="DDQN")
    plt.plot(id3, reward3, label="Dueling-DQN")
    plt.legend()
    plt.show()


