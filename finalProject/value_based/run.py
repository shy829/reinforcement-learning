from DDQN import DDQN
import matplotlib.pyplot as plt
import gym
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="BreakoutNoFrameskip-v4", help="lab env")
    args = parser.parse_args()

    ddqn = DDQN(args.env_name)
    ddqn.Train()