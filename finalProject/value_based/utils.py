from datetime import datetime
import re
import os

import cv2
import pandas as pd
import numpy as np
import torch


def date_in_string():
    time = str(datetime.now())
    time = re.sub(' ', '_', time)
    time = re.sub(':', '', time)
    time = re.sub('-', '_', time)
    time = time[0:15]
    return time


def save_log_loss(i_steps, loss, avg_qscore):
    path = os.getcwd() + '/result'
    with open(path + '/loss.txt', 'a') as outfile:
        outfile.write(date_in_string() + '\t' + str(i_steps) + '\t' + str(loss) + '\t' + str(avg_qscore) + '\n')
    return


def save_log_score(i_episodes, mean_r, max_r):
    path = os.getcwd() + '/result'
    with open(path + '/score.txt', 'a') as outfile:
        outfile.write(date_in_string() + '\t' + str(i_episodes) + '\t' + str(mean_r) + '\t' + str(max_r) + '\n')
    return


def save_model_params(model):
    path = './result'
    torch.save(model.state_dict(), path + '/DDQN_breakout' + '.pkl')
    return


def preprocess(s):
    img = np.reshape(s, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    x_t.astype(np.uint8)
    x_t = np.moveaxis(x_t, 2, 0)
    return np.array(x_t).astype(np.float32) / 255.0


def load_memory(buf_size, n_states):
    memory = np.zeros((buf_size, 1))
    if os.path.exists('learning_DDQN.csv'):
        print("open file")
        data = pd.read_csv("learning_DDQN.csv")
        for index, row in data.iterrows():
            memory[index, 0:1] = row['s1']
            memory[index, 1:2] = row['s2']
            memory[index, 2:3] = row['s3']
            memory[index, 3:4] = row['a']
            memory[index, 4:5] = row['r']
            memory[index, 5:6] = row['s_1']
            memory[index, 6:7] = row['s_2']
            memory[index, 7:8] = row['s_3']
    return memory
