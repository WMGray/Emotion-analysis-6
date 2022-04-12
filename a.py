import pandas as pd
import numpy as np
import jieba
import json
import codecs
import os
import yaml

import multiprocessing
import matplotlib.pyplot as plt
from config import Config

vocab_dim = 200  # 词向量维度
maxlen = 100     # 序列最大长度
n_iterations = 10  # 迭代次数
n_exposures = 10    # 词频截断值
window_size = 7  # 窗口大小
batch_size = 128  # 批次大小
n_epoch = 20      # 迭代次数
input_length = 100  # 输入序列长度
cpu_count = multiprocessing.cpu_count()  #


null = pd.read_csv(Config.new_List[0], header=None, index_col=None)  # null
like = pd.read_csv(Config.new_List[1], header=None, index_col=None)  # like
sad = pd.read_csv(Config.new_List[2], header=None, index_col=None)  # sad
disgust = pd.read_csv(Config.new_List[3], header=None, index_col=None)  # disgust
anger = pd.read_csv(Config.new_List[4], header=None, index_col=None)  # anger
happy = pd.read_csv(Config.new_List[5], header=None, index_col=None)  # happy

# concatenate 能够一次完成多个数组的拼接
combined = np.concatenate((null[0], like[0], sad[0], disgust[0], anger[0], happy[0]))
print(type(combined))
# null-0 like-1 sad-2 disgust-3 anger-4 happy-5
y = np.concatenate((np.zeros(len(null), dtype=int), np.ones(len(like), dtype=int),
                    np.ones(len(sad), dtype=int) * 2, np.ones(len(disgust), dtype=int) * 3,
                    np.ones(len(anger), dtype=int) * 4, np.ones(len(happy), dtype=int) * 5))

print(len(null), len(like), len(sad), len(disgust), len(anger), len(happy))
print(len(combined), len(y))

List = combined.tolist()
Set = set(List)
print(len(Set),len(List))