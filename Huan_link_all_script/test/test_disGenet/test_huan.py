import os
import dgl
import torch as th
import torch
import numpy as np
from dgl.sampling import random_walk
import pandas as pd
from collections import defaultdict
from dgl.data.utils import load_graphs, save_graphs
from tqdm import tqdm
from multiprocessing import Pool
import sys



g_npz = np.load(os.path.join("/public/home/huanhuan/test_disGenet/train_sample.npz"))
g_npz.files
c = g_npz['c']
p = g_npz['p']
n = g_npz['n']

a= th.tensor([c,p,n])




a = th.tensor([1,2])

for doc in tqdm(corpus):
    print(doc.shape)



for c in range(half_windowsize, document_length - half_windowsize):
    center_word =[doc[c]]


for i in center_word + pos_word:
    tmp_frequency[i] = 0

neg_word = np.random.choice(np.arange(frequency.shape[0]), size=(n_negword * 2 * half_windowsize),
                            p=tmp_frequency / tmp_frequency.sum()).tolist()

for doc in tqdm(corpus): #tqdm进度条库
    for c in range(half_windowsize, document_length - half_windowsize):
        center_word = [doc[c]]
        pos_word = doc[c - half_windowsize:c].tolist() + doc[c + 1:c + half_windowsize + 1].tolist()
        tmp_frequency = frequency.copy()
        for i in center_word + pos_word:
            tmp_frequency[i] = 0
        neg_word = np.random.choice(np.arange(frequency.shape[0]), size=(n_negword * 2 * half_windowsize),
                                    p=tmp_frequency / tmp_frequency.sum()).tolist()  #------#除去center_word + pos_word，p为a中每个采样点的概率分布
        center_words.append(center_word)
        pos_words.append(pos_word)
        neg_words.append(neg_word)