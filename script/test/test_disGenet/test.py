import dgl
import torch as th
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from dgl.data.utils import load_graphs, save_graphs
from torch import nn
from torch.nn import functional as F
from torch.utils import data
# from rst_pheno.homograph_word2vec import random_walk #-------------------------------
from tqdm import tqdm
import os
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


x = torch.tensor([0, 1, 2, 3, 4])
th.save(x, 'tensor.pth')

a = np.load(os.path.join('/public/home/huanhuan/test_disGenet/train_sample.npz'))

d = dict(zip(("data1{}".format(k) for k in a), (a[k] for k in a)))
th.save(d,"train_sample.pth")

d = dict(zip(("c","p","n"), (data[k] for k in data)))

c=center_words, p=pos_words, n=neg_words


th.save('/public/home/huanhuan/test_disGenet/test_sample.npz',"aaa.pth")


center_words,pos_words,neg_words = th.load(os.path.join("/public/home/huanhuan/test_disGenet/aaa.pth"))



th.save(data ,"aaa.pth")



