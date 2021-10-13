import dgl
import torch
from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx
import torch as th
from torch import nn
from torch.nn import init
from dgl import function as fn
from dgl.nn.pytorch.utils import Identity
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dgl.data.utils import load_graphs, save_graphs
from time import time
from multiprocessing import Process, Value, Lock, Queue
import glob
from torch.utils.data import DataLoader as Dl
import pymysql


class Data_reader():
    def __init__(self, path,train_sample_size,test_sample_size ,batch_size):
        self.path = path
        self.batch_size = batch_size
        self.test_sample_size = test_sample_size
        self.train_sample_size = train_sample_size
        os.chdir(self.path)
        conn = pymysql.connect(host='127.0.0.1',port = 3306,user ='yzy',passwd='zhiyu')
        self.c = conn.cursor()
        self.c.execute("use dataset;")
        self.dp = Data_permutation(os.path.join(self.path, r'trainset'), self.train_sample_size,
                                   self.batch_size) # 训练集的乱序
        self.dnp = Data_non_permutation(os.path.join(self.path, r'testset'), self.test_sample_size,
                                        self.batch_size)  # 测试集的顺序

    def next_batch(self, istrainset):
        # t1 = time()
        batch_sample_list = []
        if istrainset:
            cur_idx = self.dp.next_batch_idx
        else:
            cur_idx = self.dnp.next_batch_idx
        Pa = 'trainset' if istrainset else 'testset'
        L = ','.join([str(i + 1) for i in cur_idx])  # i+1 因为数据库起始从1开始
        self.c.execute("select path from %s where id in (%s);" % (Pa, L))
        samples_path = self.c.fetchall()
        # t2 = time()
        # t6 = 0
        for i in samples_path:
            # t5 =time()
            sample = load_graphs(i[0])
            # t6 += time() -t5
            sample_graph = sample[0][0]
            sample_graph.ndata['c'] = sample_graph.ndata['c'].type(th.float32)
            sample_graph.edata['d'] = sample_graph.edata['d'].type(th.float32)
            sample_label = int(sample[1]['label'])
            sampe_seq_left = np.array(sample[1]['l'])
            sampe_seq_right = np.array(sample[1]['r'])
            batch_sample_list.append((sample_graph, sample_label, sampe_seq_left, sampe_seq_right))
        # t3 = time()
        data = self.data_preprocess(batch_sample_list)
        # print('total load time', time() - t1, "read time", t2 - t1, "total circle time", t3 - t2, "circle time",t6,)
        # q.put((data, Pa))
        return (data, Pa)
        # return batch_sample_list
    def collate(self, samples):
        graphs, labels, seq_left, seq_right = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        # print(len(seq_left))
        # print(len(seq_right))
        return batched_graph, torch.tensor(labels), torch.tensor(seq_left).type(th.float32), torch.tensor(
            seq_right).type(th.float32)
    def data_preprocess(self, data):
        data = Dl(data, batch_size=len(data), collate_fn=self.collate)
        graphs, labels, seq_left, seq_right = next(iter(data))
        return graphs, labels, seq_left, seq_right


class Data_permutation():
    def __init__(self, path, sample_size, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.total_sample_size = len(os.listdir(self.path)) * 1000  # 目录下所有样本的数量
        self.selected_idx = np.random.permutation(self.total_sample_size)[:self.sample_size]  # 被抽到的样本序列
        self.cur_count = 0

    @property
    def next_batch_idx(self):
        if self.cur_count + self.batch_size > self.sample_size:
            self.cur_count = 0
            p = np.random.permutation(self.sample_size)
            self.selected_idx = self.selected_idx[p]
        self.cur_idx = self.selected_idx[self.cur_count:self.cur_count + self.batch_size]
        self.cur_count += self.batch_size
        return self.cur_idx


class Data_non_permutation():
    def __init__(self, path, sample_size, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.total_sample_size = len(os.listdir(self.path)) * 1000  # 目录下所有样本的数量
        self.selected_idx = np.random.permutation(self.total_sample_size)[:self.sample_size]  # 被抽到的样本序列
        self.cur_count = 0

    @property
    def next_batch_idx(self):
        if self.cur_count + self.batch_size > self.sample_size:
            self.cur_count = 0
        self.cur_idx = self.selected_idx[self.cur_count:self.cur_count + self.batch_size]
        self.cur_count += self.batch_size
        return self.cur_idx


class Data_generator():
    def __init__(self, q, path, batch_size, train_sample_size, test_sample_size):
        self.data_reader = Data_reader(path)
        self.q = q
        self.path = path
        self.batch_size = batch_size
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.train_sample_max_count = train_sample_size // batch_size - 1
        self.test_sample_max_count = test_sample_size // batch_size - 1
        self.count = 0
        self.dp = Data_permutation(os.path.join(self.path, r'trainset'), self.train_sample_size,
                                   self.batch_size)  # 训练集的乱序
        self.dnp = Data_non_permutation(os.path.join(self.path, r'testset'), self.test_sample_size,
                                        self.batch_size)  # 测试集的顺序
        self.set = 1  # 1为训练集，0为测试集

    @property
    def func(self):
        return self.data_reader.next_batch

    def run(self):
        while True:
            if self.set:
                index = self.dp.next_batch_idx
                self.func(self.q,index,True)
                if self.count < self.train_sample_max_count:
                    self.count += 1
                else:
                    self.count =0
                    self.set = 0
            else:
                index = self.dnp.next_batch_idx
                self.func(self.q,index,False)
                if self.count < self.test_sample_max_count:
                    self.count += 1
                else:
                    self.count = 0
                    self.set = 1

if __name__ == '__main__':
    path = r'/home/yzy/PycharmProjects/trianing'
