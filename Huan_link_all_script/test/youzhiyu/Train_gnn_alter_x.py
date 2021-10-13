import torch
from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx
import sys
import torch as th
from torch import nn
from torch.nn import init
from dgl import function as fn
from dgl.nn.pytorch.utils import Identity
import dgl
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader as Dl
from dgl.data.utils import load_graphs, save_graphs
from multiprocessing import Process, Value, Lock, Queue
import Mydatareader_alter_alter_alter_x as Mydatareader, gcn5
import cnn
import torch as th


# def collate(samples):
#     graphs, labels = samples[0],samples[1]
#     batched_graph = dgl.batch(graphs)
#     return batched_graph, torch.tensor(labels)


class Train():
    def __init__(self, path, train_sample_size, test_sample_size, batch_size, epochs):
        self.path = path
        self.epochs = epochs
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.batch_size = batch_size
        self.q = Queue(1)  # 待输入模型的样本队列

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

    def accuracy(self, prediction, label):
        acc = (label.shape[0] - int(th.sum(prediction.argmax(1) ^ label))) / label.shape[0]
        return acc

    def load_data(self):

        process = Mydatareader.Data_generator(self.q,  self.path, self.batch_size,self.train_sample_size, self.test_sample_size)
        P = Process(target = process.run, args=())
        P.start()

    def train(self):
        global train_epoch_losses,test_epoch_losses,train_epoch_acces,test_epoch_acces,tmp_train,tmp_test,graphs,model,labels
        # gpu = torch.device("cuda:0")
        # cpu = torch.device("cpu")
        model = gcn5.Classifier(2,1,32,2,10)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_epoch_losses = []
        train_epoch_acces = []
        test_epoch_losses = []
        test_epoch_acces = []
        self.data_reader = Mydatareader.Data_reader(self.path,self.train_sample_size,self.test_sample_size,self.batch_size)
        tmp_train =[]
        tmp_test =[]
        for epoch in tqdm(range(self.epochs)):
            train_epoch_loss = 0
            train_epoch_acc = 0
            model.train()
            for iters in tqdm(range(self.train_sample_size // self.batch_size)):
                ((graphs, labels, seq_left, seq_right),Pa) = self.data_reader.next_batch(1) #Pa是说所数据集
                # t0 = time()
                # ((batch_sample_list), Pa) = self.q.get()
                # t1 = time()
                # graphs, labels, seq_left, seq_right= self.data_preprocess(batch_sample_list)
                # t2 = time()
                # seq_left, seq_right = seq_left.to(gpu), seq_right.to(gpu)
                prediction = model(graphs)
                loss = loss_func(prediction, labels)
                optimizer.zero_grad()
                loss.backward()  # 求导
                optimizer.step()  # 更新参数
                train_epoch_loss += loss.detach().item()
                train_epoch_acc += self.accuracy(prediction.data, labels)
                # print('train time',time()-t2,'total wait time',t1 -t0,'process time',t2-t1,)
            train_epoch_loss /= (iters + 1)
            train_epoch_losses.append(train_epoch_loss)
            train_epoch_acc /= (iters + 1)
            train_epoch_acces.append(train_epoch_acc)
            test_epoch_loss = 0
            test_epoch_acc = 0
            # print('testset')
            model.eval()
            for iters in tqdm(range(self.test_sample_size // self.batch_size)):
                ((graphs, labels, seq_left, seq_right),Pa) = self.data_reader.next_batch(2)
                # ((batch_sample_list), Pa) = self.q.get()
                # graphs, labels, seq_left, seq_right = self.data_preprocess(batch_sample_list)
                # seq_left, seq_right = seq_left.to(gpu), seq_right.to(gpu)
                prediction = model(graphs)
                loss = loss_func(prediction, labels)
                test_epoch_loss += loss.detach().item()
                test_epoch_acc += self.accuracy(prediction.data, labels)
            test_epoch_loss /= (iters + 1)
            test_epoch_losses.append(train_epoch_loss)
            test_epoch_acc /= (iters + 1)
            test_epoch_acces.append(test_epoch_acc)
            # print(self.q_idx.qsize())
            print('Epoch {}, train-loss {:.4f}, test-loss {:.4f}, train-acc {:.4f}, test-acc {:.4f}'.format(epoch,
                                                                                                            train_epoch_loss,
                                                                                                            test_epoch_loss,
                                                                                                            train_epoch_acc,
                                                                                                            test_epoch_acc))
        print("end")
        th.cuda.empty_cache()


if __name__ == '__main__':
    path = r'/home/yzy/PycharmProjects/trianing'

    T = Train(path, train_sample_size=50000, test_sample_size=5000, batch_size=500, epochs=200)
    T.train()

    # dp = Mydatareader.Data_permutation(path,20000,500)
    # threads_num = 6
    # q_idx = Queue(1)
    # q_idx.put(dp)
    # q = Queue(threads_num)
    # Data_readers = [Mydatareader.Data_generator(q,q_idx,path) for i in range(threads_num)]
    # P = []
    # for i in Data_readers:
    #     P.append(Process(target=i.run,args=()))
    # for p in P:
    #     p.start()
    # t1 = time.time()
    # for i in range(10):
    #     print(q.get())
    # t2 = time.time()
    # print(t2-t1)
    # q.close()
    # q_idx.close()
    # print('done')
