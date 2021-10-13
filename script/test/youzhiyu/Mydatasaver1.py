import dgl
import torch as th
import os
from torch.utils.data import DataLoader
from dgl.data.utils import load_graphs, save_graphs
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib as mpl


class Data_saver():
    def __init__(self, sample_size, save_path):
        self.sample_size = sample_size
        self.save_path = save_path
        self.one_hot_dic = {
            'A': [0, 0, 0, 1],
            'a': [0, 0, 0, 1], 'T': [0, 0, 1, 0],
            't': [0, 0, 1, 0], 'C': [0, 1, 0, 0],
            'c': [0, 1, 0, 0], 'G': [1, 0, 0, 0],
            'g': [1, 0, 0, 0], 'N': [0.25, 0.25, 0.25, 0.25],
            'n': [0.25, 0.25, 0.25, 0.25],
        }
        self.mao = pd.read_csv(r'/home/yzy/PycharmProjects/sourcedata/ctcf_MAO.csv', sep='\t')
        self.mao_pretreat()
        self.test_neg = pd.read_csv(
            r'/home/yzy/PycharmProjects/data/%s_neg.csv' % self.save_path.split('/')[-1][:-3], header=None,
            sep='\t')[:sample_size // 2]
        self.test_pos = pd.read_csv(
            r'/home/yzy/PycharmProjects/data/%s_pos.csv' % self.save_path.split('/')[-1][:-3], header=None,
            sep='\t')[:sample_size // 2]
        self.bedfile()
        self.hic = pd.concat([self.test_pos, self.test_neg])
        self.labels = np.concatenate([np.array([1] * (sample_size // 2)), np.array([0] * (sample_size // 2))])
        self.sc = {}  # p & counts*dir
        for i in range(1, 23):
            chr = self.mao[self.mao['chr'] == i].copy()
            self.sc[i] = chr[['site', 'counts', 'dir', 'p']]
        for i in range(1, 23):
            self.sc[i]['counts'] = self.sc[i]['counts'] * self.sc[i]['dir']
        self.dist = {}  # distance
        for i in range(1, 23):
            chr = self.mao[self.mao['chr'] == i].copy()
            self.dist[i] = np.log10(chr['site'].values[1:] - chr['site'].values[:-1])
        self.cur_count = 0
        print("start binary_search")
        self.binary_search()
        # self.save_graph()
    def one_hot(self,stri):
        new = []
        for i in stri:
            new.append(self.one_hot_dic.get(i, [0, 0, 0, 0]))
        return np.array(new)
    def bedfile(self, size=750):
        self.test_pos.to_csv(r'/home/yzy/PycharmProjects/data/tmp_pos.csv', header=None, sep='\t', index=None)
        self.test_neg.to_csv(r'/home/yzy/PycharmProjects/data/tmp_neg.csv', header=None, sep='\t', index=None)
        os.system(r"awk -v OFS='\t' '{print $1,$2-%s,$2+%s}' /home/yzy/PycharmProjects/data/tmp_pos.csv > /home/yzy/PycharmProjects/data/tmp_pos_l.bed" % (size // 2, size // 2))
        os.system(r"awk -v OFS='\t' '{print $1,$3-%s,$3+%s}' /home/yzy/PycharmProjects/data/tmp_pos.csv > /home/yzy/PycharmProjects/data/tmp_pos_r.bed" % (size // 2, size // 2))
        os.system(r"awk -v OFS='\t' '{print $1,$2-%s,$2+%s}' /home/yzy/PycharmProjects/data/tmp_neg.csv > /home/yzy/PycharmProjects/data/tmp_neg_l.bed" % (size // 2, size // 2))
        os.system(r"awk -v OFS='\t' '{print $1,$3-%s,$3+%s}' /home/yzy/PycharmProjects/data/tmp_neg.csv > /home/yzy/PycharmProjects/data/tmp_neg_r.bed" % (size // 2, size // 2))
        tmp_pos_l = pd.read_csv(r'/home/yzy/PycharmProjects/data/tmp_pos_l.bed', header=None, sep='\t')
        tmp_pos_r = pd.read_csv(r'/home/yzy/PycharmProjects/data/tmp_pos_r.bed', header=None, sep='\t')
        tmp_neg_l = pd.read_csv(r'/home/yzy/PycharmProjects/data/tmp_neg_l.bed', header=None, sep='\t')
        tmp_neg_r = pd.read_csv(r'/home/yzy/PycharmProjects/data/tmp_neg_r.bed', header=None, sep='\t')
        left = pd.concat([tmp_pos_l, tmp_neg_l])
        right = pd.concat([tmp_pos_r, tmp_neg_r])
        left.to_csv(r'/home/yzy/PycharmProjects/data/tmp_left.bed', header=None, sep='\t', index=None)
        right.to_csv(r'/home/yzy/PycharmProjects/data/tmp_right.bed', header=None, sep='\t', index=None)
        os.system(
            r"bedtools getfasta -fi /home/yzy/PycharmProjects/sourcedata/genome.fa -bed /home/yzy/PycharmProjects/data/tmp_left.bed -fo /home/yzy/PycharmProjects/data/tmp_left.txt")
        os.system(
            r"bedtools getfasta -fi /home/yzy/PycharmProjects/sourcedata/genome.fa -bed /home/yzy/PycharmProjects/data/tmp_right.bed -fo /home/yzy/PycharmProjects/data/tmp_right.txt")
        left = pd.read_csv(r'/home/yzy/PycharmProjects/data/tmp_left.txt', header=None, sep='\t')
        right = pd.read_csv(r'/home/yzy/PycharmProjects/data/tmp_right.txt', header=None, sep='\t')
        self.left = left.iloc[[2 * x + 1 for x in range(int(left.shape[0] // 2))]]
        self.right = right.iloc[[2 * x + 1 for x in range(int(right.shape[0] // 2))]]
    def mao_pretreat(self):
        self.mao['counts'] /= 10
        self.mao['p'] /= 1000
    def binary_search(self):
        '''找到hic两个位点在ctcf中的位置'''
        lr = []  # hic的left和right的idx
        for idx, it in tqdm(self.hic.iterrows()):
            SC = self.sc[it[0]]
            site = SC['site'].values
            tmp = [it[0]]
            for ix, i in enumerate([it[1], it[2]]):
                left_idx = 0
                right_idx = SC.shape[0]
                while right_idx - left_idx > 1:
                    mid_idx = (left_idx + right_idx) // 2
                    if i > site[mid_idx]:
                        left_idx = mid_idx
                    elif i < site[mid_idx]:
                        right_idx = mid_idx
                    else:
                        right_idx = mid_idx
                        left_idx = mid_idx
                if ix == 0:
                    tmp.append(np.clip(left_idx - 1, 0, SC.shape[0] - 1))
                else:
                    tmp.append(np.clip(right_idx + 1, 0, SC.shape[0] - 1))
            lr.append(tmp)
        self.lr = np.array(lr)
    def next_graph(self, it):
        # self.cur_chr = it[0]
        G = nx.path_graph(it[2] - it[1] + 1)
        G = dgl.DGLGraph(G)
        G.ndata['c'] = self.sc[it[0]][['counts', 'p']].values[it[1]:it[2] + 1]
        G.ndata['c'] =G.ndata['c'].type(th.float32)
        G.edata['d'] = self.dist[it[0]][it[1]:it[2]].repeat(2).reshape([-1, 1])
        G.edata['d'] = G.edata['d'].type(th.float32)
        return G
    def save_graph(self, ):
        os.chdir(self.save_path)
        for idx, (label, it) in tqdm(enumerate(zip(self.labels, self.lr))):
            if idx % 1000 == 0:
                directory = idx // 1000
                os.mkdir(r'%s' % directory)
            cur_chr = it[0]
            graph = self.next_graph(it)
            save_graphs(os.path.join(r'%s' % directory, r'graph%s_chr%s_%s.dgl' % (idx, cur_chr, label)), graph,
                        {'label': th.tensor(label)})
    def save_graph_seq(self, ):
        os.chdir(self.save_path)
        for idx, (label, it) in tqdm(enumerate(zip(self.labels, self.lr))):
            if idx % 1000 == 0:
                directory = idx // 1000
                os.mkdir(r'%s' % directory)
            cur_chr = it[0]
            graph = self.next_graph(it)
            oh_l = np.expand_dims(self.one_hot(self.left.iloc[idx][0]),0)
            oh_r = np.expand_dims(self.one_hot(self.right.iloc[idx][0]),0)
            save_graphs(os.path.join(r'%s' % directory, r'graph%s_chr%s_%s.dgl' % (idx, cur_chr, label)), graph,
                        {'label': th.tensor(label),'l':th.tensor(oh_l),'r':th.tensor(oh_r)})

if __name__ == "__main__":

    path = r'/home/yzy/PycharmProjects/trianing/testset'
    path = r'/home/yzy/PycharmProjects/cnn_trianing/testset'
    ds = Data_saver(500000,path)
    ds.save_graph_seq()