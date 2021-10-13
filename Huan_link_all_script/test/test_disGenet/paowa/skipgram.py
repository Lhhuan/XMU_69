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
from rst_pheno.homograph_word2vec import random_walk
from tqdm import tqdm
import os
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


class EmbeddingModel(nn.Module):
    def __init__(self, nodes_nums, embed_size):
        super(EmbeddingModel, self).__init__()

        self.nodes_nums = nodes_nums
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(self.nodes_nums, self.embed_size)  # 中心词权重矩阵
        self.out_embed = nn.Embedding(self.nodes_nums, self.embed_size)  # 背景词权重矩阵
        # self.embed = nn.Embedding(self.nodes_nums, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        '''
            input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]

            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels)  # [batch_size, 1, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (window * 2 * K), embed_size]

        input_embedding = input_embedding.transpose(-1,-2)  # [batch_size, embed_size, 1] # （b * m * n）* (b * n * k) ->（b * m * k）
        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]

        neg_dot = torch.bmm(neg_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1)  # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg
        return -loss

    def lookup_table(self):
        return self.in_embed.weight.detach().to(th.device('cpu')).numpy()

class Train():
    def __init__(self,batch_size,epoches,root_path):
        self.batch_size = batch_size
        self.root_path = root_path
        self.epoches = epoches
        self.date = time.strftime('%m_%d_%H_%M')
        self.graph = self.load_g()
        self.nodes_num = self.graph.nodes().shape[0]

    def load_g(self):
        graph = load_graphs(os.path.join(self.root_path,'data',r'undirectied_Graph.dgl'))[0][0]
        return graph

    def load_data(self,setname):
        assert setname in ['train','val','test']
        center_words,pos_words,neg_words = th.load(os.path.join(self.root_path,'data',setname+r'_sample.pth'))
        dataset = TensorDataset(center_words,pos_words,neg_words)
        sample_size = center_words.shape[0]
        if setname == 'train':
            dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=False, drop_last=True)
        return sample_size,dataloader

    def test(self,checkpoint_path):
        # 仅查看loss
        gpu = th.device('cuda:0')
        cpu = th.device('cpu')
        model = EmbeddingModel(self.nodes_num,256)
        checkpoint = th.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        model = model.to(gpu)

        test_size,test_dataloader = self.load_data('test')
        test_bar = tqdm(test_dataloader)
        train_epoch_loss = 0
        cur_samples = 0
        model.train()
        for batched_center_word,batched_pos_word,batched_neg_word in test_bar:
            batched_center_word = batched_center_word.long().to(gpu)
            batched_pos_word = batched_pos_word.long().to(gpu)
            batched_neg_word = batched_neg_word.long().to(gpu)
            loss = model(batched_center_word,batched_pos_word,batched_neg_word).mean()
            loss_cp = loss.item()
            train_epoch_loss+=loss_cp
            cur_samples += self.batch_size
            test_bar.set_description(f"loss {train_epoch_loss/cur_samples:.4f}")
        train_epoch_loss /= test_size

    def get_similarity(self,lookup_table,split_point):
        mtx = cosine_similarity(X=lookup_table[:split_point],Y=lookup_table[split_point:])
        return mtx

    def compute_acc(self,lookup_table,source_file):
        split_point = source_file['diseaseidx'].max() + 1
        similarity_mtx = self.get_similarity(lookup_table,split_point)
        undirected_graph = load_graphs(r'E:\deeplearning\NLP\data\undirectied_Graph.dgl')[0][0]
        undirected_adjmtx = undirected_graph.adjacency_matrix().to_dense().numpy()
        undirected_truth_table = undirected_adjmtx[:split_point,split_point:]
        del(undirected_adjmtx)

        test_graph = load_graphs(r'E:\deeplearning\NLP\data\test_graph.dgl')[0][0]
        test_adjmtx = test_graph.adjacency_matrix().to_dense().numpy()
        test_truth_table = test_adjmtx[:split_point,split_point:]
        del(test_adjmtx)

        neg_links = undirected_truth_table[undirected_truth_table == 0]
        pos_links = test_truth_table[test_truth_table == 1]
        predicted_neg_links = similarity_mtx[undirected_truth_table == 0]
        predicted_pos_links = similarity_mtx[test_truth_table == 1]

        fpr, tpr, thresholds = roc_curve(np.concatenate([pos_links,neg_links]), np.concatenate([predicted_pos_links,predicted_neg_links]), )
        optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

    def select_nodes(self,nums1=4,nums2=1):
        # nums1 表示样本个数
        # nums2 表示windowsize
        data = np.load(os.path.join(self.root_path,'data','test'+r'_sample.npz'))
        c = data['c'][:nums1]
        p = data['p'][:nums1,4-nums2:4+nums2]
        n = data['n'][:nums1]
        nodes = np.unique(np.concatenate([c.flatten(),p.flatten(),n.flatten()])).astype(np.int32)
        return nodes,c,p

    def save_coordinate(self,selected_nodes,model,epoch):
        selected_nodes = th.LongTensor(selected_nodes).cuda()
        in_embed = model.in_embed(selected_nodes).detach().cpu().numpy()
        out_embed = model.out_embed(selected_nodes).detach().cpu().numpy()
        np.save(os.path.join(self.root_path,'skipgram\coordinate',f'in_{epoch}.npy'),in_embed)
        np.save(os.path.join(self.root_path,'skipgram\coordinate',f'out_{epoch}.npy'),out_embed)

    def train(self):
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        gpu = th.device('cuda:0')
        cpu = th.device('cpu')

        train_size,train_dataloader = self.load_data('train')
        val_size,val_dataloader = self.load_data('val')

        model = EmbeddingModel(self.nodes_num,256).to(gpu)

        b1 = 0.5
        b2 = 0.999
        optimizer = optim.Adam(model.parameters(), lr=0.005, betas=(b1, b2))

        # nodes,c,p = self.select_nodes()

        best_loss = 2**16
        for epoch in range(1,self.epoches+1):
            train_bar = tqdm(train_dataloader)
            train_epoch_loss = 0
            cur_samples = 0
            model.train()
            for batched_center_word,batched_pos_word,batched_neg_word in train_bar:
                batched_center_word = batched_center_word.long().to(gpu)
                batched_pos_word = batched_pos_word.long().to(gpu)
                batched_neg_word = batched_neg_word.long().to(gpu)
                loss = model(batched_center_word,batched_pos_word,batched_neg_word).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_cp = loss.item()
                train_epoch_loss+=loss_cp
                cur_samples += self.batch_size
                train_bar.set_description(f"training epoch {epoch} loss {train_epoch_loss/cur_samples:.4f}")
            train_epoch_loss /= train_size

            val_bar = tqdm(val_dataloader)
            test_epoch_loss = 0
            cur_samples = 0
            model.eval()
            for batched_center_word,batched_pos_word,batched_neg_word in val_bar:
                batched_center_word = batched_center_word.long().to(gpu)
                batched_pos_word = batched_pos_word.long().to(gpu)
                batched_neg_word = batched_neg_word.long().to(gpu)
                loss = model(batched_center_word,batched_pos_word,batched_neg_word).sum()
                loss_cp = loss.item()
                test_epoch_loss+=loss_cp
                cur_samples += self.batch_size
                val_bar.set_description(f"validating epoch {epoch} loss {test_epoch_loss/cur_samples:.4f}")
            test_epoch_loss /= val_size

            # self.save_coordinate(nodes,model,epoch)

            if epoch == 1:
                pd.DataFrame(columns=['epoch','train_epoch_loss','test_epoch_loss']).to_csv(
                    os.path.join(self.root_path,r'skipgram\record', rf'date{self.date}_merics.txt'), sep='\t', index=None)
            pd.DataFrame([epoch,train_epoch_loss,test_epoch_loss]).T.to_csv(
                os.path.join(self.root_path,r'skipgram\record', rf'date{self.date}_merics.txt'), sep='\t', index=None,mode='a',header=None)
            if best_loss > test_epoch_loss:
                print(f"Now best model is epoch{epoch}")
                best_loss = test_epoch_loss
                lookup_table = model.lookup_table()
                np.save(os.path.join(self.root_path,r'skipgram\model', rf'date{self.date}_lookup_table.npy'),lookup_table)
                best_test_loss = test_epoch_loss
                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                th.save(state,os.path.join(self.root_path,r'skipgram\model',rf'date{self.date}_model.pkl'))

    def plot3d(self):
        in1 = np.load(r'E:\deeplearning\NLP\skipgram\coordinate\in_1.npy')
        nodes,c,p = self.select_nodes()
        c = c.repeat(p.shape[1]).reshape([-1,1])
        p = p.flatten().reshape([-1,1])
        c_idx = np.argwhere(nodes==c)[:,1]
        p_idx = np.argwhere(nodes==p)[:,1]
        c = in1[c_idx]
        p = in1[p_idx]

        import matplotlib.animation as animation

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        for cc,pp in zip(c,p):
            ax.plot([cc[0], pp[0],0,cc[0]], [cc[ 1], pp[1],0,cc[1]],[cc[2], pp[2],0,cc[2]])
        ax.set_xlim(-3,3)
        ax.set_ylim(-3,3)
        ax.set_zlim(-3,3)
        plt.show()

if __name__ == '__main__':
    root_path =r'E:\deeplearning\NLP'
    t = Train(4096,1000,root_path)
    t.train()

