'''baseGCN'''
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
from dgl import function as fn
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from torch.nn import init


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type='mean',
                 residual=True,
                 bias=False):
        super(GCN, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self._aggre_type = aggregator_type
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            # (n, d_in, 1)
            graph.srcdata['h'] = feat_src.unsqueeze(-1)
            # (n, d_in, d_out)
            graph.update_all(fn.copy_u('h', 'm'), self.reducer('m', 'neigh'))
            rst = graph.dstdata['neigh'].squeeze(-1)  # (n, d_out)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst

class Stretching_Potential_Loss(nn.Module):
    def __init__(self,r0):
        super(Stretching_Potential_Loss, self).__init__()
        self.r0 = r0

    def forward(self,in_embed,out_embed,ispos):
        dist = th.norm((out_embed-in_embed).view([-1,in_embed.shape[-1]]),p=2,dim=1)
        # loss = th.mul(th.square(dist-self.r0),th.log10(1000*weights))
        # loss = th.square(dist - self.r0)
        if ispos:
            loss = -F.logsigmoid(self.r0 - dist)
        else:
            loss = -F.logsigmoid(dist-self.r0)
        return loss

class Electrostatic_Potential_Loss(nn.Module):
    def __init__(self):
        super(Electrostatic_Potential_Loss, self).__init__()

    def forward(self, d,in_embed, out_embed):
        loss = 1/th.norm((in_embed-out_embed).view([-1,in_embed.shape[-1]]),p=2,dim=1)
        # loss = th.mul(d.float(),loss)
        return loss

class Loss(nn.Module):
    def __init__(self, lamda,r0):
        super(Loss, self).__init__()
        self.lamda = lamda
        self.spl = Stretching_Potential_Loss(r0)
        self.epl = Electrostatic_Potential_Loss()

    def forward(self, d,in_embed, out_embed,label):
        loss = label * self.spl(in_embed,out_embed) + (1-label) * self.epl(d,in_embed, out_embed)
        return loss.sum()

class EmbeddingModel(nn.Module):
    def __init__(self, nodes_nums, embed_size,r0,k):
        super(EmbeddingModel, self).__init__()
        self.k = k

        self.nodes_nums = nodes_nums
        self.embed_size = embed_size

        self.gcn = GCN(embed_size,embed_size)

        self.embed = nn.Embedding(self.nodes_nums, self.embed_size) #51836

        self.SPL_P = Stretching_Potential_Loss(r0)
        self.SPL_N = Stretching_Potential_Loss(r0)

    # def reset_parameters(self):
    #     gain = nn.init.calculate_gain('relu')
    #     nn.init.xavier_normal_(self.embed, gain=gain)

    def forward(self,graph, batched_center_word, batched_pos_word, batched_neg_word,graphdist):
        self.rst = self.gcn(graph,self.embed.weight)
        self.rst = self.gcn(graph,self.rst)

        center_word_embedding = self.rst[batched_center_word]
        pos_word_embedding = self.rst[batched_pos_word]
        neg_word_embedding = self.rst[batched_neg_word]

        pos_loss = self.SPL_P(center_word_embedding,pos_word_embedding,True).sum()
        neg_loss = self.SPL_N(center_word_embedding, neg_word_embedding,False).sum()

        return pos_loss,self.k*neg_loss

    def lookup_table(self):
        return self.rst.detach().cpu().numpy()

class Train():
    def __init__(self,batch_size,epoches,root_path):
        self.batch_size = batch_size
        self.root_path = root_path
        self.epoches = epoches
        self.date = time.strftime('%m_%d_%H_%M')
        self.graph = self.load_g()
        self.nodes_num = self.graph.nodes().shape[0]
        self.graphdist = self.load_graphdist()

    def get_graphdist(self,input_labels,neighbor_labels):
        #input_labels,neighbor_labels
        l = []
        for i,n in zip(input_labels,neighbor_labels):
            l.append(self.graphdist[i,n])
        l = np.concatenate(l)
        return l

    def load_graphdist(self):
        graphdist = np.load(os.path.join(self.root_path,'data',r'graphdist_mtx.npy'))
        return graphdist

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

source_file = pd.read_csv(r"/public/home/huanhuan/project/GTEx/output/01_merge_all_tissue_cis_sig_eQTL_hotspot_egene_idx.txt",sep='\t')
lookup_table =  np.load("/public/home/huanhuan/project/GTEx/output/model/skip_gram_date08_31_13_08_lookup_table.npy")
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

    def compute_dist(self):
        split_point = 30170
        lookup_table = np.load(r'E:\deeplearning\NLP\skipgram\model\date04_24_15_33_lookup_table.npy')
        mtx = np.zeros([split_point,lookup_table.shape[0]-30170])

        disease_embed = lookup_table[:split_point]
        gene_embed = lookup_table[split_point:]

        bar = tqdm(range(split_point))
        for i in bar:
            mtx[i] = np.linalg.norm(gene_embed - disease_embed[i],ord=2,axis=-1)
            bar.set_description(desc='fill out the mtx')

        test_adj = np.load(r'E:\deeplearning\NLP\data\test_adj.npy').astype(int)
        all_adj = np.load(r'E:\deeplearning\NLP\data\all_adj.npy').astype(int)

        sample_site = ~(test_adj ^ all_adj) +2
        prediction = mtx[sample_site==1]

        np.save(r'E:\deeplearning\NLP\skipgram\output\md1', prediction)

        prediction = -np.load(r'E:\deeplearning\NLP\skipgram\output\md1.npy')
        ground_truth = np.load(r'E:\deeplearning\NLP\graphgan\output\ground_truth.npy')
        cm(prediction,ground_truth)

    def train(self):
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        gpu = th.device('cuda:0')
        cpu = th.device('cpu')

        train_size,train_dataloader = self.load_data('train')
        val_size,val_dataloader = self.load_data('val')

        model = EmbeddingModel(self.nodes_num,256,8,1).to(gpu)

        train_graph = load_graphs(os.path.join(self.root_path,'data',r'train_graph.dgl'))[0][0].to(gpu)
        # val_graph = load_graphs(os.path.join(self.root_path,'data',r'val_graph.dgl'))[0][0].to(gpu)

        b1 = 0.5
        b2 = 0.999
        optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(b1, b2))

        best_loss = 2**32
        for epoch in range(1,self.epoches+1):
            train_bar = tqdm(train_dataloader)
            train_epoch_loss = 0
            train_spl, train_epl = 0,0
            cur_samples = 0
            model.train()
            for batched_center_word,batched_pos_word,batched_neg_word in train_bar:
                # graphdist = th.tensor(self.get_graphdist(batched_center_word,batched_neg_word)).cuda()
                batched_center_word = batched_center_word.long().to(gpu)
                batched_pos_word = batched_pos_word.long()[:,3:5].to(gpu)
                batched_neg_word = batched_neg_word.long().to(gpu)
                spl, epl = model(train_graph,batched_center_word,batched_pos_word,batched_neg_word,0)
                loss = spl + epl
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_cp = loss.item()
                train_epoch_loss += loss_cp
                train_spl += spl.data.item()
                train_epl += epl.data.item()
                cur_samples += self.batch_size
                train_bar.set_description(
                    f"training epoch {epoch} loss {train_epoch_loss / cur_samples:.4f} spl {train_spl / cur_samples:.4f} epl {train_epl / cur_samples:.4f}")
            train_epoch_loss /= train_size
            optimizer.zero_grad()
            val_bar = tqdm(val_dataloader)
            test_epoch_loss = 0
            cur_samples = 0
            test_spl,test_epl = 0,0
            model.eval()
            for batched_center_word,batched_pos_word,batched_neg_word in val_bar:
                # graphdist = th.tensor(self.get_graphdist(batched_center_word,batched_neg_word)).cuda()
                batched_center_word = batched_center_word.long().to(gpu)
                batched_pos_word = batched_pos_word.long()[:,3:5].to(gpu)
                batched_neg_word = batched_neg_word.long().to(gpu)
                spl, epl = model(train_graph,batched_center_word,batched_pos_word,batched_neg_word,0)
                loss = spl + epl
                loss_cp = loss.item()
                test_epoch_loss += loss_cp
                test_spl += spl.data.item()
                test_epl += epl.data.item()
                cur_samples += self.batch_size
                val_bar.set_description(
                    f"validating epoch {epoch} loss {test_epoch_loss / cur_samples:.4f} spl {test_spl / cur_samples:.4f} epl {test_epl / cur_samples:.4f}")
            test_epoch_loss /= val_size

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
                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                th.save(state,os.path.join(self.root_path,r'skipgram\model', rf'date{self.date}_model.pkl'))

if __name__ == '__main__':
    root_path =r'E:\deeplearning\NLP'
    t = Train(4096,1000,root_path)
    t.train()
