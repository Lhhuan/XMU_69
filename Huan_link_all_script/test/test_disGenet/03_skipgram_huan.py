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


class EmbeddingModel(nn.Module):
    def __init__(self, nodes_nums, embed_size): #node_nums(num_embeddings (python:int)):词典的大小尺寸，比如总共出现5000个词，那就输入5000。此时index为（0-4999） embed_size(embedding_dim (python:int)）– 嵌入向量的维度，即用多少维来表示一个符号
        super(EmbeddingModel, self).__init__() #然后把类EmbeddingModel的对象self转换为类nn.Module的对象，然后“被转换”的类nn.Module对象调用自己的init函数
        self.nodes_nums = nodes_nums
        self.embed_size = embed_size
        self.in_embed = nn.Embedding(self.nodes_nums, self.embed_size)  # 中心词权重矩阵
        self.out_embed = nn.Embedding(self.nodes_nums, self.embed_size)  # 背景词权重矩阵
    def forward(self, input_labels, pos_labels, neg_labels): #前向传播
        ''' #多行注释
            input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]

            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels)  # [batch_size, 1, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (window * 2 * K), embed_size]
        input_embedding = input_embedding.transpose(-1,-2)  # [batch_size, embed_size, 1] # （b * m * n）* (b * n * k) ->（b * m * k）
        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1] #计算2个tensor的矩阵乘法
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)] #去掉第三軸
        neg_dot = torch.bmm(neg_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]
        log_pos = F.logsigmoid(pos_dot).sum(1)  # .sum()结果只为一个数，.sum(1)结果是一维的张量  #sum(1)求行和 
        log_neg = F.logsigmoid(neg_dot).sum(1) #logsigmoid 损失函数
        loss = log_pos + log_neg
        return -loss
    def lookup_table(self):
        return self.in_embed.weight.detach().to(th.device('cpu')).numpy()
#-----------------------------------------------------
def load_data(setname): 
    assert setname in ['train','val','test']
    center_words,pos_words,neg_words = th.load(os.path.join(root_path,setname+r'_sample.pth'))
    dataset = TensorDataset(center_words,pos_words,neg_words)
    sample_size = center_words.shape[0]
    if setname == 'train':
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True, drop_last=True) #shuffle 表示每一个epoch中训练样本的顺序是否相同，一般True
    else:
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False, drop_last=True)
    return sample_size,dataloader

class Train():
    def __init__(self,batch_size,epoches,root_path):
        self.batch_size = batch_size
        self.root_path = root_path
        self.epoches = epoches
        self.date = time.strftime('%m_%d_%H_%M')
        self.graph = self.load_g()
        self.nodes_num = self.graph.nodes().shape[0]

#------------------------------------


# graph = load_graphs("/public/home/huanhuan/test_disGenet/train_graph.dgl")[0][0]
graph = load_graphs("/public/home/huanhuan/test_disGenet/undirectied_Graph.dgl")[0][0]
nodes_num = graph.nodes().shape[0]
root_path= '/public/home/huanhuan/test_disGenet/'
batch_size = 1000
# nodes_num = 51836
epoch = 10
date = time.strftime('%m_%d_%H_%M')
#----------------------------------
    def train():
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # gpu = th.device('cuda:0')
        cpu = th.device('cpu')

        train_size,train_dataloader = load_data('train')
        val_size,val_dataloader = load_data('val')

        
        
        
        
        
         = EmbeddingModel(nodes_num,256).to(cpu)

        b1 = 0.5
        b2 = 0.999
        optimizer = optim.Adam(model.parameters(), lr=0.005, betas=(b1, b2)) #b1:一阶矩估计的指数衰减率（如 0.9),b2:二阶矩估计的指数衰减率（如 0.999）

        # nodes,c,p = select_nodes()

        best_loss = 2**16
        for epoch in range(1,epoches+1):
            train_bar = tqdm(train_dataloader)
            train_epoch_loss = 0
            cur_samples = 0
            model.train()
            for batched_center_word,batched_pos_word,batched_neg_word in train_bar:
                batched_center_word = batched_center_word.long().to(cpu)
                batched_pos_word = batched_pos_word.long().to(cpu)
                batched_neg_word = batched_neg_word.long().to(cpu)
                loss = model(batched_center_word,batched_pos_word,batched_neg_word).sum()
                optimizer.zero_grad()#对应d_weights = [0] * n 即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
                loss.backward() #loss.backward()对应d_weights = [d_weights[j] + (label[k] - h) * input[k][j] for j in range(n)]，即反向传播求梯度，梯度下降和损失函数的关系是：梯度下降是求解损失函数的一种最优化算法
                optimizer.step() #即更新所有参数，对应weights = [weights[k] + alpha * d_weights[k] for k in range(n)]
                loss_cp = loss.item()
                train_epoch_loss+=loss_cp
                cur_samples += batch_size
                train_bar.set_description(f"training epoch {epoch} loss {train_epoch_loss/cur_samples:.4f}")
            train_epoch_loss /= train_size

            val_bar = tqdm(val_dataloader)
            test_epoch_loss = 0
            cur_samples = 0
            model.eval()#模型验证
            for batched_center_word,batched_pos_word,batched_neg_word in val_bar:
                batched_center_word = batched_center_word.long().to(cpu)
                batched_pos_word = batched_pos_word.long().to(cpu)
                batched_neg_word = batched_neg_word.long().to(cpu)
                loss = model(batched_center_word,batched_pos_word,batched_neg_word).sum()
                loss_cp = loss.item()
                test_epoch_loss+=loss_cp
                cur_samples += batch_size
                val_bar.set_description(f"validating epoch {epoch} loss {test_epoch_loss/cur_samples:.4f}")
            test_epoch_loss /= val_size

            # save_coordinate(nodes,model,epoch)

            if epoch == 1:
                pd.DataFrame(columns=['epoch','train_epoch_loss','test_epoch_loss']).to_csv(
                    os.path.join(root_path,r'record', rf'date{date}_merics.txt'), sep='\t', index=None) #-------------r表示raw string，不识别转义, f表示format，用来格式化字符串
            pd.DataFrame([epoch,train_epoch_loss,test_epoch_loss]).T.to_csv(
                os.path.join(root_path,r'record', rf'date{date}_merics.txt'), sep='\t', index=None,mode='a',header=None)
            if best_loss > test_epoch_loss:
                print(f"Now best model is epoch{epoch}") 
                best_loss = test_epoch_loss
                lookup_table = model.lookup_table() #------------------------------------------------------------------------取weight
                np.save(os.path.join(root_path,r'model', rf'date{date}_lookup_table.npy'),lookup_table)
                best_test_loss = test_epoch_loss
                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                # state = model.state_dict()
                th.save(state,os.path.join(root_path,r'model',rf'date{date}_model.pkl'))

#---------------------------------

checkpoint_path ="/public/home/huanhuan/test_disGenet/model/date08_23_21_51_model.pkl" #model




    def test(checkpoint_path):#--------------
        # 仅查看loss
        # gpu = th.device('cuda:0') #-----------可以改成cpu?
        cpu = th.device('cpu')
        model = EmbeddingModel(nodes_num,256)
        checkpoint = th.load(checkpoint_path)
        model.load_state_dict(checkpoint['net'])
        # model.load_state_dict(checkpoint)
        model = model.to(cpu)

        test_size,test_dataloader = load_data('test')
        test_bar = tqdm(test_dataloader)
        train_epoch_loss = 0
        cur_samples = 0
        model.train()
        for batched_center_word,batched_pos_word,batched_neg_word in test_bar:
            batched_center_word = batched_center_word.long().to(cpu)
            batched_pos_word = batched_pos_word.long().to(cpu)
            batched_neg_word = batched_neg_word.long().to(cpu)
            loss = model(batched_center_word,batched_pos_word,batched_neg_word).mean()
            loss_cp = loss.item()
            train_epoch_loss+=loss_cp
            cur_samples += batch_size
            test_bar.set_description(f"loss {train_epoch_loss/cur_samples:.4f}")
        train_epoch_loss /= test_size

    def get_similarity(lookup_table,split_point):
        mtx = cosine_similarity(X=lookup_table[:split_point],Y=lookup_table[split_point:])
        return mtx

source_file = pd.read_csv(r"/public/home/huanhuan/test_disGenet/all_gene_disease_associations.tsv",sep='\t')


    def compute_acc(self,lookup_table,source_file):
        split_point = source_file['diseaseidx'].max() + 1
        similarity_mtx = get_similarity(lookup_table,split_point)
        # undirected_graph = load_graphs(r'E:\deeplearning\NLP\data\undirectied_Graph.dgl')[0][0]  #-----------------------哪里来？
        undirected_graph = load_graphs("/public/home/huanhuan/test_disGenet/undirectied_Graph.dgl")[0][0]
        undirected_adjmtx = undirected_graph.adjacency_matrix().to_dense().numpy()
        undirected_truth_table = undirected_adjmtx[:split_point,split_point:]
        del(undirected_adjmtx)

        test_graph = load_graphs(r'/public/home/huanhuan/test_disGenet/test_graph.dgl')[0][0]
        test_adjmtx = test_graph.adjacency_matrix().to_dense().numpy()
        test_truth_table = test_adjmtx[:split_point,split_point:]
        del(test_adjmtx)

        neg_links = undirected_truth_table[undirected_truth_table == 0]
        pos_links = test_truth_table[test_truth_table == 1]
        predicted_neg_links = similarity_mtx[undirected_truth_table == 0]
        predicted_pos_links = similarity_mtx[test_truth_table == 1]

        fpr, tpr, thresholds = roc_curve(np.concatenate([pos_links,neg_links]), np.concatenate([predicted_pos_links,predicted_neg_links]), )
        optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

#----------------------huan add
roc_auc = auc(fpr,tpr)
plt.figure()
lw=2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("hyh.png",dpi=600)#保存图片，dpi设置分辨率
plt.show()
#------------------------------


    # def select_nodes(self,nums1=4,nums2=1):
    #     # nums1 表示样本个数
    #     # nums2 表示windowsize
    #     data = np.load(os.path.join(root_path,'test'+r'_sample.npz'))
    #     c = data['c'][:nums1]
    #     p = data['p'][:nums1,4-nums2:4+nums2]
    #     n = data['n'][:nums1]
    #     nodes = np.unique(np.concatenate([c.flatten(),p.flatten(),n.flatten()])).astype(np.int32)
    #     return nodes,c,p

    # def save_coordinate(self,selected_nodes,model,epoch):
        
    #     selected_nodes = th.LongTensor(selected_nodes).cuda()
    #     in_embed = model.in_embed(selected_nodes).detach().cpu().numpy()
    #     out_embed = model.out_embed(selected_nodes).detach().cpu().numpy()
    #     np.save(os.path.join(self.root_path,'skipgram\coordinate',f'in_{epoch}.npy'),in_embed)
    #     np.save(os.path.join(self.root_path,'skipgram\coordinate',f'out_{epoch}.npy'),out_embed)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_data(setname): 
    assert setname in ['train','val','test']
    center_words,pos_words,neg_words = th.load(os.path.join(root_path,setname+r'_sample.pth'))
    dataset = TensorDataset(center_words,pos_words,neg_words)
    sample_size = center_words.shape[0]
    if setname == 'train':
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True, drop_last=True) #shuffle 表示每一个epoch中训练样本的顺序是否相同，一般True
    else:
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False, drop_last=True)
    return sample_size,dataloader



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

    def load_data(self,setname): #-----------输入怎么给?
        assert setname in ['train','val','test']
        center_words,pos_words,neg_words = th.load(os.path.join(self.root_path,'data',setname+r'_sample.pth'))
        dataset = TensorDataset(center_words,pos_words,neg_words)
        sample_size = center_words.shape[0]
        if setname == 'train':
            dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True, drop_last=True) #shuffle 表示每一个epoch中训练样本的顺序是否相同，一般True
        else:
            dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=False, drop_last=True)
        return sample_size,dataloader

    def test(self,checkpoint_path):  #--------------------
        # 仅查看loss
        gpu = th.device('cuda:0') #-----------可以改成cpu?
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
        train_epoch_loss /= test_size #-------------------------------------------------??????

    def get_similarity(self,lookup_table,split_point):
        mtx = cosine_similarity(X=lookup_table[:split_point],Y=lookup_table[split_point:])
        return mtx

    def compute_acc(self,lookup_table,source_file):
        split_point = source_file['diseaseidx'].max() + 1
        similarity_mtx = self.get_similarity(lookup_table,split_point)
        undirected_graph = load_graphs(r'E:\deeplearning\NLP\data\undirectied_Graph.dgl')[0][0]  #----哪里来？
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

    def select_nodes(nums1=4,nums2=1):
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
        optimizer = optim.Adam(model.parameters(), lr=0.005, betas=(b1, b2)) #b1:一阶矩估计的指数衰减率（如 0.9),b2:二阶矩估计的指数衰减率（如 0.999）

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
                    os.path.join(self.root_path,r'skipgram\record', rf'date{self.date}_merics.txt'), sep='\t', index=None) #-------------r, rf代表什么意思
            pd.DataFrame([epoch,train_epoch_loss,test_epoch_loss]).T.to_csv(
                os.path.join(self.root_path,r'skipgram\record', rf'date{self.date}_merics.txt'), sep='\t', index=None,mode='a',header=None)
            if best_loss > test_epoch_loss:
                print(f"Now best model is epoch{epoch}") #f？
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

