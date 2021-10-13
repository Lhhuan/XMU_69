from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx
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
import tqdm
import time
from torch.utils.data import DataLoader
from dgl.data.utils import load_graphs, save_graphs
from multiprocessing import Process, Value, Lock ,Queue
import Mydatareader

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, th.tensor(labels)


class GraphConv_node(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GraphConv_node, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.linear = nn.Linear(self.in_feats, self.out_feats)
        self.activation = activation

    def forward(self, node):
        w = self.linear(node.data['w'])
        w = self.activation(w)
        return {'w': w}

class GraphConv_nodes(nn.Module):
    def __init__(self, nweights,nbias, activation):
        super(GraphConv_nodes, self).__init__()
        self.weights = nweights
        self.bias = nbias
        self.activation = activation

    def forward(self, node):
        w = F.linear(node.data['w'],self.weights,self.bias)
        w = self.activation(w)
        return {'w': w}

class GraphConv_edge(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GraphConv_edge, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.linear = nn.Linear(self.in_feats, self.out_feats)
        self.activation = activation

    def forward(self, edge):
        h = self.linear(edge.data['h'])
        h = self.activation(h)
        return {'h': h}

class GraphConv_edges(nn.Module):
    def __init__(self, eweight,ebias, activation):
        super(GraphConv_edges, self).__init__()
        self.weights = eweight
        self.bias = ebias
        self.activation = activation

    def forward(self, edge):
        h = F.linear(edge.data['h'],self.weights,self.bias)
        h = self.activation(h)
        return {'h': h}

class GraphNN(nn.Module):
    def __init__(self, nodes_in_feats, edges_in_feats, out_feats, activation):
        super(GraphNN, self).__init__()
        self.GraphConv_node = GraphConv_node(nodes_in_feats, out_feats, activation)
        self.GraphConv_edge = GraphConv_edge(edges_in_feats, out_feats, activation)

    def forward(self, g, feature, efeature):
        # Initialize the node features with h.
        g.ndata['w'] = feature
        g.edata['h'] = efeature
        g.apply_edges(func=self.GraphConv_edge)
        g.apply_nodes(func=self.GraphConv_node)
        g.update_all(fn.u_mul_e('w', 'h', 'm'), fn.mean('m', 'w'))
        return g.ndata.pop('w')

class GraphNNs(nn.Module):
    def __init__(self, nweight, nbias, eweight,ebias, activation):
        super(GraphNNs, self).__init__()
        self.GraphConv_nodes = GraphConv_nodes(nweight, nbias, activation)
        self.GraphConv_edges = GraphConv_edges(eweight, ebias, activation)

    def forward(self, g, feature, efeature):
        # Initialize the node features with h.
        g.ndata['w'] = feature
        g.edata['h'] = efeature
        g.apply_edges(func=self.GraphConv_edges)
        g.apply_nodes(func=self.GraphConv_nodes)
        g.update_all(fn.u_mul_e('w', 'h', 'm'), fn.mean('m', 'w'))
        return g.ndata.pop('w')



class  Circle(nn.Module):
    def __init__(self, nodes_in_feats, edges_in_feats, hidden_dim,circles=1):
        super(Circle, self).__init__()
        self.circles = circles
        self.nweight = th.nn.Parameter(th.nn.init.xavier_normal_(th.rand([hidden_dim,hidden_dim],dtype=th.float32)))
        self.nbias = th.nn.Parameter(th.rand([hidden_dim],dtype=th.float32))
        self.eweight = th.nn.Parameter(th.nn.init.xavier_normal_(th.rand([hidden_dim,hidden_dim],dtype=th.float32)))
        self.ebias = th.nn.Parameter(th.rand([hidden_dim],dtype=th.float32))

        self.layers = nn.ModuleList([GraphNN(nodes_in_feats, edges_in_feats, hidden_dim, F.relu)])
        for i in range(1,circles):
            self.layers.append(GraphNNs(self.nweight, self.nbias, self.eweight, self.ebias, F.relu))


    def forward(self, g):
        for idx, conv in enumerate(self.layers):
            if idx == 0:
                w = g.ndata['c']
                h = g.edata['d']
                w = conv(g, w, h)
            else:
                w = conv(g, w, g.edata['h'])
        g.ndata['w'] = w
        hg = dgl.max_nodes(g, 'w')
        return hg

class graph_Embedding(nn.Module):
    def __init__(self,in_feats,out_feats):
        super(graph_Embedding, self).__init__()
        self.embed = nn.Linear(in_feats, out_feats)
    def forward(self,input):
        return self.embed(input)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.circle = Circle(2,1,32,2)
        self.embed = graph_Embedding(32,2)
    def forward(self,input):
        layer = self.circle(input)
        output = self.embed(layer)
        return output



# class Get_data():
#     def __init__(self,path,sample_size,batch_size):
#         self.q = Queue()
#         self.batch_size = batch_size
#         self.data_reader = Mydatareader.Data_reader(path,sample_size,self.batch_size)
#
#     @property
#     def get_data(self):
#         p = Process(target=self.data_reader.next_batch,)
#         p.start()
#         data = self.q.get()
#         data = DataLoader(data, batch_size=self.batch_size, shuffle=False,
#                              collate_fn=collate)
#         p.join()
#         return data

# data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
#                          collate_fn=collate)

# G = load_graphs(r'D:\PycharmProjects\pyproject\20190618\tmp\ctcf\motif\Graphs.dgl')
# g = []
# for idx, (i, j) in enumerate(zip(G[0], G[1]['label'])):
#     if idx < 100 or idx >= 19900:
#         i.ndata['c'] = i.ndata['c'].float()
#         i.edata['d'] = i.edata['d'].float()
#         g.append((i, int(j[1])))
# g_test = []
# for idx, (i, j) in enumerate(zip(G[0], G[1]['label'])):
#     if idx < 10005 and idx >= 9995:
#         i.ndata['c'] = i.ndata['c'].float()
#         i.edata['d'] = i.edata['d'].float()
#         g_test.append((i, int(j[1])))
if __name__ == '__main__':

    path = r'D:\PycharmProjects\pyproject\20190618\tmp\ctcf\motif\gcn'
    epochs = 10
    batch_size = 32
    sample_size = 1024
    # f = Mydatareader.Data_reader(path, 1024, 32)
    # q = Queue()
    # p = Process(target=f.next_batch, args=(q,))
    # p.start()
    # graphs, labels = q.get()
    # p.join()
    data = Mydatareader.Data_generator(path,sample_size,batch_size)
    # test_data_loader = DataLoader(g_test, batch_size=len(g_test), shuffle=False,
    #                          collate_fn=collate)

    model = Classifier(2, 1, 64, 2)
    loss_func = nn.CrossEntropyLoss() #会自动添加一层softmax
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # model.train()

    train_epoch_losses = []
    # test_epoch_losses = []
    os.chdir(r'D:\PycharmProjects\pyproject\20190618\tmp\ctcf\motif\gcn')

    for epoch in range(epochs):
        train_epoch_loss = 0
        for iters, (bg, label) in enumerate(data):
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()  # 求导
            optimizer.step()  # 更新参数
            train_epoch_loss += loss.detach().item()

        # itr = iter(test_data_loader)
        # test_sample,test_label = next(itr)
        # test_prediction = model(test_sample)
        # accuracy = int(th.eq(test_prediction.argmax(1),test_label).sum())/test_label.shape[0]
        # test_loss = loss_func(test_prediction, test_label)
        train_epoch_loss /= (iters + 1)
        # print('Epoch {}, loss {:.4f}, test_loss {:.4f}, test_acc {:.4f}'.format(epoch, train_epoch_loss, test_loss,accuracy))
        print('Epoch {}, loss {:.4f}'.format(epoch, train_epoch_loss,))
        # th.save(model, r'Epoch-{}, loss-{:.4f}, test_loss-{:.4f}, test_acc-{:.4f}.pkl'.format(epoch, train_epoch_loss, test_loss,accuracy))
        train_epoch_losses.append(train_epoch_loss)
        # test_epoch_losses.append(test_loss)

