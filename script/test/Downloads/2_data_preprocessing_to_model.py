#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from copy import deepcopy
import warnings 
import os
from sklearn.model_selection import KFold
import json
warnings.filterwarnings("ignore") 
import random
random.seed(1234)
np.random.seed(1234)


# In[2]:


def obtain_data(task, isbalance, balance):
    
    pwd = '/home/chujunyi/4_GNN/GAEMDA-miRNA-disease/0_data/'
    if isbalance:
        node_feature_label = pd.read_csv(pwd + 'node_feature_label.csv', index_col = 0)
    else:
        node_feature_label = pd.read_csv(pwd + 'node_feature_label__nobalance.csv', index_col = 0)
    
    train_test_id_idx = np.load('/home/chujunyi/4_GNN/GraphSAINT/miRNA_disease_data/task_' + task + balance + '__testlabel0_knn_edge_train_test_index_all.npz', allow_pickle = True)
    train_index_all = train_test_id_idx['train_index_all']
    test_index_all = train_test_id_idx['test_index_all']
    
    num_node = node_feature_label.shape[0]
    node_feat = node_feature_label.iloc[:, 3:]
    label = node_feature_label['label']

    mirna_ids = list(set(node_feature_label['miRNA']))
    disease_ids = list(set(node_feature_label['disease']))
    random.shuffle(mirna_ids)
    random.shuffle(disease_ids)
    print('# miRNA = {} | Disease = {}'.format(len(mirna_ids), len(disease_ids)))
    
    mirna_test_num = int(len(mirna_ids) / 5)
    disease_test_num = int(len(disease_ids) / 5)
    print('# Test: miRNA = {} | Disease = {}'.format(mirna_test_num, disease_test_num))
    
    return node_feature_label, num_node, node_feat, label, mirna_ids, disease_ids, train_index_all, test_index_all


# In[25]:


def generate_graphsaint_data(task, train_index_all, test_index_all, node_feat, n_neigh, label, num_node, balance):

    fold = 0
    for train_idx, test_idx in zip(train_index_all, test_index_all): #train_index与test_index为下标
        # read knn_graph
        pwd = '/home/chujunyi/4_GNN/GAEMDA-miRNA-disease/0_data/'
        knn_graph_file = 'task_' + task + balance + '__testlabel0_knn' + str(n_neigh) + 'neighbors_edge__fold' + str(fold) + '.npz'
        knn_neighbors_graph = sp.load_npz(pwd + knn_graph_file)

        edge_src_dst = knn_neighbors_graph.nonzero()
        #print(edge_src_dst)
        
        # save dir
        save_dir = '/home/chujunyi/4_GNN/GraphSAINT/miRNA_disease_data/task_' + task + balance + '__testlabel0_' + str(n_neigh) + 'knn_edge_fold' + str(fold) + '/'

        try:
            os.mkdir(save_dir)
        except OSError as error:
            print(error, save_dir)

        # feats.npy，不需要自己标准化！因为在utils.py中的load_data中有标准化的步骤哦！
        feats = np.array(node_feat)
        np.save(save_dir + 'feats.npy', feats)
        
        try:
            train_idx, test_idx = train_idx.tolist(), test_idx.tolist()
        except:
            train_idx, test_idx = train_idx, test_idx
            
        # role.json
        role = dict()
        role['tr'] = train_idx
        role['va'] = test_idx
        role['te'] = test_idx
        with open(save_dir + 'role.json','w') as f:
            json.dump(role, f)

        # class_map.json
        y = np.array(label)
        class_map = dict()
        for i in range(num_node):
            class_map[str(i)] = y[i].tolist()
        with open(save_dir + 'class_map.json', 'w') as f:
            json.dump(class_map, f)

        # adj_*.npz
        train_idx_set = set(train_idx)
        test_idx_set = set(test_idx)
        
        row_full, col_full = edge_src_dst[0], edge_src_dst[1]
        
        row_train = []
        col_train = []
        row_val = []
        col_val = []
        for i in tqdm(range(row_full.shape[0])):
            if row_full[i] in train_idx_set and col_full[i] in train_idx_set:
                row_train.append(row_full[i])
                col_train.append(col_full[i])
            if row_full[i] in test_idx_set and col_full[i] in test_idx_set:
                row_val.append(row_full[i])
                col_val.append(col_full[i])

        row_train = np.array(row_train)
        col_train = np.array(col_train)
        row_val = np.array(row_val)
        col_val = np.array(col_val)
        dtype = np.bool

        adj_full = sp.coo_matrix(
            (
                np.ones(row_full.shape[0], dtype=dtype),
                (row_full, col_full),
            ),
            shape=(num_node, num_node)
        ).tocsr()

        adj_train = sp.coo_matrix(
            (
                np.ones(row_train.shape[0], dtype=dtype),
                (row_train, col_train),
            ),
            shape=(num_node, num_node)
        ).tocsr()

        adj_val = sp.coo_matrix(
            (
                np.ones(row_val.shape[0], dtype=dtype),
                (row_val, col_val),
            ),
            shape=(num_node, num_node)
        ).tocsr()

        print('adj_full  num edges:', adj_full.nnz)
        print('adj_val   num edges:', adj_val.nnz)
        print('adj_train num edges:', adj_train.nnz)
        sp.save_npz(save_dir + 'adj_full.npz', adj_full)
        sp.save_npz(save_dir + 'adj_train.npz', adj_train)
        sp.save_npz(save_dir + 'adj_val.npz', adj_val) # adj_val not used in GraphSAINT source code

        fold += 1
    
    print('--Complete--', fold)
    return feats, role, class_map, adj_full, adj_train, adj_val, edge_src_dst


# In[32]:


def run(task, isbalance):
    
    if isbalance:
        balance = ''
    else:
        balance = '__nobalance'

    for n_neigh in [1, 3, 5, 7, 10, 15]:

        node_feature_label, num_node, node_feat, label, mirna_ids, disease_ids, train_index_all, test_index_all = obtain_data(task, 
                                                                                                                              isbalance,
                                                                                                                              balance)
        feats, role, class_map, adj_full, adj_train, adj_val, edge_src_dst = generate_graphsaint_data(task, 
                                                                                        train_index_all, 
                                                                                        test_index_all, 
                                                                                        node_feat, 
                                                                                        n_neigh, 
                                                                                        label, 
                                                                                        num_node, 
                                                                                        balance)
    return node_feature_label, num_node, node_feat, label, mirna_ids, disease_ids, train_index_all, test_index_all,     feats, role, class_map, adj_full, adj_train, adj_val, edge_src_dst


# # RUN nobalance

# In[34]:


for task in ['Tp']:
    node_feature_label, num_node, node_feat, label, mirna_ids, disease_ids, train_index_all, test_index_all,     feats, role, class_map, adj_full, adj_train, adj_val, edge_src_dst = run(task = task, isbalance = False)


# # RUN balance

# In[15]:


# for n_neigh in [1, 3, 5, 7, 15]: 
node_feature_label, num_node, node_feat, label, mirna_ids, disease_ids, train_index_all, test_index_all, feats, role, class_map, adj_full, adj_train, adj_val, edge_src_dst = run(task = 'Tm', isbalance = True)


# In[16]:


# for n_neigh in [1, 3, 5, 7, 15]: 
node_feature_label, num_node, node_feat, label, mirna_ids, disease_ids, train_index_all, test_index_all, feats, role, class_map, adj_full, adj_train, adj_val, edge_src_dst = run(task = 'Td', isbalance = True)


# In[19]:


# for n_neigh in [1, 3, 5, 7, 15]: 
node_feature_label, num_node, node_feat, label, mirna_ids, disease_ids, train_index_all, test_index_all, feats, role, class_map, adj_full, adj_train, adj_val, edge_src_dst = run(task = 'Tp', isbalance = True)

