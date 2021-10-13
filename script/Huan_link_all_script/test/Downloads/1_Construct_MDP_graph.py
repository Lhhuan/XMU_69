#!/usr/bin/env python
# coding: utf-8

# In[9]:


import warnings
warnings.filterwarnings("ignore")


# In[11]:


import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
import torch as th

import dgl
from dgl.data.utils import download, extract_archive, get_download_dir

from itehttp://121.192.180.20:8888/notebooks/code/1_Construct%20MDP%20graph.ipynb#rtools import product
from collections import Counter
from copy import deepcopy
from sklearn.model_selection import KFold
from tqdm import tqdm

import random
random.seed(1234)
np.random.seed(1234)


# In[3]:


def load_data(directory):
    D_SSM1 = np.loadtxt(directory + '/D_SSM1.txt')
    D_SSM2 = np.loadtxt(directory + '/D_SSM2.txt')
    D_GSM = np.loadtxt(directory + '/D_GSM.txt')
    M_FSM = np.loadtxt(directory + '/M_FSM.txt')
    M_GSM = np.loadtxt(directory + '/M_GSM.txt')
    D_SSM = (D_SSM1 + D_SSM2) / 2

    ID = np.zeros(shape=(D_SSM.shape[0], D_SSM.shape[1]))
    IM = np.zeros(shape=(M_FSM.shape[0], M_FSM.shape[1]))
    for i in range(D_SSM.shape[0]):
        for j in range(D_SSM.shape[1]):
            if D_SSM[i][j] == 0:
                ID[i][j] = D_GSM[i][j]
            else:
                ID[i][j] = D_SSM[i][j]
    for i in range(M_FSM.shape[0]):
        for j in range(M_FSM.shape[1]):
            if M_FSM[i][j] == 0:
                IM[i][j] = M_GSM[i][j]
            else:
                IM[i][j] = M_FSM[i][j]
                
    ID = pd.DataFrame(ID).reset_index()
    IM = pd.DataFrame(IM).reset_index()
    ID.rename(columns = {'index':'id'}, inplace = True)
    IM.rename(columns = {'index':'id'}, inplace = True)
    ID['id'] = ID['id'] + 1
    IM['id'] = IM['id'] + 1
    
    return ID, IM


# In[4]:


def sample(directory, random_seed):
    all_associations = pd.read_csv(directory + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)

    return sample_df


# In[5]:


def obtain_data(directory, isbalance):
    ID, IM = load_data(directory)
    
    if isbalance:
        dtp = sample(directory, random_seed = 1234)
    else:
        dtp = pd.read_csv(directory + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label'])
        
    mirna_ids = list(set(dtp['miRNA']))
    disease_ids = list(set(dtp['disease']))
    random.shuffle(mirna_ids)
    random.shuffle(disease_ids)
    print('# miRNA = {} | Disease = {}'.format(len(mirna_ids), len(disease_ids)))

    mirna_test_num = int(len(mirna_ids) / 5)
    disease_test_num = int(len(disease_ids) / 5)
    print('# Test: miRNA = {} | Disease = {}'.format(mirna_test_num, disease_test_num))
    
    knn_x = pd.merge(pd.merge(dtp, ID, left_on = 'disease', right_on = 'id'), IM, left_on = 'miRNA', right_on = 'id')
    label = dtp['label']
    knn_x.drop(labels = ['miRNA', 'disease', 'label', 'id_x', 'id_y'], axis = 1, inplace = True)
    assert ID.shape[0] + IM.shape[0] == knn_x.shape[1]
    print(knn_x.shape, Counter(label))
    
    return ID, IM, dtp, mirna_ids, disease_ids, mirna_test_num, disease_test_num, knn_x, label


# In[6]:


def generate_task_Tp_train_test_idx(knn_x):
    kf = KFold(n_splits = 5, shuffle = True, random_state = 1234)

    train_index_all, test_index_all, n = [], [], 0
    train_id_all, test_id_all = [], []
    fold = 0
    for train_idx, test_idx in tqdm(kf.split(knn_x)): #train_index与test_index为下标
        print('-------Fold ', fold)
        train_index_all.append(train_idx) 
        test_index_all.append(test_idx)

        train_id_all.append(np.array(dtp.iloc[train_idx][['miRNA', 'disease']]))
        test_id_all.append(np.array(dtp.iloc[test_idx][['miRNA', 'disease']]))

        print('# Pairs: Train = {} | Test = {}'.format(len(train_idx), len(test_idx)))
        fold += 1
    return train_index_all, test_index_all, train_id_all, test_id_all


# In[7]:


def generate_task_Tm_Td_train_test_idx(item, ids, dtp):
    
    test_num = int(len(ids) / 5)
    
    train_index_all, test_index_all = [], []
    train_id_all, test_id_all = [], []
    
    for fold in range(5):
        print('-------Fold ', fold)
        if fold != 4:
            test_ids = ids[fold * test_num : (fold + 1) * test_num]
        else:
            test_ids = ids[fold * test_num :]

        train_ids = list(set(ids) ^ set(test_ids))
        print('# {}: Train = {} | Test = {}'.format(item, len(train_ids), len(test_ids)))

        test_idx = dtp[dtp[item].isin(test_ids)].index.tolist()
        train_idx = dtp[dtp[item].isin(train_ids)].index.tolist()
        random.shuffle(test_idx)
        random.shuffle(train_idx)
        print('# Pairs: Train = {} | Test = {}'.format(len(train_idx), len(test_idx)))
        assert len(train_idx) + len(test_idx) == len(dtp)

        train_index_all.append(train_idx) 
        test_index_all.append(test_idx)
        
        train_id_all.append(train_ids)
        test_id_all.append(test_ids)
        
    return train_index_all, test_index_all, train_id_all, test_id_all


# # KNN

# In[8]:


from sklearn.neighbors import KNeighborsClassifier


# In[9]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report


# In[10]:


def generate_knn_graph_save(knn_x, label, n_neigh, train_index_all, test_index_all, pwd, task, balance):
    
    fold = 0
    for train_idx, test_idx in zip(train_index_all, test_index_all): 
        print('-------Fold ', fold)
        
        knn_y = deepcopy(label)
        knn_y[test_idx] = 0
        print('Label: ', Counter(label))
        print('knn_y: ', Counter(knn_y))

        knn = KNeighborsClassifier(n_neighbors = n_neigh)
        knn.fit(knn_x, knn_y)

        knn_y_pred = knn.predict(knn_x)
        knn_y_prob = knn.predict_proba(knn_x)
        knn_neighbors_graph = knn.kneighbors_graph(knn_x, n_neighbors = n_neigh)

        prec_reca_f1_supp_report = classification_report(knn_y, knn_y_pred, target_names = ['label_0', 'label_1'])
        tn, fp, fn, tp = confusion_matrix(knn_y, knn_y_pred).ravel()

        pos_acc = tp / sum(knn_y)
        neg_acc = tn / (len(knn_y_pred) - sum(knn_y_pred)) # [y_true=0 & y_pred=0] / y_pred=0
        accuracy = (tp+tn)/(tn+fp+fn+tp)

        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1 = 2*precision*recall / (precision+recall)

        roc_auc = roc_auc_score(knn_y, knn_y_prob[:, 1])
        prec, reca, _ = precision_recall_curve(knn_y, knn_y_prob[:, 1])
        aupr = auc(reca, prec)

        print('acc={:.4f}|precision={:.4f}|recall={:.4f}|f1={:.4f}|auc={:.4f}|aupr={:.4f}|pos_acc={:.4f}|neg_acc={:.4f}'.format(accuracy, precision, recall, f1, roc_auc, aupr, pos_acc, neg_acc))
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: ', Counter(knn_y_pred))
        print('y_true: ', Counter(knn_y))
#         print('knn_score = {:.4f}'.format(knn.score(knn_x, knn_y)))

        sp.save_npz(pwd + 'task_' + task + balance + '__testlabel0_knn' + str(n_neigh) + 'neighbors_edge__fold' + str(fold) + '.npz', knn_neighbors_graph)
        fold += 1
    return knn_x, knn_y, knn, knn_neighbors_graph


# # Run

# In[ ]:


for isbalance in [False, True]:
    print('************isbalance = ', isbalance)
    
    for task in ['Tp', 'Td', 'Tm']:
        print('=================task = ', task)
        
        ID, IM, dtp, mirna_ids, disease_ids, mirna_test_num, disease_test_num, knn_x, label = obtain_data('data', isbalance)

        if task == 'Tp':
            train_index_all, test_index_all, train_id_all, test_id_all = generate_task_Tp_train_test_idx(knn_x)
        elif task == 'Tm':
            item = 'miRNA'
            ids = mirna_ids
            train_index_all, test_index_all, train_id_all, test_id_all = generate_task_Tm_Td_train_test_idx(item, ids, dtp)
        elif task == 'Td':
            item = 'disease'
            ids = disease_ids
            train_index_all, test_index_all, train_id_all, test_id_all = generate_task_Tm_Td_train_test_idx(item, ids, dtp)

        if isbalance:
            balance = ''
        else:
            balance = '__nobalance'

        np.savez_compressed('/home/chujunyi/4_GNN/GraphSAINT/miRNA_disease_data/task_' + task + balance + '__testlabel0_knn_edge_train_test_index_all.npz', 
                               train_index_all = train_index_all, 
                               test_index_all = test_index_all,
                               train_id_all = train_id_all, 
                               test_id_all = test_id_all)

        pwd = '/home/chujunyi/4_GNN/GAEMDA-miRNA-disease/0_data/'
        for n_neigh in [1, 3, 5, 7, 10, 15]: 
            print('--------------------------n_neighbors = ', n_neigh)
            knn_x, knn_y, knn, knn_neighbors_graph = generate_knn_graph_save(knn_x, label, n_neigh, train_index_all, test_index_all, pwd, task, balance)


# In[ ]:


node_feature_label = pd.concat([dtp, knn_x], axis = 1)
node_feature_label


# In[ ]:


pwd = '/home/chujunyi/4_GNN/GAEMDA-miRNA-disease/0_data/'
node_feature_label.to_csv(pwd + 'node_feature_label.csv')

