import os
import dgl
import torch as th
import torch
import numpy as np
from dgl.sampling import random_walk
import pandas as pd
from collections import defaultdict
from dgl.data.utils import load_graphs, save_graphs
from tqdm import tqdm
from multiprocessing import Pool
import sys


def get_corpus(Graph, length):
    np.random.seed(1)
    # start_node = Graph.edges()[0][np.random.randint(0,Graph.edges()[0].shape,doc_nums)]
    start_node = Graph.edges()[0]
    corpus, node_type = random_walk(Graph, start_node, length=length) #node type是什么? #If omitted, DGL assumes that g only has one node & edge type
    return corpus


def get_frequency(Graph, corpus, compress_rate=0.75):
    nodes_nums = Graph.nodes().shape[0]
    frequency = defaultdict(float)
    for i in range(nodes_nums):
        frequency[i] = 0
    for i in corpus:
        for j in i:
            frequency[int(j)] += 1
    for key in frequency:
        frequency[key] = frequency[key] ** compress_rate
    # frequency = sorted(frequency.items(), key=lambda i: i[0]) #按照键排序
    return frequency


def get_sample(corpus, frequency, half_windowsize, n_negword):
    assert half_windowsize >= 1
    frequency = np.array(sorted(frequency.items(), key=lambda i: i[0]))[:, 1] #按照键排序
    # frequency = frequency/frequency.sum()
    document_length = corpus[0].shape[0] #-------------------------------------------------------------------
    center_words, pos_words, neg_words = [], [], []
    for doc in tqdm(corpus): #tqdm进度条库
        for c in range(half_windowsize, document_length - half_windowsize):
            center_word = [doc[c]]
            pos_word = doc[c - half_windowsize:c].tolist() + doc[c + 1:c + half_windowsize + 1].tolist()
            tmp_frequency = frequency.copy()
            for i in center_word + pos_word:
                tmp_frequency[i] = 0
            neg_word = np.random.choice(np.arange(frequency.shape[0]), size=(n_negword * 2 * half_windowsize),
                                        p=tmp_frequency / tmp_frequency.sum()).tolist()  #------#除去center_word + pos_word，p为a中每个采样点的概率分布
            center_words.append(center_word)
            pos_words.append(pos_word)
            neg_words.append(neg_word)
    return np.array(center_words).astype(np.uint16), np.array(pos_words).astype(np.uint16), np.array(neg_words).astype(
        np.uint16)


if __name__ == '__main__':
    graph_file = sys.argv[1]
    threads = sys.argv[2]
    root_path,t = os.path.split(graph_file)
    t = t.split('graph')[0]
    g = load_graphs(graph_file)[0][0]
    try:
        corpus = np.load(os.path.join(root_path,t+'corpus.npy'))
    except:
        corpus = get_corpus(g,15)
        corpus = np.array(corpus)
        np.save(os.path.join(root_path,t+'corpus.npy'),corpus)
    frequency = get_frequency(g, corpus)
    threads = 36
    half_windowsize = 4
    n_negword = 32 #_------------------------------------------------?
    output = []
    document_nums = corpus.shape[0]
    step = int(np.ceil(document_nums / threads))
    pool = Pool(threads)
    for i in range(threads):
        print(i, ' start')
        splitted_corpus = corpus[i * step:(i + 1) * step].copy()
        output.append(pool.apply_async(get_sample, (splitted_corpus, frequency, half_windowsize, n_negword,)))
        print(i, ' done')
    pool.close()
    pool.join()
    center_words = np.concatenate([i.get()[0] for i in output], axis=0)
    pos_words = np.concatenate([i.get()[1] for i in output], axis=0)
    neg_words = np.concatenate([i.get()[2] for i in output], axis=0)
    c=th.tensor(center_words.astype(np.int32))
    p=th.tensor(pos_words.astype(np.int32))
    n=th.tensor(neg_words.astype(np.int32))
    th.save(d,os.path.join(root_path,t+r'sample.pth'))
    #-------------------------------------------------------------------------
    # np.savez_compressed(os.path.join(root_path,t+r'sample.npz'), c=center_words, p=pos_words, n=neg_words)
    # #------------------

    # #------------------------------------------ huan add
    # a = np.load(os.path.join(root_path,t+r'sample.npz'))
    # # print(a['c'][k] for k in a['c'])
    # c=th.tensor(a['c'].astype(np.int32))
    # p=th.tensor(a['p'].astype(np.int32))
    # n=th.tensor(a['n'].astype(np.int32))
  
    # d=c,p,n
    # # d2 = th.tensor(d)
    # th.save(d,os.path.join(root_path,t+r'sample2.pth'))
#--------------------------------------------------------------------
    # th.save(d,os.path.join(root_path,t+r'sample3.pth'))

    # d = dict(zip(("data1{}".format(k) for k in a), (a[k] for k in a)))
    # th.save(d,os.path.join(root_path,t+r'sample.pth'))

    # np.save(r'/public/home/zhiyu/NLP/skipgram/data/center_words', center_words)
    # np.save(r'/public/home/zhiyu/NLP/skipgram/data/pos_words', pos_words)
    # np.save(r'/public/home/zhiyu/NLP/skipgram/data/neg_words', neg_words)
    # th.save("aaa.npz","aaa.pth")


#pool.apply_async: 进程池 apply_async(func[, args=()[, kwds={}[, callback=None]]]) ,Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果池还没有满，就会创建一个新的进程来执行请求。如果池满，请求就会告知先等待，直到池中有进程结束，才会创建新的进程来执行这些请求














