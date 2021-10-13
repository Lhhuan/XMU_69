import dgl
import torch as th
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from dgl.data.utils import load_graphs, save_graphs
import networkx as nx
from tqdm import tqdm

def sourcefile_preparing(file_path): #add diseaseidx and geneidx
    source_file = pd.read_csv(file_path,sep='\t')

    diseases = np.unique(source_file['diseaseId'])
    dic1 = defaultdict(int)
    for idx,it in enumerate(diseases):
        print(it,idx)
        dic1[it] = idx
    diseases_idx = [dic1[i] for i in source_file['diseaseId'].values]

    gene = np.unique(source_file['geneId'])
    dic2 = defaultdict(int)
    for idx,it in enumerate(gene):
        dic2[it] = idx
    gene_idx = [dic2[i] for i in source_file['geneId'].values]

    source_file['diseaseidx'] = diseases_idx
    source_file['geneidx'] = gene_idx
    source_file.to_csv(file_path,index=None,sep='\t')
    return source_file

# source_file = pd.read_csv(r'E:\deeplearning\NLP\data\all_variant_disease_associations.csv',sep='\t')
# source_file = pd.read_csv(r'E:\deeplearning\NLP\fdata\filtered_variant_disease_associations.csv',sep='\t')
source_file = pd.read_csv(r"/public/home/huanhuan/test/test_disGenet/all_gene_disease_associations.tsv",sep='\t')

list(source_file.columns.values)

def homograph_gen(source_file):
    # 生成有向图
    source_file = source_file.copy()
    nodes_nums = source_file['geneidx'].max() +1 #diseaseidx is ahead of geneidx
    Graph = dgl.DGLGraph()
    Graph.add_nodes(nodes_nums)
    Graph.add_edges(source_file['diseaseidx'].values, source_file['geneidx'].values)
    return Graph.to_simple() #--------------------------------------------------------------------------------------------------?
    # Graph.add_edges(source_file['geneidx'].values, source_file['diseaseidx'].values)

    # gcmc = source_file['geneidx'].values.tolist() + source_file['diseaseidx'].values.tolist()
    # dst = source_file['diseaseidx'].values.tolist() + source_file['geneidx'].values.tolist()
    # Graph = dgl.graph((gcmc, dst))
    # Graph.ndata['oh_idx'] = Graph.nodes().unsqueeze(-1)
    

Graph = homograph_gen(source_file)




disease_nums =source_file['diseaseidx'].max()+1

def break_edges(directedgraph):
    edges_num = directedgraph.edges()[0].shape[0]
    np.random.seed(edges_num)
    disrupted_egdes = np.random.permutation(edges_num)
    partition_nums = int(edges_num//10)
    train_edges = disrupted_egdes[:8*partition_nums]
    val_edges = disrupted_egdes[8 * partition_nums:9 * partition_nums]
    test_edges = disrupted_egdes[9 * partition_nums:]
    train_graph = dgl.to_bidirected(dgl.edge_subgraph(directedgraph,train_edges,preserve_nodes=True,store_ids=False))
    val_graph = dgl.to_bidirected(dgl.edge_subgraph(directedgraph,val_edges,preserve_nodes=True,store_ids=False))
    test_graph = dgl.to_bidirected(dgl.edge_subgraph(directedgraph,test_edges,preserve_nodes=True,store_ids=False))
    return train_graph,val_graph,test_graph

train_graph,val_graph,test_graph = break_edges(Graph)

save_graphs(r'train_graph.dgl',train_graph)
save_graphs(r'val_graph.dgl',val_graph)
save_graphs(r'test_graph.dgl',test_graph)





#---------------------------------- bg = dgl.to_bidirected(g) 无向图

def bipartile_graph():
    pass


def sourcefile_filter(directedgraph,source_file):
    #去掉度<=3的节点
    degree = directedgraph.in_degrees() + directedgraph.out_degrees()
    keep_idx = np.array(degree)>3
    drop_nodes = np.argwhere(keep_idx==False)
    l = []
    for disease,gene in zip(source_file['diseaseidx'].values,source_file['geneidx'].values):
        pass

save_graphs(r'homograph.dgl',Graph)
#-------------huan add
Graph = dgl.to_bidirected(Graph)
save_graphs(r'huan_undirectied_Graph.dgl',Graph)


















