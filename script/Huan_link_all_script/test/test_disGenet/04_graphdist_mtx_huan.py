import numpy as np
import dgl
import os
import pickle
import queue
from tqdm import tqdms

def save_dict(file, name,root_path ):
    file_path = os.path.join(root_path,name+ '.pkl') if not name.endswith('.pkl') else os.path.join(root_path,name)
    with open(file_path, 'wb') as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name,root_path):
    file_path = os.path.join(root_path,name+ '.pkl') if not name.endswith('.pkl') else os.path.join(root_path,name)
    with open(file_path , 'rb') as f:
        return pickle.load(f)


def load_tree(root_path):
    tree_path = os.path.join(root_path,'train_tree.pkl')
    checkfile = os.path.exists(tree_path)
    if checkfile:
        print('loading tree')
        with open(tree_path, 'rb') as f:
            tree = pickle.load(f)
        return tree


def get_graphdist_mtx():
    nodes_nums = 51836
    mtx = np.zeros([nodes_nums,nodes_nums])
    for i in tqdm(range(nodes_nums)):
        used_nodes = []
        cur_nodes = queue.deque([i])
        next_nodes = []
        layer = 0
        while True:
            cur_node = cur_nodes.popleft()
            used_nodes.append(cur_node)
            mtx[i, cur_node] = layer
            next_nodes.extend(tree[cur_node])
            if len(cur_nodes) == 0:
                layer += 1
                next_nodes = list(set(next_nodes) - set(used_nodes))
                if len(next_nodes) == 0:
                    break
                cur_nodes = queue.deque(next_nodes)
                next_nodes = []
    l, r = np.where(mtx == 0)
    l, r = l[l != r], r[l != r]
    mtx[l, r] = mtx.max() + 1
    return mtx



if __name__ == '__main__':
    root_path = r'/public/home/huanhuan/test_disGenet/'
    tree = load_tree(root_path)
    mtx = get_graphdist_mtx()
# tree = {0:[1,2],1:[0,3],2:[0,3],3:[1,2,4],4:[3]}
# mtx = np.zeros([5,5])
# for i in range(5):
#     used_nodes = []
#     cur_nodes = queue.deque([i])
#     next_nodes = []
#     layer = 0
#     while True:
#         cur_node = cur_nodes.popleft()
#         used_nodes.append(cur_node)
#         mtx[i,cur_node] =layer
#         next_nodes.extend(tree[cur_node])
#         if len(cur_nodes) == 0:
#             layer += 1
#             next_nodes = list(set(next_nodes)-set(used_nodes))
#             if len(next_nodes) == 0:
#                 break
#             cur_nodes = queue.deque(next_nodes)
#             next_nodes = []