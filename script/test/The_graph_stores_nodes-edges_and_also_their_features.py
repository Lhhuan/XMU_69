# coding: utf-8
#The graph stores nodes, edges and also their features
G = dgl.DGLGraph()
G.add_nodes(10)  # 10 isolated nodes are added
G.add_edge(0, 1)
G.add_edges([1, 2, 3], [3, 4, 5])  # three edges: 1->3, 2->4, 3->5
G.add_edges(4, [7, 8, 9])  # three edges: 4->7, 4->8, 4->9
G.add_edges([2, 6, 8], 5)  # three edges: 2->5, 6->5, 8->5
import torch as th
G.add_edges(th.tensor([3, 4, 5]), 1)  # three edges: 3->1, 4->1, 5->1
G = dgl.DGLGraph()
G.add_nodes(3)
G.ndata['x'] = th.zeros((3, 5))
G.ndata
G.nodes[[0, 2]].data['x'] = th.ones((2, 5))
G.ndata
G.add_edges([0, 1], 2)  # 0->2, 1->2
G.edata['y'] = th.zeros((2, 4))  # init 2 edges with zero vector(len=4)
G.edata
G.edges[1, 2].data['y'] = th.ones((1, 4))
G.edata
G.edges[0].data['y'] += 2.
G.edata
# get_ipython().run_line_magic('save', 'myhistory 65-90')
