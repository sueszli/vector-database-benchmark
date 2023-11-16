"""
- Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
- gene: discrete ops information (w/o output connection)
- dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import ops
from ops import PRIMITIVES
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

def to_dag(C_in, gene, reduction, bn_affine=True):
    if False:
        while True:
            i = 10
    ' generate discrete ops from gene '
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for (op_name, s_idx) in edges:
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](C_in, stride, bn_affine)
            if not isinstance(op, ops.Identity):
                op = nn.Sequential(op, ops.DropPath_())
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)
    return dag

def from_str(s):
    if False:
        print('Hello World!')
    ' generate genotype from string\n    e.g. "Genotype(\n            normal=[[(\'sep_conv_3x3\', 0), (\'sep_conv_3x3\', 1)],\n                    [(\'sep_conv_3x3\', 1), (\'dil_conv_3x3\', 2)],\n                    [(\'sep_conv_3x3\', 1), (\'sep_conv_3x3\', 2)],\n                    [(\'sep_conv_3x3\', 1), (\'dil_conv_3x3\', 4)]],\n            normal_concat=range(2, 6),\n            reduce=[[(\'max_pool_3x3\', 0), (\'max_pool_3x3\', 1)],\n                    [(\'max_pool_3x3\', 0), (\'skip_connect\', 2)],\n                    [(\'max_pool_3x3\', 0), (\'skip_connect\', 2)],\n                    [(\'max_pool_3x3\', 0), (\'skip_connect\', 2)]],\n            reduce_concat=range(2, 6))"\n    '
    genotype = eval(s)
    return genotype

def parse(alpha, beta, k):
    if False:
        return 10
    "\n    parse continuous alpha to discrete gene.\n    alpha is ParameterList:\n    ParameterList [\n        Parameter(n_edges1, n_ops),\n        Parameter(n_edges2, n_ops),\n        ...\n    ]\n\n    beta is ParameterList:\n    ParameterList [\n        Parameter(n_edges1),\n        Parameter(n_edges2),\n        ...\n    ]\n\n    gene is list:\n    [\n        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],\n        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],\n        ...\n    ]\n    each node has two edges (k=2) in CNN.\n    "
    gene = []
    assert PRIMITIVES[-1] == 'none'
    connect_idx = []
    for (edges, w) in zip(alpha, beta):
        (edge_max, primitive_indices) = torch.topk((w.view(-1, 1) * edges)[:, :-1], 1)
        (topk_edge_values, topk_edge_indices) = torch.topk(edge_max.view(-1), k)
        node_gene = []
        node_idx = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))
            node_idx.append((edge_idx.item(), prim_idx.item()))
        gene.append(node_gene)
        connect_idx.append(node_idx)
    return (gene, connect_idx)

def parse_gumbel(alpha, beta, k):
    if False:
        return 10
    "\n    parse continuous alpha to discrete gene.\n    alpha is ParameterList:\n    ParameterList [\n        Parameter(n_edges1, n_ops),\n        Parameter(n_edges2, n_ops),\n        ...\n    ]\n\n    beta is ParameterList:\n    ParameterList [\n        Parameter(n_edges1),\n        Parameter(n_edges2),\n        ...\n    ]\n\n    gene is list:\n    [\n        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],\n        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],\n        ...\n    ]\n    each node has two edges (k=2) in CNN.\n    "
    gene = []
    assert PRIMITIVES[-1] == 'none'
    connect_idx = []
    for (edges, w) in zip(alpha, beta):
        discrete_a = F.gumbel_softmax(edges[:, :-1].reshape(-1), tau=1, hard=True)
        for i in range(k - 1):
            discrete_a = discrete_a + F.gumbel_softmax(edges[:, :-1].reshape(-1), tau=1, hard=True)
        discrete_a = discrete_a.reshape(-1, len(PRIMITIVES) - 1)
        reserved_edge = (discrete_a > 0).nonzero()
        node_gene = []
        node_idx = []
        for i in range(reserved_edge.shape[0]):
            edge_idx = reserved_edge[i][0].item()
            prim_idx = reserved_edge[i][1].item()
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx))
            node_idx.append((edge_idx, prim_idx))
        gene.append(node_gene)
        connect_idx.append(node_idx)
    return (gene, connect_idx)