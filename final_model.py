import torch
from torch_geometric.nn import (GCNConv, global_add_pool, global_mean_pool, Linear, GINConv, global_sort_pool, global_max_pool)
# from torch_geometric.nn import global_add_pool
from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import degree
from torch_geometric.data import Data
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from math import comb
import torch.nn as nn
from layers import *
from torch_sparse import SparseTensor
from torch_geometric.utils import to_dense_adj
import numpy as np
import os
# torch.set_printoptions(threshold=float('inf'))
device = torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")

def pagerank_pooling(data):
    graphs = Batch.to_data_list(data)
    prs = []
    for graph in graphs:
        edge_index = graph.edge_index
        num_nodes = max(edge_index[0]).item() + 1

        # 初始化PageRank值
        pr = (torch.ones(num_nodes) / num_nodes).to(device)
        damping_factor = 0.85  # 阻尼因子
        max_iter = 100  # 最大迭代次数
        tol = 1e-6  # 收敛容忍度

        # 邻接矩阵
        adj = to_dense_adj(edge_index).squeeze(0)

        # 计算出度
        out_degrees = adj.sum(dim=1)

        # 迭代计算PageRank
        for _ in range(max_iter):
            new_pr = (1 - damping_factor) / num_nodes + damping_factor * adj.T @ (pr / out_degrees)

            # 检查收敛
            if torch.norm(new_pr - pr) < tol:
                break
            pr = new_pr.t()
        prs.append(pr.unsqueeze(1))
    prs = torch.cat(prs, dim=0)

    return prs



def l_edge_count(edge_index):
    if edge_index.numel() != 0:
        num_nodes = torch.max(edge_index) + 1  # 获取顶点数量
        degrees = torch.zeros(num_nodes, dtype=torch.int32)  # 初始化每个顶点的度数
        for edge in edge_index.t():  # 遍历每一条边
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1

        degrees = torch.div(degrees, 2, rounding_mode='trunc')
        edge_count = 0
        for deg_v in degrees:
            if deg_v >= 2:
                edge_count += comb(deg_v.item(), 2)  # 计算组合数 C(deg_v, 2)
    else:
        edge_count = 0
    return edge_count

def build_T(data):
    """
    构建去除冗余边的 T 矩阵
    :param data: PyG 数据对象，包含 edge_index
    :return: 稀疏矩阵 T (num_nodes, num_edges)
    """
    edge_index = data.edge_index
    num_nodes = data.x.shape[0]
    device = edge_index.device

    # 去重处理：只保留 (u, v) 中 u < v 的边
    edge_index = edge_index[:, edge_index[0] < edge_index[1]]  # 去除重复边
    num_edges = edge_index.shape[1]  # 去重后的边数量

    # 构建 T 矩阵的行、列和值
    rows = torch.cat([edge_index[0], edge_index[1]]).to(device)  # 起点和终点都作为节点
    cols = torch.cat([torch.arange(num_edges), torch.arange(num_edges)]).to(device)  # 边索引
    values = torch.ones(cols.size(0)).to(device)  # T 矩阵的值初始化为 1

    # 构建稀疏矩阵 T
    T = SparseTensor(row=rows, col=cols, value=values, sparse_sizes=(num_nodes, num_edges))
    return T



def to_line_graph(T):
    """
    生成线图的邻接矩阵

    参数:
        data: 原图数据（预留参数，可用于后续扩展）
        num_edge: 原图的边数量（线图的顶点数量）
        T: 原图的关联矩阵（形状为 [顶点数, 边数]，T[i][j]=1 表示顶点i与边j关联）

    返回:
        adj_matrix: 线图的邻接矩阵（形状为 [num_edge, num_edge]，元素为0或1）
    """
    # 将T转换为numpy矩阵以支持矩阵运算
    T_mat = np.array(T)

    # 计算T与T的转置的乘积（结果矩阵的[i][j]表示边i和边j共享的顶点数）
    product_matrix = T_mat @ T_mat.T  # 等价于 np.dot(T_mat, T_mat.T)

    # 二值化处理：非零值（共享至少1个顶点）设为1，零值（无共享顶点）设为0
    adj_matrix = np.where(product_matrix != 0, 1, 0)

    # 线图中顶点（原边）不与自身相邻，因此将对角线元素置为0
    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix



class IMPGNN(nn.Module):
    def __init__(self, num_path, hidden_dim, task_num, k, p):
        super(WNet, self).__init__()
        self.feature_encoder = AtomEncoder(hidden_dim).to(device)
        self.metafeat_encoder = Linear(num_path, hidden_dim).to(device)
        self.bond_encoder = BondEncoder(hidden_dim).to(device).to(device)
        self.conv_org = GCN(hidden_dim, hidden_dim, p).to(device)
        self.conv_org_1 = GCN(hidden_dim, hidden_dim, p).to(device)
        self.conv_lg = GCN(hidden_dim, hidden_dim, p).to(device)
        self.conv_lg_1 = GCN(hidden_dim, hidden_dim, p).to(device)
        self.conv_meta = GCN(hidden_dim, hidden_dim, p).to(device)
        self.conv_meta_1 = GCN(hidden_dim, hidden_dim, p).to(device)
        self.pooling_add = global_add_pool
        self.pooling_mean = global_mean_pool
        self.pooling_max = global_max_pool
        self.pooling_sort = global_sort_pool
        self.k = k
        self.cat1 = Linear(hidden_dim * 2, hidden_dim).to(device)
        self.cat2 = Linear(hidden_dim * 2, hidden_dim).to(device)
        # self.cat3 = Linear(hidden_dim * 2, hidden_dim).to(device)
        self.Pred = Linear(hidden_dim, task_num).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.feature_encoder, "atom_embedding_list"):
            for emb in self.feature_encoder.atom_embedding_list:
                nn.init.xavier_uniform_(emb.weight.data)
        else:
            nn.init.xavier_uniform_(self.feature_encoder.weight.data)

        if hasattr(self, "bond_encoder"):
            if hasattr(self.bond_encoder, "bond_embedding_list"):
                for emb in self.bond_encoder.bond_embedding_list:
                    nn.init.xavier_uniform_(emb.weight.data)
            else:
                nn.init.xavier_uniform_(self.bond_encoder.weight.data)

        self.metafeat_encoder.reset_parameters()
        self.conv_org.reset_parameters()
        self.conv_org_1.reset_parameters()
        self.conv_lg.reset_parameters()
        self.conv_lg_1.reset_parameters()
        self.conv_meta.reset_parameters()
        self.conv_meta_1.reset_parameters()
        self.cat1.reset_parameters()
        self.cat2.reset_parameters()
        # self.cat3.reset_parameters()
        self.Pred.reset_parameters()

    def forward(self, data):

        # 构建转移矩阵
        T = build_T(data)
        node_feat = self.feature_encoder(data.x.long())
        meta_feat = self.metafeat_encoder(data.metafeat.float())

        # 构建线图
        line_graphs = []
        data_lg = data.clone()
        graphs = Batch.to_data_list(data_lg)
        for graph in graphs:
            num_edge = l_edge_count(graph.edge_index)
            line_graph = to_line_graph(T)
            # line_graph = to_line_graph(graph)
            line_graphs.append(line_graph)
        data_lg = Batch.from_data_list(line_graphs)
        edge_feat = self.bond_encoder(data_lg.x)


        # 第一层卷积
        node_feat_1 = node_feat.clone()
        meta_feat_1 = meta_feat.clone()
        node_feat = node_feat_1 * (T @ edge_feat)
        meta_feat = meta_feat_1 * (T @ edge_feat)
        edge_feat = edge_feat * (T.t() @ node_feat_1) * (T.t() @ meta_feat_1)
        node_feat = F.elu(node_feat)
        meta_feat = F.elu(meta_feat)
        edge_feat = F.elu(edge_feat)
        h_org = self.conv_org(node_feat, data.edge_index)
        h_meta = self.conv_meta(meta_feat, data.edge_index)
        h_lg = self.conv_lg(edge_feat, data_lg.edge_index)


        # 第二层卷积
        h_org_hat = h_org * (T @ h_lg)
        h_meta_hat = h_meta * (T @ h_lg)
        h_lg_hat = h_lg * (T.t() @ h_org) * (T.t() @ h_meta)
        h_org_hat = F.elu(h_org_hat)
        h_meta_hat = F.elu(h_meta_hat)
        h_lg_hat = F.elu(h_lg_hat)
        h_org_hat = self.conv_org_1(h_org_hat, data.edge_index)
        h_meta_hat = self.conv_meta_1(h_meta_hat, data.edge_index)
        h_lg_hat = self.conv_lg_1(h_lg_hat, data_lg.edge_index)

        # pagerank根据结构获得每个节点的权重
        data_c = data.clone()
        data_c.x = h_org_hat
        prs = pagerank_pooling(data_c)
        h_org_hat = h_org_hat * prs
        h_meta_hat = h_meta_hat * prs


        z_org = self.pooling_add(h_org_hat, data.batch)
        z_meta = self.pooling_add(h_meta_hat, data.batch)

        Z = self.cat2(torch.cat((z_meta, z_org), dim=-1))
        pred = self.Pred(Z)

        return pred, h_meta_hat, h_org_hat

