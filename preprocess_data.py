import torch_geometric.utils

from utils import *

import os
import torch,random
from tqdm import tqdm
import argparse
import time
from typing import Callable, List, Optional
### importing OGB
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, download_url, extract_zip
import os.path as osp
import pandas as pd
import sys
from mol import smiles2graph
# from dig.threedgraph.dataset import QM93D
from sklearn.utils import shuffle
from torch_geometric.datasets import TUDataset
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict)


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

meta_paths_BBBP = [
    ('C', 'C', 'C'),
    ('I', 'C', 'C'),
    ('S', 'C', 'C'),
    ('F', 'C', 'C'),
    ('Ca', 'C', 'C'),
    ('N', 'C', 'C'),
    ('C', 'N', 'C'),
    ('Cl', 'C', 'C'),
    ('B', 'C', 'C'),
    ('Na', 'C', 'C'),
    ('O', 'C', 'C'),
    ('C', 'O', 'C'),
    ('Br', 'C', 'C'),
    ('P', 'C', 'C'),
    ('C', 'S', 'C'),
    ('S', 'C', 'N'),
] # 16

meta_paths_Bace = [
    ('C', 'C', 'C'),
    ('I', 'C', 'C'),
    ('Cl', 'C', 'C'),
    ('O', 'C', 'C'),
    ('C', 'O', 'C'),
    ('N', 'C', 'C'),
    ('C', 'N', 'C'),
    ('Br', 'C', 'C'),
    ('S', 'C', 'C'),
    ('C', 'S', 'C'),
    ('F', 'C', 'C'),
    ('S', 'C', 'N'),
] # 12

meta_paths_Esol = [
    ('C', 'C', 'C'),
    ('O', 'C', 'C'),
    ('C', 'O', 'C'),
    ('Cl', 'C', 'C'),
    ('S', 'C', 'C'),
    ('S', 'C', 'N'),
    ('Br', 'C', 'C'),
    ('I', 'C', 'C'),
    ('P', 'C', 'C'),
    ('N', 'C', 'C'),
    ('C', 'N', 'C'),
    ('F', 'C', 'C'),
] # 12 Freesolv

meta_paths_Freesolv = [
    ('C', 'C', 'C'),
    ('O', 'C', 'C'),
    ('C', 'O', 'C'),
    ('Cl', 'C', 'C'),
    ('S', 'C', 'C'),
    ('S', 'C', 'N'),
    ('Br', 'C', 'C'),
    ('I', 'C', 'C'),
    ('P', 'C', 'C'),
    ('N', 'C', 'C'),
    ('C', 'N', 'C'),
    ('F', 'C', 'C'),
] # 12

meta_paths_Lipop = [
    ('C', 'C', 'C'),
    ('O', 'C', 'C'),
    ('C', 'O', 'C'),
    ('Cl', 'C', 'C'),
    ('S', 'C', 'C'),
    ('S', 'C', 'N'),
    ('Br', 'C', 'C'),
    ('I', 'C', 'C'),
    ('P', 'C', 'C'),
    ('N', 'C', 'C'),
    ('C', 'N', 'C'),
    ('F', 'C', 'C'),
    ('C', 'Si', 'C'),
    ('Si', 'C', 'C'),
    ('Se', 'C', 'C'),
    ('C', 'Se', 'C'),
    ('B', 'C', 'C'),
    ('C', 'B', 'C'),
] # 18

meta_paths_ClinTox = [
    ('C', 'C', 'C'),
    ('O', 'C', 'C'),
    ('C', 'O', 'C'),
    ('Cl', 'C', 'C'),
    ('S', 'C', 'C'),
    ('S', 'C', 'N'),
    ('Br', 'C', 'C'),
    ('I', 'C', 'C'),
    ('P', 'C', 'C'),
    ('N', 'C', 'C'),
    ('C', 'N', 'C'),
    ('F', 'C', 'C'),
    ('B', 'C', 'C'),
    ('C', 'B', 'C'),
    ('C', 'C', 'Hg'),
    ('O', 'Hg', 'C'),
    ('Fe', 'N', 'O'),
    ('O', 'Pt', 'O'),
    ('N', 'Pt', 'N'),
    ('S', 'Au', 'P'),
    ('C', 'S', 'Au'),
    ('C', 'P', 'Au'),
    ('Cl', 'Mn', 'Cl'),
    ('C', 'Hg', 'Cl'),
    ('O', 'Si', 'O'),
    ('C', 'Si', 'O'),
    ('Si', 'O', 'Si'),
    ('Al', 'O', 'Al'),
    ('S', 'O', 'Al'),
    ('O', 'Al', 'O'),
    ('O', 'B', 'C'),
    ('N', 'C', 'B'),
    ('S', 'Se', 'S'),
    ('O', 'Bi', 'O'),
    ('C', 'O', 'Bi'),
    ('O', 'Tc', 'O'),
    ('O', 'Ti', 'O'),
    ('As', 'O', 'As'),
    ('C', 'O', 'Ca'),
    ('Cl', 'C', 'C'),
    ('Cl', 'Cr', 'Cl'),
    ('Cl', 'Zn', 'Cl'),
    ('N', 'Fe', 'N'),
    ('Fe', 'N', 'O'),
    ('Cl', 'Cu', 'Cl'),
] # 45

meta_paths_Tox21 = [
    ('Cl', 'Ba', 'Cl'),
    ('Na', 'O', 'C'),
    ('F', 'C', 'C'),
    ('Cl', 'Pt', 'Cl'),
    ('N', 'Pt', 'N'),
    ('N', 'Pt', 'O'),
    ('O', 'Pt', 'O'),
    ('C', 'O', 'Pt'),
    ('C', 'Ag', 'O'),
    ('O', 'B', 'C'),
    ('N', 'C', 'B'),
    ('B', 'C', 'C'),
    ('C', 'B', 'C'),
    ('C', 'Ge', 'C'),
    ('C', 'Ge', 'Cl'),
    ('C', 'C', 'Ge'),
    ('O', 'Se', 'O'),
    ('C', 'Se', 'C'),
    ('C', 'C', 'Se'),
    ('N', 'C', 'Se'),
    ('N', 'Se', 'C'),
    ('C', 'S', 'Au'),
    ('Cl', 'Au', 'Cl'),
    ('P', 'Au', 'S'),
    ('C', 'P', 'Au'),
    ('N', 'Au', 'N'),
    ('O', 'Al', 'O'),
    ('C', 'O', 'Al'),
    ('C', 'Hg', 'Cl'),
    ('O', 'Hg', 'C'),
    ('C', 'C', 'Hg'),
    ('C', 'O', 'Hg'),
    ('C', 'O', 'Sr'),
    ('O', 'Cd', 'O'),
    ('N', 'O', 'Cd'),
    ('C', 'Ti', 'C'),
    ('C', 'C', 'Ti'),
    ('O', 'Ti', 'O'),
    ('C', 'O', 'Ti'),
    ('C', 'Bi', 'C'),
    ('C', 'C', 'Bi'),
    ('O', 'Bi', 'Cl'),
    ('O', 'As', 'O'),
    ('O', 'C', 'As'),
    ('As', 'O', 'As'),
    ('Cl', 'Yb', 'Cl'),
    ('O', 'Si', 'O'),
    ('C', 'Si', 'O'),
    ('C', 'C', 'Si'),
    ('C', 'Si', 'Cl'),
    ('C', 'Si', 'C'),
    ('Br', 'C', 'C'),
    ('Cl', 'Ni', 'Cl'),
    ('S', 'Ni', 'S'),
    ('C', 'S', 'Ni'),
    ('Cl', 'In', 'Cl'),
    ('Fe', 'N', 'O'),
    ('N', 'Fe', 'N'),
    ('S', 'Zn', 'S'),
    ('C', 'S', 'Zn'),
    ('P', 'C', 'C'),
    ('O', 'Cr', 'O'),
    ('Cr', 'O', 'Cr'),
    ('C', 'O', 'Cr'),
    ('O', 'Mo', 'O'),
    ('Cl', 'Dy', 'Cl'),
    ('O', 'Sn', 'O'),
    ('C', 'Sn', 'C'),
    ('C', 'Sn', 'O'),
    ('C', 'O', 'Sn'),
    ('C', 'C', 'Sn'),
    ('Cl', 'Sn', 'C'),
    ('F', 'Sn', 'C'),
    ('Cl', 'Nd', 'Cl'),
    ('O', 'Sb', 'O'),
    ('C', 'O', 'Sb'),
    ('Cl', 'Sb', 'Cl'),
    ('Br', 'Ca', 'Br'),
    ('Cl', 'V', 'Cl'),
    ('C', 'C', 'V'),
    ('C', 'V', 'C'),
    ('C', 'V', 'Cl'),
] # 82

meta_paths_Sider = [
    ('C', 'C', 'C'),
    ('O', 'C', 'C'),
    ('C', 'O', 'C'),
    ('Cl', 'C', 'C'),
    ('S', 'C', 'C'),
    ('S', 'C', 'N'),
    ('Br', 'C', 'C'),
    ('P', 'C', 'C'),
    ('N', 'C', 'C'),
    ('C', 'N', 'C'),
    ('F', 'C', 'C'),
    ('S', 'Se', 'S'),
    ('B', 'C', 'C'),
    ('C', 'B', 'C'),
    ('Cl', 'Zn', 'Cl'),
    ('Cl', 'Pt', 'Cl'),
    ('Li', 'O', 'C'),
] # 17

meta_paths_qm9 = [
    ('C', 'C', 'C'),
    ('C', 'C', 'F'),
    ('C', 'C', 'H'),
    ('C', 'C', 'N'),
    ('C', 'C', 'O'),
    ('C', 'N', 'C'),
    ('C', 'N', 'H'),
    ('C', 'N', 'N'),
    ('C', 'N', 'O'),
    ('C', 'O', 'C'),
    ('C', 'O', 'H'),
    ('C', 'O', 'N'),
    ('F', 'C', 'F'),
    ('F', 'C', 'N'),
    ('F', 'C', 'O'),
    ('H', 'C', 'H'),
    ('H', 'C', 'N'),
    ('H', 'C', 'O'),
    ('H', 'N', 'H'),
    ('H', 'N', 'N'),
    ('H', 'N', 'O'),
    ('H', 'O', 'H'),
    ('H', 'O', 'N'),
    ('N', 'C', 'N'),
    ('N', 'C', 'O'),
    ('N', 'N', 'N'),
    ('N', 'N', 'O'),
    ('N', 'O', 'N'),
    ('O', 'C', 'O'),
    ('O', 'N', 'O')
] #30


def murcko_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

def scaffold_split_indices(smiles_list, frac_train=0.8, frac_valid=0.1, seed=42, balanced=False):
    """
    根据scaffold划分，返回train/valid/test索引
    """
    np.random.seed(seed)
    scaffolds_map = defaultdict(list)
    for i, s in enumerate(smiles_list):
        scaf = murcko_scaffold(s)
        scaffolds_map[scaf].append(i)

    scaffold_sets = list(scaffolds_map.values())
    if balanced:
        np.random.shuffle(scaffold_sets)
    else:
        scaffold_sets.sort(key=lambda x: len(x), reverse=True)

    n_total = len(smiles_list)
    n_train = int(frac_train * n_total)
    n_valid = int(frac_valid * n_total)

    train_idx, valid_idx, test_idx = [], [], []
    for group in scaffold_sets:
        if len(train_idx) + len(group) <= n_train:
            train_idx.extend(group)
        elif len(valid_idx) + len(group) <= n_valid:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)
    return torch.tensor(train_idx), torch.tensor(valid_idx), torch.tensor(test_idx)

class Meta_pathTransform(object):
    def __init__(self, metapath):
        self.mp = metapath

    def __call__(self, data):
        graph_nx= data
        G = torch_geometric.utils.to_networkx(graph_nx)
        one_hot = pd.DataFrame(0, index=G.nodes(), columns=[f"{t1}_{t2}_{t3}" for t1, t2, t3 in globals()[self.mp]]) #这里需要改
        symbol = data.symbol
        symbol = {i: symbol[i] for i in range(len(symbol))}
        for node in G.nodes():
            node_type = symbol[node]
            for t1, t2, t3 in globals()[self.mp]:
                if node_type == t1:
                    # 找到与当前节点相连的邻居
                    paths = set()
                    for neighbor1 in G.neighbors(node):
                        if symbol[neighbor1] == t2:
                            # 在邻居的邻居中查找符合条件的节点
                            for neighbor2 in G.neighbors(neighbor1):
                                if neighbor2 != node and symbol[neighbor2] == t3:  # 确保节点唯一
                                    path = (node, neighbor1, neighbor2)
                                    paths.add(path)
                                    if paths:
                                        one_hot.loc[node, f"{t1}_{t2}_{t3}"] = 1
                                        one_hot.loc[neighbor1, f"{t1}_{t2}_{t3}"] = 1
                                        one_hot.loc[neighbor2, f"{t1}_{t2}_{t3}"] = 1

        node_embeddings = torch.tensor(one_hot.values)
        # x = torch.cat((data.x, node_embeddings), dim=1)
        data.metafeat = node_embeddings
        return data

class BBBPDataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset",
            smiles2graph=smiles2graph,
            transform=None,
            pre_transform=None
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "BBBP")
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "BBBP.csv"

    @property
    def processed_file_names(self):
        return "BBBP_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "BBBP.csv")
        )
        smiles_list = data_df["smiles"]
        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)

            if graph != None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max()+1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([(data_df["p_np"].iloc[i])])
                    data.cliques = cliques
                    data.symbol = (graph['atom_symbol'])

                    data_list.append(data)

                except AssertionError:
                    continue
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time()-t0

        data, slices = self.collate(data_list)
        data.y = data.y.view(len(data.y), 1)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class BBBPDataset_scaffold(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=None,
                 transform=None,
                 pre_transform=None):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "BBBP")
        self.task_type = "classification"
        self.num_tasks = 1   # BBBP 只有一个二分类任务
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "BBBP.csv"

    @property
    def processed_file_names(self):
        return "bbbp_processed.pt"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "BBBP.csv"))
        smiles_list = data_df["smiles"]
        labels = ["p_np"]  # BBBP 标签列名是 p_np

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()
            smiles = smiles_list[i]
            y = data_df.iloc[i][labels]

            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)

            if graph is not None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.cliques = cliques
                    data.symbol = graph['atom_symbol']
                    data.smiles = smiles  # 保存 SMILES
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def get_idx_split(self, data_size, train_size, valid_size, seed):
    
        random.seed(seed)

        smiles_all = [d.smiles for d in self]
        labels = torch.cat([d.y for d in self], dim=0)  # 假设 d.y 是 shape [num_tasks]

        def scaffold(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

        # 1. 先按 scaffold 分组
        scaffolds = {}
        for i, s in enumerate(smiles_all):
            scaf = scaffold(s)
            scaffolds.setdefault(scaf, []).append(i)

        scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)

        train_idx, val_idx, test_idx = [], [], []
        num_train, num_val = train_size, valid_size
        for scaf_set in scaffold_sets:
            if len(train_idx) + len(scaf_set) <= num_train:
                train_idx += scaf_set
            elif len(val_idx) + len(scaf_set) <= num_val:
                val_idx += scaf_set
            else:
                test_idx += scaf_set

        train_idx, val_idx, test_idx = map(lambda x: torch.tensor(x, dtype=torch.long),
                                        [train_idx, val_idx, test_idx])

        # 2. 检查正负样本覆盖情况，如果没有正样本，随机从其他 split 拿一些
        def ensure_pos_neg(idx):
            y_split = labels[idx]
            # 对多任务：每个任务列都要检查
            need_fix = []
            for task in range(y_split.shape[1] if y_split.dim() > 1 else 1):
                y_t = y_split[:, task] if y_split.dim() > 1 else y_split
                if (y_t == 1).sum() == 0 or (y_t == 0).sum() == 0:
                    need_fix.append(task)
            return need_fix

        # 对 val/test 分别检查并补充正样本
        for split_name, idx_var in [('valid', val_idx), ('test', test_idx)]:
            need_fix = ensure_pos_neg(idx_var)
            if need_fix:
                # 从 train 找出缺少的类别样本，随机搬一部分过来
                for task in need_fix:
                    y_train = labels[train_idx][:, task]
                    y_split = labels[idx_var][:, task]
                    if (y_split == 1).sum() == 0:  # 没有正样本
                        pos_candidates = (y_train == 1).nonzero(as_tuple=True)[0]
                        if len(pos_candidates) > 0:
                            move_idx = pos_candidates[:max(1, len(pos_candidates)//10)]  # 随机搬一部分
                            idx_var = torch.cat([idx_var, train_idx[move_idx]])
                            mask = torch.ones(len(train_idx), dtype=torch.bool)
                            mask[move_idx] = False
                            train_idx = train_idx[mask]
                    if (y_split == 0).sum() == 0:  # 没有负样本类似处理
                        neg_candidates = (y_train == 0).nonzero(as_tuple=True)[0]
                        if len(neg_candidates) > 0:
                            move_idx = neg_candidates[:max(1, len(neg_candidates)//10)]
                            idx_var = torch.cat([idx_var, train_idx[move_idx]])
                            mask = torch.ones(len(train_idx), dtype=torch.bool)
                            mask[move_idx] = False
                            train_idx = train_idx[mask]
                if split_name == 'valid':
                    val_idx = idx_var
                else:
                    test_idx = idx_var

        return {'train': train_idx, 'valid': val_idx, 'test': test_idx}


    # def get_idx_split(self, data_size, train_size, valid_size, seed):
    #     # Scaffold 划分
    #     smiles_all = [d.smiles for d in self]

    #     def scaffold(smiles):
    #         mol = Chem.MolFromSmiles(smiles)
    #         return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

    #     scaffolds = {}
    #     for i, s in enumerate(smiles_all):
    #         scaf = scaffold(s)
    #         if scaf not in scaffolds:
    #             scaffolds[scaf] = [i]
    #         else:
    #             scaffolds[scaf].append(i)

    #     scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)

    #     train_idx, val_idx, test_idx = [], [], []
    #     num_train, num_val = train_size, valid_size
    #     for scaf_set in scaffold_sets:
    #         if len(train_idx) + len(scaf_set) <= num_train:
    #             train_idx += scaf_set
    #         elif len(val_idx) + len(scaf_set) <= num_val:
    #             val_idx += scaf_set
    #         else:
    #             test_idx += scaf_set

    #     return {
    #         'train': torch.tensor(train_idx, dtype=torch.long),
    #         'valid': torch.tensor(val_idx, dtype=torch.long),
    #         'test': torch.tensor(test_idx, dtype=torch.long)
    #     }

class BBBPD1ataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset",
            smiles2graph=smiles2graph,
            transform=None,
            pre_transform=None
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "BBBP")
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "BBBP.csv"

    @property
    def processed_file_names(self):
        return "BBBP_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "BBBP.csv")
        )
        smiles_list = data_df["smiles"]
        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]

            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            if graph != None and cliques:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([(data_df["p_np"].iloc[i])])
                data.cliques = cliques
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time()-t0

        data, slices = self.collate(data_list)
        data.y = data.y.view(len(data.y), 1)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class ClinToxDataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "ClinTox")
        self.task_type = "classification"
        self.num_tasks = 2
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "clintox.csv.gz"

    @property
    def processed_file_names(self):
        return "clintox_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "clintox.csv.gz")
        )
        smiles_list = data_df["smiles"]
        lable = ["FDA_APPROVED",
                 "CT_TOX"]


        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            # print(graph['edge_index'])
            if graph != None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.cliques = cliques
                    data.symbol = (graph['atom_symbol'])
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time()-t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class ClinToxDataset_scaffold(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "ClinTox")
        self.task_type = "classification"
        self.num_tasks = 2
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "clintox.csv.gz"

    @property
    def processed_file_names(self):
        return "clintox_processed.pt"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "clintox.csv.gz"))
        smiles_list = data_df["smiles"]
        labels = ["FDA_APPROVED", "CT_TOX"]

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()
            smiles = smiles_list[i]
            y = data_df.iloc[i][labels]

            graph = self.smiles2graph(smiles)
            cliques = motif_decomp(smiles)

            if graph is not None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.cliques = cliques
                    data.symbol = graph['atom_symbol']
                    data.smiles = smiles  # **保存SMILES字符串**
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        # 先提取所有smiles
        smiles_all = [d.smiles for d in self]

        # Scaffold划分

        def scaffold(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

        scaffolds = {}
        for i, s in enumerate(smiles_all):
            scaf = scaffold(s)
            if scaf not in scaffolds:
                scaffolds[scaf] = [i]
            else:
                scaffolds[scaf].append(i)

        scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)

        train_idx, val_idx, test_idx = [], [], []
        num_train, num_val = train_size, valid_size
        for scaf_set in scaffold_sets:
            if len(train_idx) + len(scaf_set) <= num_train:
                train_idx += scaf_set
            elif len(val_idx) + len(scaf_set) <= num_val:
                val_idx += scaf_set
            else:
                test_idx += scaf_set

        return {
            'train': torch.tensor(train_idx, dtype=torch.long),
            'valid': torch.tensor(val_idx, dtype=torch.long),
            'test': torch.tensor(test_idx, dtype=torch.long)
        }



class ClinTox1Dataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "ClinTox")
        self.task_type = "classification"
        self.num_tasks = 2
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "clintox.csv.gz"

    @property
    def processed_file_names(self):
        return "clintox_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "clintox.csv.gz")
        )
        smiles_list = data_df["smiles"]
        lable = ["FDA_APPROVED",
                 "CT_TOX"]


        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            if graph != None and cliques:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data.cliques = cliques
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time()-t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class SiderDataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Sider")
        self.task_type = "classification"
        self.num_tasks = 27
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "Sider.csv.gz"

    @property
    def processed_file_names(self):
        return "sider_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "sider.csv.gz")
        )
        smiles_list = data_df["smiles"]
        lable = ["Hepatobiliary disorders",
                 "Metabolism and nutrition disorders",
                 "Product issues",
                 "Eye disorders",
                 "Investigations",
                 "Musculoskeletal and connective tissue disorders",
                 "Gastrointestinal disorders",
                 "Social circumstances",
                 "Immune system disorders",
                 "Reproductive system and breast disorders",
                 "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                 "General disorders and administration site conditions",
                 "Endocrine disorders",
                 "Surgical and medical procedures",
                 "Vascular disorders",
                 "Blood and lymphatic system disorders",
                 "Skin and subcutaneous tissue disorders",
                 "Congenital, familial and genetic disorders",
                 "Infections and infestations",
                 "Respiratory, thoracic and mediastinal disorders",
                 "Psychiatric disorders",
                 "Renal and urinary disorders",
                 "Pregnancy, puerperium and perinatal conditions",
                 "Ear and labyrinth disorders",
                 "Cardiac disorders",
                 "Nervous system disorders",
                 "Injury, poisoning and procedural complications",
                 ]


        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            if graph != None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.cliques = cliques
                    data.symbol = (graph['atom_symbol'])
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time()-t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict
class Sider1Dataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Sider")
        self.task_type = "classification"
        self.num_tasks = 27
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "Sider.csv.gz"

    @property
    def processed_file_names(self):
        return "sider_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "sider.csv.gz")
        )
        smiles_list = data_df["smiles"]
        lable = ["Hepatobiliary disorders",
                 "Metabolism and nutrition disorders",
                 "Product issues",
                 "Eye disorders",
                 "Investigations",
                 "Musculoskeletal and connective tissue disorders",
                 "Gastrointestinal disorders",
                 "Social circumstances",
                 "Immune system disorders",
                 "Reproductive system and breast disorders",
                 "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                 "General disorders and administration site conditions",
                 "Endocrine disorders",
                 "Surgical and medical procedures",
                 "Vascular disorders",
                 "Blood and lymphatic system disorders",
                 "Skin and subcutaneous tissue disorders",
                 "Congenital, familial and genetic disorders",
                 "Infections and infestations",
                 "Respiratory, thoracic and mediastinal disorders",
                 "Psychiatric disorders",
                 "Renal and urinary disorders",
                 "Pregnancy, puerperium and perinatal conditions",
                 "Ear and labyrinth disorders",
                 "Cardiac disorders",
                 "Nervous system disorders",
                 "Injury, poisoning and procedural complications",
                 ]


        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            if graph != None and cliques:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data.cliques = cliques
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time()-t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class Tox21Dataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Tox21")
        self.task_type = "classification"
        self.num_tasks = 12
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "tox21.csv.gz"

    @property
    def processed_file_names(self):
        return "tox21_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "tox21.csv.gz")
        )
        smiles_list = data_df["smiles"]
        lable = ["NR-AR",
                 "NR-AR-LBD",
                 "NR-AhR",
                 "NR-Aromatase",
                 "NR-ER",
                 "NR-ER-LBD",
                 "NR-PPAR-gamma",
                 "SR-ARE",
                 "SR-ATAD5",
                 "SR-HSE",
                 "SR-MMP",
                 "SR-p53"]

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            if graph != None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:

                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1
                    assert graph["edge_index"].shape[1] == (graph["num_nodes"]-1)*2

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.cliques = cliques
                    data.symbol = (graph['atom_symbol'])
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict
class Tox211Dataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Tox21")
        self.task_type = "classification"
        self.num_tasks = 12
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "tox21.csv.gz"

    @property
    def processed_file_names(self):
        return "tox21_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "tox21.csv.gz")
        )
        smiles_list = data_df["smiles"]
        lable = ["NR-AR",
                 "NR-AR-LBD",
                 "NR-AhR",
                 "NR-Aromatase",
                 "NR-ER",
                 "NR-ER-LBD",
                 "NR-PPAR-gamma",
                 "SR-ARE",
                 "SR-ATAD5",
                 "SR-HSE",
                 "SR-MMP",
                 "SR-p53"]

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            if graph != None and cliques:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data.cliques = cliques
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class HIVDataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "HIV")
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "HIV.csv"

    @property
    def processed_file_names(self):
        return "HIV_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "HIV.csv")
        )
        smiles_list = data_df["smiles"]
        lable = ["HIV_active"]

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            if graph != None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.symbol = (graph['atom_symbol'])
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict
class HIV1Dataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "HIV")
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "HIV.csv"

    @property
    def processed_file_names(self):
        return "HIV_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "HIV.csv")
        )
        smiles_list = data_df["smiles"]
        lable = ["HIV_active"]

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            if graph != None and cliques:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data.cliques = cliques
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class BaceDataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Bace")
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "bace.csv"

    @property
    def processed_file_names(self):
        return "bace_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "bace.csv")
        )
        smiles_list = data_df["mol"]
        lable = ["Class"]

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)

            if graph != None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:

                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.symbol = (graph['atom_symbol'])
                    data.cliques = cliques

                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class Bace1Dataset(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=smiles2graph,
                 transform=None,
                 pre_transform=None):

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Bace")
        self.task_type = "classification"
        self.num_tasks = 1
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "bace.csv"

    @property
    def processed_file_names(self):
        return "bace_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "bace.csv")
        )
        smiles_list = data_df["mol"]
        lable = ["Class"]

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)

            if graph != None and cliques:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data.cliques = cliques
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class EsolDataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset",
            smiles2graph=smiles2graph,
            transform=None,
            pre_transform=None
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Esol")
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "delaney-processed.csv"

    @property
    def processed_file_names(self):
        return "Esol_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "delaney-processed.csv")
        )
        smiles_list = data_df["smiles"]
        lable = ["measured log solubility in mols per litre"]
        data_df.loc[:, lable] = data_df.loc[:, lable].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            if graph != None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:

                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.cliques = cliques
                    data.symbol = (graph['atom_symbol'])
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict
class Esol1Dataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset",
            smiles2graph=smiles2graph,
            transform=None,
            pre_transform=None
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Esol")
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "delaney-processed.csv"

    @property
    def processed_file_names(self):
        return "Esol_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "delaney-processed.csv")
        )
        smiles_list = data_df["smiles"]
        lable = ["measured log solubility in mols per litre"]
        data_df.loc[:, lable] = data_df.loc[:, lable].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            if graph != None and cliques:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data.cliques = cliques
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class FreeSolvDataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset",
            smiles2graph=smiles2graph,
            transform=None,
            pre_transform=None
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "FreeSolv")
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "SAMPL.csv"

    @property
    def processed_file_names(self):
        return "FreeSolv_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "SAMPL.csv")
        )
        smiles_list = data_df["smiles"]
        lable = ["expt"]
        data_df.loc[:, lable] = data_df.loc[:, lable].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            if graph != None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.symbol = (graph['atom_symbol'])
                    data.cliques = cliques
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict
class FreeSolv1Dataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset",
            smiles2graph=smiles2graph,
            transform=None,
            pre_transform=None
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "FreeSolv")
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "SAMPL.csv"

    @property
    def processed_file_names(self):
        return "FreeSolv_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "SAMPL.csv")
        )
        smiles_list = data_df["smiles"]
        lable = ["expt"]
        data_df.loc[:, lable] = data_df.loc[:, lable].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            if graph != None and cliques:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data.cliques = cliques
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict

class LipopDataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset",
            smiles2graph=smiles2graph,
            transform=None,
            pre_transform=None
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Lipop")
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "Lipophilicity.csv"

    @property
    def processed_file_names(self):
        return "Lipop_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "Lipophilicity.csv")
        )
        smiles_list = data_df["smiles"]
        lable = ["exp"]
        data_df.loc[:, lable] = data_df.loc[:, lable].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)

            if graph != None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.cliques = cliques
                    data.symbol = (graph['atom_symbol'])
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict
class Lipop1Dataset(InMemoryDataset):
    def __init__(
            self,
            root="dataset",
            smiles2graph=smiles2graph,
            transform=None,
            pre_transform=None
    ):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Lipop")
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "Lipophilicity.csv"

    @property
    def processed_file_names(self):
        return "Lipop_precessed.pt"

    def download(self):
        file_path = download_url(self.url, self.raw_dir)
        return file_path

    def process(self):
        data_df = pd.read_csv(
            osp.join(self.raw_dir, "Lipophilicity.csv")
        )
        smiles_list = data_df["smiles"]
        lable = ["exp"]
        data_df.loc[:, lable] = data_df.loc[:, lable].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            y = data_df.iloc[i][lable]
            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)
            if graph != None and cliques:

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_node__ = int(graph["num_nodes"])
                data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.y = torch.Tensor([y])
                data.cliques = cliques
                data.smiles = smiles
                data_list.append(data)
            else:
                continue

        t0 = time.time()
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.preprocessing_time = time.time() - t0

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict



######将 BACE，Sider， ESOL， Lipop， Freesolv这几个数据集按照BBBP_scaffold的形式修改，不要再原代码上修改，新写一个****_scaffold类
class BACEDataset_scaffold(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=None,
                 transform=None,
                 pre_transform=None):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Bace")
        self.task_type = "classification"
        self.num_tasks = 1  # BACE为二分类任务
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"  # BACE数据集URL
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "bace.csv"

    @property
    def processed_file_names(self):
        return "bace_processed.pt"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "bace.csv"))
        smiles_list = data_df["mol"]
        labels = ["Class"]  # BACE标签列名

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()
            smiles = smiles_list[i]
            y = data_df.iloc[i][labels]

            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)  # 假设motif_decomp函数已在utils中定义

            if graph is not None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.cliques = cliques
                    data.symbol = graph['atom_symbol']
                    data.smiles = smiles  # 保存SMILES用于scaffold划分
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        # 提取所有SMILES用于scaffold划分
        smiles_all = [d.smiles for d in self]
        labels = torch.cat([d.y for d in self], dim=0)

        def scaffold(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

        # 按scaffold分组
        scaffolds = {}
        for i, s in enumerate(smiles_all):
            scaf = scaffold(s)
            scaffolds.setdefault(scaf, []).append(i)

        scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)

        train_idx, val_idx, test_idx = [], [], []
        num_train, num_val = train_size, valid_size
        for scaf_set in scaffold_sets:
            if len(train_idx) + len(scaf_set) <= num_train:
                train_idx += scaf_set
            elif len(val_idx) + len(scaf_set) <= num_val:
                val_idx += scaf_set
            else:
                test_idx += scaf_set

        train_idx, val_idx, test_idx = map(lambda x: torch.tensor(x, dtype=torch.long),
                                        [train_idx, val_idx, test_idx])

        # 检查并确保每个split包含正负样本（分类任务）
        def ensure_pos_neg(idx):
            y_split = labels[idx]
            need_fix = []
            for task in range(y_split.shape[1] if y_split.dim() > 1 else 1):
                y_t = y_split[:, task] if y_split.dim() > 1 else y_split
                if (y_t == 1).sum() == 0 or (y_t == 0).sum() == 0:
                    need_fix.append(task)
            return need_fix

        # 修复val和test集的类别不平衡
        for split_name, idx_var in [('valid', val_idx), ('test', test_idx)]:
            need_fix = ensure_pos_neg(idx_var)
            if need_fix:
                for task in need_fix:
                    y_train = labels[train_idx][:, task]
                    y_split = labels[idx_var][:, task]
                    if (y_split == 1).sum() == 0:  # 无正样本
                        pos_candidates = (y_train == 1).nonzero(as_tuple=True)[0]
                        if len(pos_candidates) > 0:
                            move_idx = pos_candidates[:max(1, len(pos_candidates)//10)]
                            idx_var = torch.cat([idx_var, train_idx[move_idx]])
                            mask = torch.ones(len(train_idx), dtype=torch.bool)
                            mask[move_idx] = False
                            train_idx = train_idx[mask]
                    if (y_split == 0).sum() == 0:  # 无负样本
                        neg_candidates = (y_train == 0).nonzero(as_tuple=True)[0]
                        if len(neg_candidates) > 0:
                            move_idx = neg_candidates[:max(1, len(neg_candidates)//10)]
                            idx_var = torch.cat([idx_var, train_idx[move_idx]])
                            mask = torch.ones(len(train_idx), dtype=torch.bool)
                            mask[move_idx] = False
                            train_idx = train_idx[mask]
                if split_name == 'valid':
                    val_idx = idx_var
                else:
                    test_idx = idx_var

        return {'train': train_idx, 'valid': val_idx, 'test': test_idx}


class SiderDataset_scaffold(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=None,
                 transform=None,
                 pre_transform=None):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Sider")
        self.task_type = "classification"
        self.num_tasks = 27  # Sider有27个任务
        self.eval_metric = "rocauc"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "sider.csv.gz"

    @property
    def processed_file_names(self):
        return "sider_processed.pt"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "sider.csv.gz"))
        smiles_list = data_df["smiles"]
        # Sider的27个标签列
        labels = ["Hepatobiliary disorders",
                 "Metabolism and nutrition disorders",
                 "Product issues",
                 "Eye disorders",
                 "Investigations",
                 "Musculoskeletal and connective tissue disorders",
                 "Gastrointestinal disorders",
                 "Social circumstances",
                 "Immune system disorders",
                 "Reproductive system and breast disorders",
                 "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
                 "General disorders and administration site conditions",
                 "Endocrine disorders",
                 "Surgical and medical procedures",
                 "Vascular disorders",
                 "Blood and lymphatic system disorders",
                 "Skin and subcutaneous tissue disorders",
                 "Congenital, familial and genetic disorders",
                 "Infections and infestations",
                 "Respiratory, thoracic and mediastinal disorders",
                 "Psychiatric disorders",
                 "Renal and urinary disorders",
                 "Pregnancy, puerperium and perinatal conditions",
                 "Ear and labyrinth disorders",
                 "Cardiac disorders",
                 "Nervous system disorders",
                 "Injury, poisoning and procedural complications"]

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()
            smiles = smiles_list[i]
            y = data_df.iloc[i][labels]

            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)

            if graph is not None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.cliques = cliques
                    data.symbol = graph['atom_symbol']
                    data.smiles = smiles
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        smiles_all = [d.smiles for d in self]
        labels = torch.cat([d.y for d in self], dim=0)

        def scaffold(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

        scaffolds = defaultdict(list)
        for i, s in enumerate(smiles_all):
            scaf = scaffold(s)
            scaffolds[scaf].append(i)

        scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)

        train_idx, val_idx, test_idx = [], [], []
        num_train, num_val = train_size, valid_size
        for scaf_set in scaffold_sets:
            if len(train_idx) + len(scaf_set) <= num_train:
                train_idx += scaf_set
            elif len(val_idx) + len(scaf_set) <= num_val:
                val_idx += scaf_set
            else:
                test_idx += scaf_set

        train_idx, val_idx, test_idx = map(lambda x: torch.tensor(x, dtype=torch.long),
                                        [train_idx, val_idx, test_idx])

        # 多任务分类的正负样本检查
        def ensure_pos_neg(idx):
            y_split = labels[idx]
            need_fix = []
            for task in range(y_split.shape[1]):
                y_t = y_split[:, task]
                if (y_t == 1).sum() == 0 or (y_t == 0).sum() == 0:
                    need_fix.append(task)
            return need_fix

        for split_name, idx_var in [('valid', val_idx), ('test', test_idx)]:
            need_fix = ensure_pos_neg(idx_var)
            if need_fix:
                for task in need_fix:
                    y_train = labels[train_idx][:, task]
                    y_split = labels[idx_var][:, task]
                    if (y_split == 1).sum() == 0:
                        pos_candidates = (y_train == 1).nonzero(as_tuple=True)[0]
                        if len(pos_candidates) > 0:
                            move_idx = pos_candidates[:max(1, len(pos_candidates)//10)]
                            idx_var = torch.cat([idx_var, train_idx[move_idx]])
                            mask = torch.ones(len(train_idx), dtype=torch.bool)
                            mask[move_idx] = False
                            train_idx = train_idx[mask]
                    if (y_split == 0).sum() == 0:
                        neg_candidates = (y_train == 0).nonzero(as_tuple=True)[0]
                        if len(neg_candidates) > 0:
                            move_idx = neg_candidates[:max(1, len(neg_candidates)//10)]
                            idx_var = torch.cat([idx_var, train_idx[move_idx]])
                            mask = torch.ones(len(train_idx), dtype=torch.bool)
                            mask[move_idx] = False
                            train_idx = train_idx[mask]
                if split_name == 'valid':
                    val_idx = idx_var
                else:
                    test_idx = idx_var

        return {'train': train_idx, 'valid': val_idx, 'test': test_idx}


class ESOLDataset_scaffold(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=None,
                 transform=None,
                 pre_transform=None):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Esol")
        self.task_type = "mse_regression"  # ESOL是回归任务
        self.num_tasks = 1
        self.eval_metric = "rmse"  # 回归常用MAE评估
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/esol.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "esol.csv"

    @property
    def processed_file_names(self):
        return "esol_processed.pt"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "esol.csv"))
        smiles_list = data_df["smiles"]
        labels = ["measured log solubility in mols per litre"]  # ESOL标签列

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()
            smiles = smiles_list[i]
            y = data_df.iloc[i][labels]

            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)

            if graph is not None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.cliques = cliques
                    data.symbol = graph['atom_symbol']
                    data.smiles = smiles
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        # 回归任务无需正负样本检查，仅保留scaffold划分
        smiles_all = [d.smiles for d in self]

        def scaffold(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

        scaffolds = defaultdict(list)
        for i, s in enumerate(smiles_all):
            scaf = scaffold(s)
            scaffolds[scaf].append(i)

        scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)

        train_idx, val_idx, test_idx = [], [], []
        num_train, num_val = train_size, valid_size
        for scaf_set in scaffold_sets:
            if len(train_idx) + len(scaf_set) <= num_train:
                train_idx += scaf_set
            elif len(val_idx) + len(scaf_set) <= num_val:
                val_idx += scaf_set
            else:
                test_idx += scaf_set

        return {
            'train': torch.tensor(train_idx, dtype=torch.long),
            'valid': torch.tensor(val_idx, dtype=torch.long),
            'test': torch.tensor(test_idx, dtype=torch.long)
        }


class LipopDataset_scaffold(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=None,
                 transform=None,
                 pre_transform=None):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Lipop")
        self.task_type = "mse_regression"  # Lipop是回归任务
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipop.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "Lipop.csv"

    @property
    def processed_file_names(self):
        return "lipop_processed.pt"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "Lipop.csv"))
        smiles_list = data_df["smiles"]
        labels = ["exp"]  # Lipop标签列

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()
            smiles = smiles_list[i]
            y = data_df.iloc[i][labels]

            graph = smiles2graph(smiles)
            cliques = motif_decomp(smiles)

            if graph is not None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.cliques = cliques
                    data.symbol = graph['atom_symbol']
                    data.smiles = smiles
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        # 回归任务scaffold划分
        smiles_all = [d.smiles for d in self]

        def scaffold(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

        scaffolds = defaultdict(list)
        for i, s in enumerate(smiles_all):
            scaf = scaffold(s)
            scaffolds[scaf].append(i)

        scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)

        train_idx, val_idx, test_idx = [], [], []
        num_train, num_val = train_size, valid_size
        for scaf_set in scaffold_sets:
            if len(train_idx) + len(scaf_set) <= num_train:
                train_idx += scaf_set
            elif len(val_idx) + len(scaf_set) <= num_val:
                val_idx += scaf_set
            else:
                test_idx += scaf_set

        return {
            'train': torch.tensor(train_idx, dtype=torch.long),
            'valid': torch.tensor(val_idx, dtype=torch.long),
            'test': torch.tensor(test_idx, dtype=torch.long)
        }


class FreesolvDataset_scaffold(InMemoryDataset):
    def __init__(self,
                 root="dataset",
                 smiles2graph=None,
                 transform=None,
                 pre_transform=None):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "Freesolv")
        self.task_type = "mse_regression"  # Freesolv是回归任务
        self.num_tasks = 1
        self.eval_metric = "rmse"
        self.root = root

        self.url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/freesolv.csv"
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "freesolv.csv"

    @property
    def processed_file_names(self):
        return "freesolv_processed.pt"

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "freesolv.csv"))
        smiles_list = data_df["smiles"]
        labels = ["expt"]  # Freesolv标签列

        print("Converting SMILES string into graphs ...")
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()
            smiles = smiles_list[i]
            y = data_df.iloc[i][labels]

            graph = smiles2graph(smiles)
            # graph = self.smiles2graph(smiles)这里把self去掉
            cliques = motif_decomp(smiles)

            if graph is not None and graph['edge_index'].shape[1] != 0 and graph['num_nodes'] > 2 and cliques:
                try:
                    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                    assert len(graph["node_feat"]) == graph["num_nodes"]
                    assert len(graph["node_feat"]) == graph["edge_index"].max() + 1

                    data.__num_node__ = int(graph["num_nodes"])
                    data.edge_index = torch.unique(torch.from_numpy(graph["edge_index"]).to(torch.int64), dim=1)
                    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
                    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                    data.y = torch.Tensor([y])
                    data.cliques = cliques
                    data.symbol = graph['atom_symbol']
                    data.smiles = smiles
                    data_list.append(data)
                except AssertionError:
                    continue
            else:
                continue

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        # 回归任务scaffold划分
        smiles_all = [d.smiles for d in self]

        def scaffold(smiles):
            mol = Chem.MolFromSmiles(smiles)
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

        scaffolds = defaultdict(list)
        for i, s in enumerate(smiles_all):
            scaf = scaffold(s)
            scaffolds[scaf].append(i)

        scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)

        train_idx, val_idx, test_idx = [], [], []
        num_train, num_val = train_size, valid_size
        for scaf_set in scaffold_sets:
            if len(train_idx) + len(scaf_set) <= num_train:
                train_idx += scaf_set
            elif len(val_idx) + len(scaf_set) <= num_val:
                val_idx += scaf_set
            else:
                test_idx += scaf_set

        return {
            'train': torch.tensor(train_idx, dtype=torch.long),
            'valid': torch.tensor(val_idx, dtype=torch.long),
            'test': torch.tensor(test_idx, dtype=torch.long)
        }


class Qm9dataset(InMemoryDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        self.task_type = "mse_regression"
        self.num_tasks = 1
        self.eval_metric = 'mae'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        try:
            import rdkit  # noqa
            return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
        except ImportError:
            return ['qm9_v3.pt']

    @property
    def processed_file_names(self) -> str:
        return 'data_v3.pt'

    def download(self):
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])

            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        with open(self.raw_paths[1], 'r') as f:
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)
        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            atom_symbol_list = []
            if i in skip:
                continue


            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))
                symbol = atom.GetSymbol()
                atom_symbol_list.append(symbol)
            # z = torch.tensor(atomic_number, dtype=torch.long)

            num_bond_features = 3  # bond type, bond stereo, is_conjugated
            if len(mol.GetBonds()) > 0:  # mol has bonds
                edges_list = []
                edge_features_list = []
                for bond in mol.GetBonds():
                    m = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    edge_feature = bond_to_feature_vector(bond)

                    # add edges in both directions
                    edges_list.append((m, j))
                    edge_features_list.append(edge_feature)
                    edges_list.append((j, m))
                    edge_features_list.append(edge_feature)

                # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
                edge_index = torch.tensor(edges_list, dtype=torch.int64).T

                # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
                edge_attr = torch.tensor(edge_features_list, dtype=torch.int64)
            else:  # mol has no bonds
                edge_index = torch.tensor((2, 0), dtype=torch.int64)
                edge_attr = torch.tensor((0, num_bond_features), dtype=torch.int64)


            x = torch.tensor(atom_features_list, dtype=torch.int64)


            y = target[i][:12].unsqueeze(0)
            name = mol.GetProp('_Name')

            data = Data(x=x, edge_index=edge_index, pos=pos,
                        edge_attr=edge_attr, y=y, name=name, idx=i, symbol=atom_symbol_list)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def get_idx_split(self, data_size, train_size, valid_size, seed):
        ids = shuffle(range(data_size), random_state=seed)
        train_idx, val_idx, test_idx = torch.tensor(ids[:train_size]), torch.tensor(
            ids[train_size:train_size + valid_size]), torch.tensor(ids[train_size + valid_size:])
        split_dict = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return split_dict



####在最下面的有个get_dataset.（具体函数名称我忘记了）在着其中更新调用类，就可以了
def get_dataset(dataset, output_dir="./"):
    print(f"Preprocessing {dataset}".upper())
    if not os.path.exists(os.path.join(output_dir, "dataset")):
        os.makedirs(os.path.join(output_dir, "dataset"))
    root = os.path.join(
        output_dir, "dataset", dataset)

    if dataset == "BBBP":
        mp = "meta_paths_{}".format(dataset)
        data = BBBPDataset_scaffold(
            root=root, pre_transform=Meta_pathTransform(mp)
        )
    elif dataset == "BBBP1":
        data = BBBPD1ataset(
            root=root, pre_transform=None
        )
    elif dataset == "ClinTox":
        mp = "meta_paths_{}".format(dataset)
        data = ClinToxDataset_scaffold(
            root=root, pre_transform=Meta_pathTransform(mp)
            # root=root, pre_transform=None
        )
    elif dataset == "ClinTox1":
        data = ClinTox1Dataset(
            root=root, pre_transform=None
        )
    elif dataset == "Tox21":
        mp = "meta_paths_{}".format(dataset)
        data = Tox21Dataset(
            root=root, pre_transform=Meta_pathTransform(mp)
        )
    elif dataset == "Tox211":
        data = Tox211Dataset(
            root=root, pre_transform=None
        )
    elif dataset == "Sider":
        mp = "meta_paths_{}".format(dataset)
        data = SiderDataset_scaffold(
            # root=root, pre_transform=None
            root=root, pre_transform=Meta_pathTransform(mp)
        )
    elif dataset == "Sider1":
        data = Sider1Dataset(
            root=root, pre_transform=None
        )
    elif dataset == "HIV":
        data = HIVDataset(
            root=root, pre_transform=None
        )
    elif dataset == "HIV1":
        data = HIV1Dataset(
            root=root, pre_transform=None
        )
    elif dataset == "Esol":
        mp = "meta_paths_{}".format(dataset)
        data = ESOLDataset_scaffold(
            root=root, pre_transform=Meta_pathTransform(mp)
        )
    elif dataset == "Esol1":
        data = Esol1Dataset(
            root=root, pre_transform=None
        )
    elif dataset == "Freesolv":
        mp = "meta_paths_{}".format(dataset)
        data = FreesolvDataset_scaffold(
            root=root, pre_transform=Meta_pathTransform(mp)
        )
    elif dataset == "Freesolv1":
        data = FreeSolv1Dataset(
            root=root, pre_transform=None
        )
    elif dataset == "Lipop":
        mp = "meta_paths_{}".format(dataset)
        data = LipopDataset_scaffold(
            root=root, pre_transform=Meta_pathTransform(mp)
        )
    elif dataset == "Lipop1":
        data = Lipop1Dataset(
            root=root, pre_transform=None
        )
    elif dataset == "Bace":
        mp = "meta_paths_{}".format(dataset)
        data = BACEDataset_scaffold(
            root=root, pre_transform=Meta_pathTransform(mp)
        )
    elif dataset == "qm9":
        mp = "meta_paths_{}".format(dataset)
        data = Qm9dataset(
            root=root, pre_transform=Meta_pathTransform(mp)
        )
    elif dataset == "Bace1":
        data = Bace1Dataset(
            root=root, pre_transform=None
        )
    elif dataset =="mutag":
        data = TUDataset(root=root, name='MUTAG', use_node_attr=True)
    return data



# def get_dataset(dataset, output_dir="./"):
#     print(f"Preprocessing {dataset}".upper())
#     if not os.path.exists(os.path.join(output_dir, "dataset")):
#         os.makedirs(os.path.join(output_dir, "dataset"))
#     root = os.path.join(
#         output_dir, "dataset", dataset)
#
#     if dataset == "BBBP":
#         mp = "meta_paths_{}".format(dataset)
#         data = BBBPDataset_scaffold(
#             root=root, pre_transform=Meta_pathTransform(mp)
#         )
#     elif dataset == "BBBP1":
#         data = BBBPD1ataset(
#             root=root, pre_transform=None
#         )
#     elif dataset == "ClinTox":
#         mp = "meta_paths_{}".format(dataset)
#         data = ClinToxDataset_scaffold(
#             root=root, pre_transform=Meta_pathTransform(mp)
#             # root=root, pre_transform=None
#         )
#     elif dataset == "ClinTox1":
#         data = ClinTox1Dataset(
#             root=root, pre_transform=None
#         )
#     elif dataset == "Tox21":
#         mp = "meta_paths_{}".format(dataset)
#         data = Tox21Dataset(
#             root=root, pre_transform=Meta_pathTransform(mp)
#         )
#     elif dataset == "Tox211":
#         data = Tox211Dataset(
#             root=root, pre_transform=None
#         )
#     elif dataset == "Sider":
#         mp = "meta_paths_{}".format(dataset)
#         data = SiderDataset(
#             # root=root, pre_transform=None
#             root=root, pre_transform=Meta_pathTransform(mp)
#         )
#     elif dataset == "Sider1":
#         data = Sider1Dataset(
#             root=root, pre_transform=None
#         )
#     elif dataset == "HIV":
#         data = HIVDataset(
#             root=root, pre_transform=None
#         )
#     elif dataset == "HIV1":
#         data = HIV1Dataset(
#             root=root, pre_transform=None
#         )
#     elif dataset == "Esol":
#         mp = "meta_paths_{}".format(dataset)
#         data = EsolDataset(
#             root=root, pre_transform=Meta_pathTransform(mp)
#         )
#     elif dataset == "Esol1":
#         data = Esol1Dataset(
#             root=root, pre_transform=None
#         )
#     elif dataset == "Freesolv":
#         mp = "meta_paths_{}".format(dataset)
#         data = FreeSolvDataset(
#             root=root, pre_transform=Meta_pathTransform(mp)
#         )
#     elif dataset == "Freesolv1":
#         data = FreeSolv1Dataset(
#             root=root, pre_transform=None
#         )
#     elif dataset == "Lipop":
#         mp = "meta_paths_{}".format(dataset)
#         data = LipopDataset(
#             root=root, pre_transform=Meta_pathTransform(mp)
#         )
#     elif dataset == "Lipop1":
#         data = Lipop1Dataset(
#             root=root, pre_transform=None
#         )
#     elif dataset == "Bace":
#         mp = "meta_paths_{}".format(dataset)
#         data = BaceDataset(
#             root=root, pre_transform=Meta_pathTransform(mp)
#         )
#     elif dataset == "Bace1":
#         data = Bace1Dataset(
#             root=root, pre_transform=None
#         )
#     elif dataset =="mutag":
#         data = TUDataset(root=root, name='MUTAG', use_node_attr=True)
#     return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbgmol* data with Pytorch Geometrics"
    )
    parser.add_argument("--dataset", type=str, default="ogbg-molhiv")
    parser.add_argument("--min_cutoff", type=int, default=3)
    parser.add_argument(
        "--max_cutoff", type=int, default=5, help="Max length of shortest paths"
    )
    args = parser.parse_args()

    for cutoff in range(args.min_cutoff, args.max_cutoff + 1):
        # for dataset in ["ogbg-molhiv", "ogbg-molpcba", "ZINC", "peptides-functional", "peptides-structural"] :
        for dataset in [args.dataset]:

            for path_type in [
                "shortest_path",
                "all_shortest_paths",
                "all_simple_paths",
            ]:

                data = get_dataset(dataset, cutoff, path_type)
                print(data.preprocessing_time)
