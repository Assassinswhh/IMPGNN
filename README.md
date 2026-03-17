This repository is the official implementation of the paper  
"**Bond-aware Molecular Graph Learning with Multi-graph Interleaved Message Passing**".

---

## Introduction

We propose a novel graph neural network framework for molecular property prediction that explicitly models **bond heterogeneity** in molecular graphs.
<img width="1754" height="471" alt="image" src="https://github.com/user-attachments/assets/2c7e3aed-6991-478f-9aa6-a1c267c39d91" />


Unlike conventional approaches that treat molecules as homogeneous graphs, our method constructs **bond-centric graph representations** to better capture the diverse types of chemical bonds. Based on this design, we introduce a **multi-graph learning framework** that leverages three complementary views of a molecule:

- Atom graph
- Bond graph
- Augmented bond graph

To enable effective interaction across these views, we propose an **Interleaved Message Passing Graph Neural Network (IMPGNN)**, which performs message passing not only within each graph but also **across different graph views during node representation learning**. This design allows the model to capture richer structural and relational information compared to traditional multi-view methods that rely on late-stage fusion.
。

---

## Environment

It is recommended to use **Anaconda** or **Miniconda** to manage the environment.

### Requirements

- python >= 3.9.0
- pytorch >= 2.0
- torch_geometric >= 2.3.0
- rdkit >= 2022.09.1
- numpy == 1.23.5
- pandas == 1.5.3
- scipy == 1.11.4

---

## Datasets

we used eight benchmark datasets from MoleculeNet and the QM9 dataset。

### Tasks

- **Classification Tasks**: evaluated using ROC-AUC  
- **Regression Tasks**: evaluated using RMSE / MAE  

All datasets will be automatically downloaded and processed during the first run.


