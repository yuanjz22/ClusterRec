# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import faiss
import torch.nn.functional as F

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ex. target_word: .csv / in target_path find 123.csv file
def find_filepath(target_path, target_word):
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)
            
    return file_paths


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu", niter=20):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size,niter=niter)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        index = faiss.IndexFlatL2(hidden_size)
        return clus, index

    def train(self, x):
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        centroids = torch.tensor(centroids, requires_grad=True).to(self.device)
        # self.centroids = nn.functional.normalize(centroids, p=2, dim=1)
        self.centroids = centroids

    def query(self, x):
        D, I = self.index.search(x, 1)  
        seq2cluster = [int(n[0]) for n in I]
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]



# class FusionGate(nn.Module):
#     def __init__(self, hidden_dim):
#         super(FusionGate, self).__init__()
#         # 输入维度是 2 * hidden_dim (因为是 concat)
#         # 输出维度是 hidden_dim (生成 gate 向量)
#         self.gate_layer = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.Sigmoid()  # 关键：输出限制在 (0, 1)
#         )
        
#         # 可选：初始化偏置为 0，让初始状态接近 0.5 (完全混合)
#         # 或者初始化为负数，让模型初始偏向某一方
#         nn.init.xavier_uniform_(self.gate_layer[0].weight)
#         nn.init.zeros_(self.gate_layer[0].bias)

#     def forward(self, h_group, h_sas):
        
#         combined = torch.cat([h_group, h_sas], dim=-1)  # [Batch, 2*D]
        
#         g = self.gate_layer(combined)  # [Batch, D]
        
#         u_final = g * h_group + (1 - g) * h_sas
        
#         return u_final, g

    
    
    