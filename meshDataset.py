# Dataset generates data from the mesh file using the SDFSampler library on CPU
# Moving data generation to GPU should speed up this process significantly
import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from torchgp.point_sample import point_sample
from utils.distance_utils import *

class MeshDataset(Dataset):
    def __init__(self,sphereSampler,samplenum=250000,verbose=True, sphere32 = None,inner_sphere = None, k=4):
        self.k = k
        self.sphere_sampler = sphereSampler
        # self.safeSpheres = safeSpheres
        self.sample_num = samplenum
        self.mesh = sphereSampler.mesh
        self.importance_sampler = sphereSampler.importanceSampler
        self.sample_list = ['rand','importance','importance','importance','importance','importance','importance','importance','importance','rand','trace']
        # self.sample_list = ['rand', 'rand', 'rand', 'rand', 'rand', 'rand']
        # self.sample_list = [ 'trace', 'trace']
        mesh_name = os.path.split(sphereSampler.mesh_file)[1]
        self.sphere32 = sphere32
        self.inner_sphere = inner_sphere
        if (verbose):
            logging.info("Loaded " + mesh_name)

        # vertices, faces = MeshLoader.read(voxelFile)
        # normalizeMeshToSphere(vertices, faces)

        # sampler = PointSampler(vertices, faces)
        ##########################################################

        ##########################################################
        # Testing indicated very poor SDF accuracy outside the mesh boundary which complicated
        # raymarching operations.
        # Adding samples through the unit sphere improves accuracy farther from the boundary,
        # but still within the unit sphere
        #  general_points = sampler.sample(int((1-boundary_ratio)*num_samples), 1)
        #ptsNp = readTest.doSample(sphereFile, voxelFile, num_samples)
        self.resample()
        # self.computeKnn()

    def getAllData(self):
        return self.trainData.numpy()

    def computeKnn(self):
        self.nearst_info = query_k_point(k = self.k, xyz = self.trainData[:, :3], new_xyz = self.sphere32[:, :3], sphere_r = self.sphere32[:, 3])

    def resample(self):
        f = torch.from_numpy(np.array(self.mesh.F())).long()
        v = torch.from_numpy(np.array(self.mesh.V()))
        self.queryPts , sphere_list = point_sample(v, f, self.sphere32,self.inner_sphere,self.sample_list,self.importance_sampler, self.sample_num)


        # self.queryPts  = point_sample(v, f, self.sphere32,self.sample_list,self.importance_sampler, self.sample_num).detach().cpu().numpy()
        split = np.array_split(self.queryPts,10,axis=0)
        S = []
        # self.sphere_sampler.sdf.
        print("开始计算sdf真值")
        for u in tqdm(split):
            S.append(self.sphere_sampler.sdf.query(u))
        S = np.concatenate(S,axis=0)
        # half = int(len(self.queryPts)/2)
        # s1 = self.sphere_sampler.sdf.query(self.queryPts[:half])
        # s2 = self.sphere_sampler.sdf.query(self.queryPts[half:])
        # S = np.concatenate([s1,s2])
        # S = self.sphere_sampler.sdf.query(self.queryPts)
        # trainData = np.concatenate((self.queryPts, S), axis=1)
        trainData = np.concatenate((self.queryPts, sphere_list), axis=1)
        # trainData =  np.concatenate((trainData,S),axis=1)
        trainData = np.concatenate((trainData, S.reshape(-1, 1)), axis=1)
        self.trainData = torch.from_numpy(trainData).type(torch.float32)

    def __getitem__(self, index):
        return self.trainData[index, :4], self.trainData[index, 4]
        # return self.trainData[index,:3],self.trainData[index,3]

    # def __getitem__(self, index):
    #     return torch.tensor([self.pts[index, :99]], dtype=torch.float32), torch.tensor([self.pts[index, 99]],
    #                                                                                    dtype=torch.float32)

    def __len__(self):
        return self.trainData.shape[0]


class MeshDataset_define(Dataset):
    def __init__(self, file_path):
        self.train_data = torch.as_tensor(np.load(file_path), dtype=torch.float)
    def __len__(self):
        return self.train_data.shape[0]
    def __getitem__(self, index):
        return self.train_data[index, :3], self.train_data[index, 3]