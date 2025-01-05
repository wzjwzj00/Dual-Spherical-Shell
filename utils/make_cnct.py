import numpy as np
import os
from torch_geometric.nn import knn
import torch


class PreProsessVoxelSphere():
    def __init__(self, key):
        self.key = key
        # self.device = device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run(self, size, sphere_pos, sphere_r, k):
        self.size = size
        self.sphere_pos = sphere_pos
        self.sphere_r = sphere_r
        self.k = k
        self.voxel = self.make_voxel()
        nearst_dis, nearst_idx = self.query_k_point(self.k, self.voxel, self.sphere_pos, self.sphere_r)
        if self.key == 'general':
            return nearst_dis, nearst_idx
        else:
            raise NotImplemented

    def make_voxel(self):
        x_num = torch.linspace(-1, 1, self.size, device=self.device)
        y_num = torch.linspace(-1, 1, self.size, device=self.device)
        z_num = torch.linspace(-1, 1, self.size, device=self.device)
        X, Y, Z = torch.meshgrid(x_num, y_num, z_num)
        # grid = np.concatenate((X.reshape(128,128,128,1), Y.reshape(128,128,128,1), Z.reshape(128,128,128)), axis=-1)
        # save_voxel_sdf = np.concatenate((Z.reshape(-1,1), Y.reshape(-1,1), X.reshape(-1,1), data[:, 3].reshape(-1, 1)), axis=-1)
        voxel = torch.cat((Z.reshape(-1,1), X.reshape(-1,1), Y.reshape(-1,1)), dim=-1)
        # return torch.as_tensor(voxel.reshape(-1, 3))
        return voxel.reshape(-1, 3)


    def defined_distance(self, src, dst, sphere_r):
    
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [N, C]
            dst: target points, [M, C]
            sphere_r: sphere radius, [M]
        Output:
            dist: defined distance, [N, M], ||src - dst|| - sphere_r
        """
    
        N, _ = src.shape
        M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(1, 0))
        dist += torch.sum(src ** 2, -1).view(N, 1)
        dist += torch.sum(dst ** 2, -1).view(1, M)
        sphere_r = sphere_r.view(1, M).repeat([N, 1])
        # print('sphere', sphere_r.shape)
        dist = torch.sqrt(dist) - sphere_r
        # print('dist', dist)
        # print(dist.shape)
        return dist




    def query_k_point(self, k, xyz, new_xyz, sphere_r):
    
        """
        Input:
            k: max sample number in local region
            xyz: all voxel points, [N, 3]
            new_xyz: query sphere points, [S, 3]
            sphere_r: sphere radius
        Return:
            nearst_idx: top k nearst points index, [S, k]
            inside_sphere_idx: top k nearst points index inside the sphere, [S, k]
        #TODO what if none of the sphere is inside, what would the index be?
        """
    
        # device = xyz.device
        N, C = xyz.shape
        S, _ = new_xyz.shape
    
        group_idx = torch.arange(N, dtype=torch.long).view(N, 1).repeat([1, 8])
        # print(group_idx)
        dists = self.defined_distance(xyz, new_xyz, sphere_r)
        # group_idx[sqrdists > radius ** 2] = N
        
        nearst_dis, nearst_idx = torch.topk(dists, k=k, dim=-1, largest=False)
        # print(nearst_dis, nearst_idx)
        '''dists_inside_sphere = dists.clone()
        dists_inside_sphere[dists_inside_sphere>=0] = 0
        inside_sphere_idx = torch.topk(dists_inside_sphere, k=k, dim=-1, largest=False)[1]'''
        '''
        # vertify the index was right
        print('nearst_idx', nearst_idx)
        print(dists[group_idx, nearst_idx])
        print(nearst_idx.shape)
        print('inside_sphere_idx', inside_sphere_idx)
        print(dists_inside_sphere[group_idx, nearst_idx])
        print(inside_sphere_idx.shape)
        '''

        return nearst_dis, nearst_idx


def read_sphere_file(sphere_dir_path):
    sphere_dir_path = sphere_dir_path
    for sphere in os.listdir(sphere_dir_path):
        sphere_file = os.path.join(sphere_dir_path, sphere)
        sphere = open(sphere_file)
        sphere_lines = sphere.readlines()
        sphere_indexes = []
        sphere_r = []
        for i in range(1, len(sphere_lines)-1):
            tmp = sphere_lines[i].split(' ')
            tmp_sphere_index = []
            for i in range(3):
                tmp_sphere_index.append(int(tmp[i]))
            sphere_r.append(float(tmp[i]))
            sphere_indexes.append(tmp_sphere_index)
        sphere_indexes = np.array(sphere_indexes)
        sphere_r = np.array(sphere_r)
        # print(sphere_indexes)
        return torch.as_tensor((sphere_indexes-63.5)*(2/127)), torch.as_tensor((sphere_r-63.5)*(2/127))

if __name__ == "__main__":
    sphere_dir_path = '/home/rhodaliu/code/dcc/dcc_lyq/data/sphere/ni/abc_sphere'
    sphere_pos, sphere_r = read_sphere_file(sphere_dir_path)
    preprosess = PreProsessVoxelSphere('general')
    nearst_idx = preprosess.run(size=64, sphere_pos = sphere_pos, sphere_r = sphere_r, k=8)
