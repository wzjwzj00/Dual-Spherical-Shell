import time

import numpy as np
import torch
import h5py
import math
import sys
import os
import kaolin.render.spc as spc_render
from kaolin.ops.mesh import sample_points
from utils.lib.loadObjNglod import load_obj

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class NewSafeDis():
    def __init__(self, InnerSphere_path, MeshPath, SampleNum):
        self.ins_path = InnerSphere_path
        self.meshpath = MeshPath
        self.samplenum = SampleNum
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def load_my_obj(self):
        self.V, self.F = load_obj(self.meshpath)

    def my_sample_points(self):
        self.V = self.V.unsqueeze(0)
        self.sam_points, _ = sample_points(self.V, self.F, self.samplenum)
        self.sam_points = self.sam_points.to(self.device)
        self.sam_points = self.sam_points.squeeze()

    def get_inner_spheres(self):
        f = h5py.File(self.ins_path, 'r')
        f.keys()
        self.spheres = torch.as_tensor(f['data'][:, :4], dtype=torch.float, device=self.device)
        self.spheres[:, 3] = abs(self.spheres[:, 3])
        self.spheres_num = len(self.spheres)
        f.close()

    def distance(self, point, center, rad):  # 计算点到球心的距离
        return torch.linalg.norm(point - center) - rad

    def find_nearest_balls(self):
        a = spc_render.near_sphere(self.spheres, self.sam_points)
        self.point_nearest_sphere_list = a[0].long().squeeze()
        self.point_nearest_sphere_dist = a[1].squeeze()

    def max_dis(self):
        self.max_distance = torch.zeros_like(self.spheres[:, 0])
        for i in range(self.samplenum):
            if(i%300000==0):
              print(i)
            if self.max_distance[self.point_nearest_sphere_list[i]] < self.point_nearest_sphere_dist[i]:
                self.max_distance[self.point_nearest_sphere_list[i]] = self.point_nearest_sphere_dist[i]

    def max_dis_new(self):
        self.point_cond = torch.ones_like(self.sam_points[:, 0], dtype=torch.bool)
        self.max_distance = torch.zeros_like(self.spheres[:, 0])
        self.max_cond = torch.ones_like(self.spheres[:, 0], dtype=torch.bool)
        # 第一次选出一个
        a = spc_render.near_sphere_new(self.spheres, self.sam_points, self.point_cond)
        self.point_nearest_sphere_list = a[0].long().squeeze()
        self.point_nearest_sphere_dist = a[1].squeeze()

        for i in range(self.samplenum):
            if (i % 10000 == 0):
                print(i)
            if self.max_distance[self.point_nearest_sphere_list[i]] < self.point_nearest_sphere_dist[i]:
                self.max_distance[self.point_nearest_sphere_list[i]] = self.point_nearest_sphere_dist[i]

        for i in range(self.spheres.shape[0]):
            current_min_index = -1
            current_min_dis = 99  # 大小于号修改
            for z in range(self.spheres.shape[0]):
                if self.max_cond[z] & (self.max_distance[z] > current_min_dis):  # 大小于号修改
                    current_min_index = z  # current_max_index = z
                    current_min_dis = self.max_distance[z]
            self.max_cond[current_min_index] = 0
            current_single_safe_sphere = torch.cat((self.spheres[current_min_index, 0:3], torch.unsqueeze(
                current_min_dis + self.spheres[current_min_index, 3], dim=0)))
            current_single_safe_sphere = torch.unsqueeze(current_single_safe_sphere, dim=0)
            is_in_list = spc_render.point_is_in_sphere(current_single_safe_sphere, self.sam_points)
            self.point_cond[torch.squeeze(is_in_list == 1)] = 0

            # 再开始选剩下的
            a = spc_render.near_sphere_new(self.spheres, self.sam_points, self.point_cond)
            self.point_nearest_sphere_list = a[0].long().squeeze()
            self.point_nearest_sphere_dist = a[1].squeeze()

            for j in range(self.samplenum):
                print(i, j)
                if (self.point_cond[j] == True) & (self.max_cond[self.point_nearest_sphere_list[j]] == True) & (
                        self.max_distance[self.point_nearest_sphere_list[j]] < self.point_nearest_sphere_dist[
                    j]):  # 如果点没被排除、球没被选完、距离较小
                    self.max_distance[self.point_nearest_sphere_list[j]] = self.point_nearest_sphere_dist[j]

    def get_new_safe_spheres(self):
        safe_r = torch.add(self.spheres[:, 3], self.max_distance)
        print(torch.mean(safe_r))
        safe_r = safe_r.unsqueeze(1)
        safe_r = safe_r + 0.0001
        self.safe_spheres = torch.cat((self.spheres[:, 0:3], safe_r), dim=1)
        # flag = 0
        # for i in range(self.samplenum):
        #     print(i)
        #     for j in range(self.safe_spheres.shape[0]):
        #         if(((self.sam_points[i,0]-self.safe_spheres[j,0])**2+(self.sam_points[i,1]-self.safe_spheres[j,1])**2+(self.sam_points[i,2]-self.safe_spheres[j,2])**2)<self.safe_spheres[j,3]**2):
        #             flag += 1
        #             break
        #         else:
        #             continue
        b = spc_render.point_is_in_sphere(self.safe_spheres, self.sam_points)
        zero_idx = (torch.nonzero(b == 0).T)[0, :]
        a = 0

    def save_h5_spheres(self, path):
        self.safe_spheres_numpy = self.safe_spheres.cpu().numpy()
        with h5py.File(path, 'w') as f:
            f["data"] = self.safe_spheres_numpy

    def save_sam_points(self, path):
        self.sam_points_numpy = self.sam_points.cpu().numpy()
        np.savetxt(path, self.sam_points_numpy)


if __name__ == "__main__":
    # nsd = NewSafeDis('/home/wzj/project/Sphere/myobj256inout441708_instant_edition_normalized/InnerSpheres_64.h5',
    #                  '/home/wzj/project/Sphere/myobj/256inout441708_instant_edition_normalized.obj',
    #                  1000000)
    # nsd.load_my_obj()
    # nsd.my_sample_points()
    # # nsd.save_sam_points("/home/wzj/project/dcc_copy/new_safe_spheres/sam_point.txt")
    # nsd.get_inner_spheres()
    # time1 = time.time()
    # nsd.find_nearest_balls()
    # nsd.max_dis()
    # # nsd.max_dis_new()
    # print(time.time()-time1)
    # nsd.get_new_safe_spheres()
    # nsd.save_h5_spheres("/home/wzj/project/dcc_copy/new_safe_spheres/" + "441708_ins" + "_" + "64_+0_003" + '.h5')
    #
    inner_sphere_path = '/home/wzj/project/Sphere/myobj512inner'
    normalized_mesh_dir = '/home/wzj/project/Sphere/myobj'
    output_dir = '/home/wzj/project/Sphere/myobj512safe/'
    files = os.listdir(normalized_mesh_dir)
    sphere_num = 512
    for m in files:

        mname = os.path.splitext(m)[0]


        mesh_path = os.path.join(normalized_mesh_dir, m)
        inner_sphere = os.path.join(inner_sphere_path, mname + ".h5")
        nsd = NewSafeDis(inner_sphere, mesh_path, 1200000)
        nsd.load_my_obj()
        nsd.my_sample_points()
        nsd.get_inner_spheres()
        time1 = time.time()
        nsd.find_nearest_balls()
        nsd.max_dis()
        nsd.get_new_safe_spheres()
        nsd.save_h5_spheres(output_dir + mname + ".h5")
        print(mname+"saved")
