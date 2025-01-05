# Sphere Skip Tracer
import math
from time import time

import kaolin.render.spc as spc_render
from tqdm import tqdm

import readTest
# import cv2 as cv
from utils.lib.tracer.BaseTracer import BaseTracer
from utils.lib.tracer.BaseTracer_new import BaseTracerNew
from utils.lib.tracer.RenderBuffer import RenderBuffer
from utils.lib.diffutils import gradient
# import open3d.visualization.rendering as rendering
# import open3d.visualization.gui as gui
# import open3d as o3d
import numpy as np
import h5py
import torch.nn.functional as F
import torch
import sys
import os
import kaolin.render.spc as spc_render
from utils.lib.geoutils import sample_unif_sphere
import utils.geometry as geo
# import open3d as o3d
from kaolin.ops.mesh import sample_points
from utils.lib.loadObjNglod import load_obj
from utils.lib.utils import PerfTimer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))


class SSTracer(BaseTracerNew):
    def forward(self, net, ray_o, ray_d):
        # timer = PerfTimer()
        max_num = 25
        normal = torch.zeros_like(ray_o)
        device = ray_o.device
        N = ray_o.shape[0]
        cond = torch.ones(N, dtype=torch.bool, device=self.device)
        time22 = time()

        ray_list,cond,list_cond,list_d1,list_d2 = spc_render.test_raytrace_new(ray_o,ray_d,cond,net.sphere32,net.innersphere,max_num)
        acc_max_num = list_cond.view(N, max_num).sum(dim=1).max()  # 实际最大相交数量

        # 步骤1: 创建光线索引数组，每个索引重复 max_num 次
        # ray_indices_expanded = torch.arange(N).repeat_interleave(max_num).to("cuda")

        # 步骤2: 使用 list_cond 作为掩码筛选出相交的球体编号和对应的光线索引
        intersecting_spheres = ray_list[list_cond]
        # intersecting_rays = ray_indices_expanded[list_cond.squeeze(1)]

        # 步骤1: 扩展 ray_o，使其与 ray_list 的长度相同
        ray_o_expanded = ray_o.repeat_interleave(max_num, dim=0)

        # 步骤2: 使用 list_cond 作为掩码筛选出相交情况的光线起点
        intersecting_ray_o = ray_o_expanded[list_cond.squeeze(1)]
        ray_d_expanded = ray_d.repeat_interleave(max_num, dim=0)
        intersecting_ray_d = ray_d_expanded[list_cond.squeeze(1)]

        # 使用 list_cond 作为掩码筛选出相交情况的最近和最远距离值
        intersecting_d1 = list_d1[list_cond]
        intersecting_d2 = list_d2[list_cond]
        compressed_new_pos = intersecting_ray_o.clone()
        compressed_new_pos= torch.addcmul(compressed_new_pos,intersecting_ray_d,intersecting_d1.unsqueeze(1))



        hit_type = torch.zeros_like(intersecting_d1, dtype=torch.bool).squeeze()
        step_type = torch.zeros_like(intersecting_d1)
        step_type += intersecting_d1
        step_type = step_type.squeeze()
        type_cond = torch.ones_like(intersecting_d1, dtype=torch.bool)

        hit = torch.zeros_like(cond)
        # 步骤1: 重塑 list_cond 为 [N, max_num]

        list_cond_reshaped = list_cond.view(N, max_num)
        intersection_counts_per_ray = list_cond_reshaped.sum(dim=1)
        actual_intersection_counts = intersection_counts_per_ray[cond].to(torch.int32)


        prefix_sum = (torch.cumsum(actual_intersection_counts,dim=0)-actual_intersection_counts).to(torch.int32)
        time33 = time()
        print('pre time:{}'.format(time33 - time22))
        sdf_time = 0
        array_time = 0
        with torch.no_grad():

            _d_type = net.sdf1(compressed_new_pos[type_cond],intersecting_spheres[type_cond])
            type_d =_d_type.squeeze(1)

            d_prev = _d_type.clone().squeeze(1)

            type_cond[(type_d < -5 * self.min_dis).squeeze()] = False
            valid_ray_num = type_cond[type_cond].shape[0]
            # time1 = time()
            total_st_nums = 0
            for i in range(256):
                curr_ray_num = type_cond[type_cond].shape[0]
                total_st_nums +=curr_ray_num
                timekk = time()
                step_type[type_cond] +=type_d[type_cond]
                compressed_new_pos = torch.where(type_cond.view(type_d.shape[0],1),
                                                 torch.addcmul(intersecting_ray_o,intersecting_ray_d,step_type.unsqueeze(1)),compressed_new_pos)
                hit_type = torch.where(type_cond,torch.abs(type_d.squeeze())<self.min_dis,hit_type)
                hit_type |= torch.where(type_cond,torch.abs(type_d+d_prev)*0.5<self.min_dis*5,hit_type)
                type_cond = torch.where(type_cond,(step_type<intersecting_d2.squeeze()),type_cond)
                type_cond &= ~hit_type
                if not type_cond.any():
                    print('stop time' + str(i))
                    break
                compressed_new_pos = torch.where(type_cond.view(type_cond.shape[0],1),
                                                 torch.addcmul(intersecting_ray_o,intersecting_ray_d,step_type.unsqueeze(1)),compressed_new_pos)
                d_prev = torch.where(type_cond,type_d,d_prev)
                time_array = time()
                # print("数组并行:{}".format(time() - timekk))
                array_time+=(time_array-timekk)
                sdf_time1 = time()
                _d_type = net.sdf1(compressed_new_pos[type_cond],intersecting_spheres[type_cond])
                sdf_time2 = time()
                sdf_time+=(sdf_time2-sdf_time1)
                # print(
                    # 'round {} ,sdf_time: {},alive num {}'.format(i, sdf_time2 - sdf_time1, type_cond[type_cond].shape[0]))
                type_d[type_cond] = _d_type.squeeze(1) *self.step_size
            # time2 = time()
            # print("loop time:",time2-time1)

            timeaft = time()
            type_final_cond,nice_t,type_sphere_tmp = spc_render.find_nice_ball(intersecting_d1,step_type,hit_type,actual_intersection_counts,prefix_sum,self.min_dis*5)


            pos_type_final = torch.where(type_final_cond,torch.addcmul(ray_o[cond],ray_d[cond],nice_t),ray_o[cond])
            hit[cond] = type_final_cond.squeeze(1)
            final_pos = torch.zeros_like(ray_o)
            final_pos[cond] = pos_type_final

            intersecting_ray_indices = cond.nonzero().squeeze()
            final_sphere_index = torch.zeros_like(cond, dtype=torch.int, device=self.device)
            final_sphere_index[intersecting_ray_indices] = intersecting_spheres[(prefix_sum+type_sphere_tmp.squeeze().long())]
            final_dis = torch.zeros_like(cond, device=self.device,dtype=torch.float)
            final_dis[cond] = nice_t.squeeze()
            print(time()-timeaft)

        # print("loop_time:{}".format(time_sum2 - time_sum1))
        print('sdf time', sdf_time)
        print('array_time',array_time)
        print("total steps per hit={}/{} = {}".format(total_st_nums, valid_ray_num, (total_st_nums / valid_ray_num)))
        print("all_time:{}".format(time()-time22))
        # timer.check("alltime")
        time_grad = time()
        grad = gradient(final_pos[hit], net, final_sphere_index[hit], method=self.grad_method)  # finitediff

        normal[hit] = F.normalize(grad, p=2, dim=-1, eps=1e-5)
        print("time_grad:",time()-time_grad)
        # np_points = final_pos[hit].cpu().numpy()
        # np.savetxt("final_hit.txt", np_points)
        return RenderBuffer(x=final_pos, depth=final_dis, hit=hit, normal=normal)

    def sample_surface(self, n, net, sampler):
        with torch.no_grad():
            i = 0
            while i < 1000:
                ray_o = torch.rand((n, 3), device=self.device) * 2.0 - 1.0
                # ray_o = self.sample_uni_sphere(n).to("cuda")
                ray_d = torch.from_numpy(sample_unif_sphere(n)).float().to(self.device)
                b = spc_render.point_is_in_sphere(net.sphere32, ray_o)
                ray_o = ray_o[b.squeeze() == 0]
                ray_d = ray_d[b.squeeze() == 0]
                ray_o_np = ray_o.cpu().numpy()
                # sphere_sampler = readTest.readSampler(
                #     mesh_file=mesh_path
                # )
                # mesh = geo.Mesh(mesh_path,doNormalize=False)
                # sdf = geo.SDF(mesh)
                S = sampler.sdf.query(ray_o_np)
                S = torch.tensor(S)  # instant
                ray_o = ray_o[S.squeeze() > 0.001]
                ray_d = ray_d[S.squeeze() > 0.001]
                ray_o_np = ray_o.cpu().numpy()
                np.savetxt("ray_o_np.txt",ray_o_np)
                rb = self.forward(net, ray_o, ray_d)
                pts_pr = rb.x[rb.hit] if i == 0 else torch.cat([pts_pr, rb.x[rb.hit]], dim=0)
                pts_pr_np = pts_pr.cpu().numpy()
                # np.savetxt("hit_aaa.txt",pts_pr_np)
                if pts_pr.shape[0] >= n:
                    break
                i += 1
                print("current points:{}".format(pts_pr.shape[0]))
                if i == 50:
                    print('Taking an unusually long time to sample desired # of points.')

        return pts_pr

    def sample_uni_sphere(self, num_points):
        theta = 2 * math.pi * torch.rand(num_points)  # 方位角 [0, 2π)
        phi = torch.acos(2 * torch.rand(num_points) - 1)  # 极角 [0, π]

        # 转换为笛卡尔坐标
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        return torch.stack([x, y, z], dim=1)
