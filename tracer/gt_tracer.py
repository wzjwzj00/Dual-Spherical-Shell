import torch
import numpy as np
import torch.nn.functional as F

import readTest
from utils.lib.utils import PerfTimer
from render_gt_utils.diffutils import gradient
from render_gt_utils.geoutils import sample_unif_sphere
from utils.lib.tracer.RenderBuffer import RenderBuffer
from utils.lib.tracer.BaseTracer import BaseTracer
from sol_nglod import aabb
from time import time

# import cv2 as cv

device = torch.device("cuda")


def write_surfpoint_normals(fp_p, fp_n, fp_h, p, n, h):
    p = p.cpu().numpy()
    n = n.cpu().numpy()
    h = h.cpu().numpy()
    np.savetxt(fp_p, p, fmt='%.6f')
    np.savetxt(fp_n, n, fmt='%.6f')
    np.savetxt(fp_h, h, fmt='%d')


# (net,x[cond]）
def split_net(model, x):
    if x.shape[0] == 0:
        return torch.zeros((0, 1)).cuda()
    else:
        x1 = torch.split(x, 2048, dim=0)
        d = model(x1[0])
        for i in range(len(x1) - 1):
            d = torch.cat((d, model(x1[i + 1])), dim=0)
        return d


class SphereTracer(BaseTracer):

    def forward(self, net, sampler,ray_o, ray_d):
        """Native implementation of sphere tracing."""
        # Distanace from ray origin
        t = torch.zeros(ray_o.shape[0], 1, device=ray_o.device)

        cond = torch.ones_like(t).bool()[:, 0]

        '''t = torch.zeros(ray_o.shape[0], 1, device=ray_o.device)

        # Position in model space
        x = torch.addcmul(ray_o, ray_d, t)

        cond = torch.ones_like(t).bool()[:,0]
        x, t, cond = aabb(ray_o, ray_d)'''
        # Position in model space
        # x = torch.addcmul(ray_o, ray_d, t)
        # when eval, annotation

        # x是hit后的那个坐标，t是原点到hit点的距离（大概？），cond是是否hit了，一个布尔矩阵
        # 这个地方cond到底在撞什么呢，在撞-1～1的正方体表面呢，先把-1～1的正方体空间画出来了
        # print('ray_o', ray_o)
        # print('ray_d', ray_d)
        # x是当前hit的那个点，t是hit的距离
        x, t, cond = aabb(ray_o, ray_d)

        # PRINT

        normal = torch.zeros_like(x)
        # This function is in fact differentiable, but we treat it as if it's not, because
        # it evaluates a very long chain of recursive neural networks (essentially a NN with depth of
        # ~1600 layers or so). This is not sustainable in terms of memory use, so we return the final hit
        # locations, where additional quantities (normal, depth, segmentation) can be determined. The
        # gradients will propagate only to these locations.
        d = torch.zeros_like(t)
        time0 = time()
        # sphere_sampler = readTest.readSampler(
        #     mesh_file='/home/wzj/PycharmProjects/sphere_resconstruct/tmp/441708.obj'
        # )
        sphere_sampler = sampler
        with torch.no_grad():
            # 返回sdf值
            x_np = x.cpu().numpy()
            cond_np = cond.cpu().numpy()

            # mesh = geo.Mesh(mesh_path,doNormalize=False)
            # sdf = geo.SDF(mesh)
            S = sphere_sampler.sdf.query(x_np[cond_np])
            S = torch.tensor(S).to(torch.float32) # instant
            d[cond] = S.to("cuda")
            # d[cond] = split_net(net, x[cond]) * self.step_size

            d[~cond] = 20
            dprev = d.clone()

            # If cond is TRUE, then the corresponding ray has not hit yet.
            # OR, the corresponding ray has exit the clipping plane.
            # cond = torch.ones_like(d).bool()[:,0]

            # If miss is TRUE, then the corresponding ray has missed entirely.
            hit = torch.zeros_like(cond)
            # num_step = 256
            counter = 0
            for i in range(self.num_steps):
                counter += 1
                # t往前走d个值
                t += d
                x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)

                hit = torch.where(cond, torch.abs(d)[..., 0] < self.min_dis, hit)
                hit |= torch.where(cond, torch.abs(d + dprev)[..., 0] * 0.5 < (self.min_dis * 5), hit)
                cond = cond & ~(torch.abs(x) > 1.0).any(dim=-1)

                cond = torch.where(cond, (t < self.camera_clamp[1])[..., 0], cond)
                cond &= ~hit
                # cond1 = cond & ~(torch.abs(x) < 1.0).all(dim=-1)

                if not cond.any():
                    print('counter', counter)
                    break

                # x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)
                # dprev = torch.where(cond.view(cond.shape[0], 1), d, dprev)

                # _, d1, _ = aabb(x[cond1], -ray_d[cond1])
                # t[cond1]-=d1
                # x[cond1] = torch.addcmul(ray_o[cond1], ray_d[cond1], t[cond1])

                # d[cond] = split_net(net, x[cond]) * self.step_size

                x_np = x.cpu().numpy()
                cond_np = cond.cpu().numpy()

                # mesh = geo.Mesh(mesh_path,doNormalize=False)
                # sdf = geo.SDF(mesh)
                S = sphere_sampler.sdf.query(x_np[cond_np])
                S = torch.tensor(S).to(torch.float32) # instant
                d[cond] = S.to("cuda")

        print('counter', counter)
        # AABB cull
        hit = hit & ~(torch.abs(x) > 1.0).any(dim=-1)

        # np.savetxt('./renderpoints.txt', x1, fmt='%.6f')

        # The function will return
        #  x: the final model-space coordinate of the render
        #  t: the final distance from origin
        #  d: the final distance value from
        #  miss: a vector containing bools of whether each ray was a hit or miss
        # _normal = F.normalize(gradient(x[hit], net, method='finitediff'), p=2, dim=-1, eps=1e-5)
        sphere_index = torch.zeros(0)
        time1 = time()
        grad = gradient(x[hit], net,sampler, sphere_index,method=self.grad_method)
        grad = grad.squeeze(1)
        _normal = F.normalize(grad, p=2, dim=-1, eps=1e-5)
        normal[hit] = _normal.to(torch.float32)
        time2 = time()
        print('trace time: {}, normal time: {}'.format(time1 - time0, time2 - time1))
        # fpp = './allpoints.txt'
        # fpn = './normals.txt'
        # fph = './hit.txt'
        # write_surfpoint_normals(fpp,fpn,fph,x,normal,hit)

        return RenderBuffer(x=x, depth=t, hit=hit, normal=normal)

    def get_min(self, net, ray_o, ray_d):

        timer = PerfTimer(activate=False)
        nettimer = PerfTimer(activate=False)

        # Distance from ray origin
        t = torch.zeros(ray_o.shape[0], 1, device=ray_o.device)

        # Position in model space
        x = torch.addcmul(ray_o, ray_d, t)

        x, t, hit = aabb(ray_o, ray_d);

        normal = torch.zeros_like(x)

        with torch.no_grad():
            d = net(x)
            dprev = d.clone()
            mind = d.clone()
            minx = x.clone()

            # If cond is TRUE, then the corresponding ray has not hit yet.
            # OR, the corresponding ray has exit the clipping plane.
            cond = torch.ones_like(d).bool()[:, 0]

            # If miss is TRUE, then the corresponding ray has missed entirely.
            hit = torch.zeros_like(d).byte()

            for i in range(self.num_steps):
                timer.check("start")

                hit = (torch.abs(t) < self.camera_clamp[1])[:, 0]

                # 1. not hit surface
                cond = (torch.abs(d) > self.min_dis)[:, 0]

                # 2. not oscillating
                cond = cond & (torch.abs((d + dprev) / 2.0) > self.min_dis * 3)[:, 0]

                # 3. not a hit
                cond = cond & hit

                # cond = cond & ~hit

                # If the sum is 0, that means that all rays have hit, or missed.
                if not cond.any():
                    break

                # Advance the x, by updating with a new t
                x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)

                new_mins = (d < mind)[..., 0]
                mind[new_mins] = d[new_mins]
                minx[new_mins] = x[new_mins]

                # Store the previous distance
                dprev = torch.where(cond.unsqueeze(1), d, dprev)

                nettimer.check("nstart")
                # Update the distance to surface at x
                d[cond] = net(x[cond]) * self.step_size

                nettimer.check("nend")

                # Update the distance from origin
                t = torch.where(cond.view(cond.shape[0], 1), t + d, t)
                timer.check("end")

        # AABB cull

        hit = hit & ~(torch.abs(x) > 1.0).any(dim=-1)
        # hit = torch.ones_like(d).byte()[...,0]

        # The function will return
        #  x: the final model-space coordinate of the render
        #  t: the final distance from origin
        #  d: the final distance value from
        #  miss: a vector containing bools of whether each ray was a hit or miss
        # _normal = F.normalize(gradient(x[hit], net, method='finitediff'), p=2, dim=-1, eps=1e-5)
        _normal = gradient(x[hit], net, method=self.grad_method)
        normal[hit] = _normal

        return RenderBuffer(x=x, depth=t, hit=hit, normal=normal, minx=minx)

    def sample_surface(self, n, net):

        # Sample surface using random tracing (resample until num_samples is reached)

        timer = PerfTimer(activate=True)

        with torch.no_grad():
            i = 0
            while i < 1000:
                ray_o = torch.rand((n, 3), device=self.device) * 2.0 - 1.0
                # this really should just return a torch array in the first place
                ray_d = torch.from_numpy(sample_unif_sphere(n)).float().to(self.device)
                rb = self.forward(net, ray_o, ray_d)

                # d = torch.abs(net(rb.x)[..., 0])
                # idx = torch.where(d < 0.0003)
                # pts_pr = rb.x[idx] if i == 0 else torch.cat([pts_pr, rb.x[idx]], dim=0)

                pts_pr = rb.x[rb.hit] if i == 0 else torch.cat([pts_pr, rb.x[rb.hit]], dim=0)
                if pts_pr.shape[0] >= n:
                    break
                i += 1
                if i == 50:
                    print('Taking an unusually long time to sample desired # of points.')

        return pts_pr
