# The MIT License (MIT)
#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import torch
import numpy as np
import torch.nn.functional as F
from utils.lib.utils import PerfTimer
from utils.lib.diffutils import gradient
from utils.lib.geoutils import sample_unif_sphere
from utils.lib.tracer.RenderBuffer import RenderBuffer
from utils.lib.tracer.BaseTracer import BaseTracer
from sol_nglod import aabb
from time import time
#import cv2 as cv

device = torch.device("cuda")
def write_surfpoint_normals(fp_p,fp_n,fp_h,p,n,h):
    p = p.cpu().numpy()
    n = n.cpu().numpy()
    h = h.cpu().numpy()
    np.savetxt(fp_p,p,fmt='%.6f')
    np.savetxt(fp_n,n, fmt='%.6f')
    np.savetxt(fp_h,h,fmt='%d')

#(net,x[cond]）
def split_net(model,x):
    if x.shape[0]==0:
        return torch.zeros((0,1)).cuda()
    else:
        x1 = torch.split(x, 2048, dim=0)
        d = model(x1[0])
        for i in range(len(x1) - 1):
            d = torch.cat((d, model(x1[i + 1])), dim=0)
        return d

class SphereTracer(BaseTracer):

    def forward(self, net, ray_o, ray_d):
        """Native implementation of sphere tracing."""
        # Distanace from ray origin
        t = torch.zeros(ray_o.shape[0], 1, device=ray_o.device)

        cond = torch.ones_like(t).bool()[:,0]

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
        '''indexes = torch.where(cond==1)[0].cpu().detach().numpy()
        print(indexes)
        print(x[indexes])
        print(t[indexes])
        width, height = 512_64_60, 512_64_60
        mask = np.zeros((width*height), np.uint8)
        mask[indexes]=255
        mask = mask.reshape(width, height)
        cv.namedWindow("bitwise_not", cv.WINDOW_GUI_EXPANDED)
        cv.imshow("bitwise_not", mask)
        cv.waitKey(5000)'''
        # PRINT
        
        normal = torch.zeros_like(x)
        # This function is in fact differentiable, but we treat it as if it's not, because
        # it evaluates a very long chain of recursive neural networks (essentially a NN with depth of
        # ~1600 layers or so). This is not sustainable in terms of memory use, so we return the final hit
        # locations, where additional quantities (normal, depth, segmentation) can be determined. The
        # gradients will propagate only to these locations.
        d = torch.zeros_like(t)
        time0 = time()
        with torch.no_grad():
            # 返回sdf值
            d[cond] = split_net(net,x[cond])*self.step_size
            # x_ = x[cond]
            # t_ = t[cond]
            # d_  = d[cond]
            # ray_o = ray_o[cond]
            # ray_d = ray_d[cond]
            # q = torch.addcmul(ray_o, ray_d, t_+d_)
            # q1 = q.cpu().numpy()
            # 没有碰到的设置为大值
            d[~cond] = 20
            dprev = d.clone()

            # If cond is TRUE, then the corresponding ray has not hit yet.
            # OR, the corresponding ray has exit the clipping plane.
            #cond = torch.ones_like(d).bool()[:,0]

            # If miss is TRUE, then the corresponding ray has missed entirely.
            hit = torch.zeros_like(cond)
            # num_step = 256
            counter = 0
            for i in range(self.num_steps):
                counter+=1
                # t往前走d个值
                t += d
                x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)

                hit = torch.where(cond, torch.abs(d)[..., 0] < self.min_dis, hit)
                hit |= torch.where(cond, torch.abs(d + dprev)[..., 0] * 0.5 < (self.min_dis * 5), hit)
                cond = cond & ~(torch.abs(x) > 1.0).any(dim=-1)

                cond = torch.where(cond, (t < self.camera_clamp[1])[..., 0], cond)
                cond &= ~hit
                #cond1 = cond & ~(torch.abs(x) < 1.0).all(dim=-1)

                if not cond.any():
                    print('counter', counter)
                    break

                # x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)
                # dprev = torch.where(cond.view(cond.shape[0], 1), d, dprev)

                # _, d1, _ = aabb(x[cond1], -ray_d[cond1])
                # t[cond1]-=d1
                # x[cond1] = torch.addcmul(ray_o[cond1], ray_d[cond1], t[cond1])
                d[cond] = split_net(net,x[cond])* self.step_size


        print('counter', counter)
        # AABB cull
        hit = hit & ~(torch.abs(x) > 1.0).any(dim=-1)

        # np.savetxt('./renderpoints.txt', x1, fmt='%.6f')
        
        # The function will return 
        #  x: the final model-space coordinate of the render
        #  t: the final distance from origin
        #  d: the final distance value from
        #  miss: a vector containing bools of whether each ray was a hit or miss
        #_normal = F.normalize(gradient(x[hit], net, method='finitediff'), p=2, dim=-1, eps=1e-5)

        time1 = time()
        grad = gradient(x[hit], net, method=self.grad_method)
        _normal = F.normalize(grad, p=2, dim=-1, eps=1e-5)
        normal[hit] = _normal
        time2 = time()
        print('trace time: {}, normal time: {}'.format(time1-time0,time2-time1))
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
            cond = torch.ones_like(d).bool()[:,0]

            # If miss is TRUE, then the corresponding ray has missed entirely.
            hit = torch.zeros_like(d).byte()
            
            for i in range(self.num_steps):
                timer.check("start")

                hit = (torch.abs(t) < self.camera_clamp[1])[:,0]
                
                # 1. not hit surface
                cond = (torch.abs(d) > self.min_dis)[:,0] 

                # 2. not oscillating
                cond = cond & (torch.abs((d + dprev) / 2.0) > self.min_dis * 3)[:,0]
                
                # 3. not a hit
                cond = cond & hit
        
                
                #cond = cond & ~hit
                
                # If the sum is 0, that means that all rays have hit, or missed.
                if not cond.any():
                    break

                # Advance the x, by updating with a new t
                x = torch.where(cond.view(cond.shape[0], 1), torch.addcmul(ray_o, ray_d, t), x)
                
                new_mins = (d<mind)[...,0]
                mind[new_mins] = d[new_mins]
                minx[new_mins] = x[new_mins]
            
                # Store the previous distance
                dprev = torch.where(cond.unsqueeze(1), d, dprev)

                nettimer.check("nstart")
                # Update the distance to surface at x
                d[cond] = net(x[cond]) * self.step_size

                nettimer.check("nend")
                
                # Update the distance from origin 
                t = torch.where(cond.view(cond.shape[0], 1), t+d, t)
                timer.check("end")

        # AABB cull 

        hit = hit & ~(torch.abs(x) > 1.0).any(dim=-1)
        #hit = torch.ones_like(d).byte()[...,0]
        
        # The function will return 
        #  x: the final model-space coordinate of the render
        #  t: the final distance from origin
        #  d: the final distance value from
        #  miss: a vector containing bools of whether each ray was a hit or miss
        #_normal = F.normalize(gradient(x[hit], net, method='finitediff'), p=2, dim=-1, eps=1e-5)
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

                #d = torch.abs(net(rb.x)[..., 0])
                #idx = torch.where(d < 0.0003)
                #pts_pr = rb.x[idx] if i == 0 else torch.cat([pts_pr, rb.x[idx]], dim=0)
                
                pts_pr = rb.x[rb.hit] if i == 0 else torch.cat([pts_pr, rb.x[rb.hit]], dim=0)
                if pts_pr.shape[0] >= n:
                    break
                i += 1
                if i == 50:
                    print('Taking an unusually long time to sample desired # of points.')
        
        return pts_pr

