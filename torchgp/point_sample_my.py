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
from .sample_near_surface import sample_near_surface
from .sample_surface import sample_surface
from .sample_uniform import sample_uniform
from .area_weighted_distribution import area_weighted_distribution
import kaolin.render.spc as spc_render
def point_sample(
    V : torch.Tensor, 
    F : torch.Tensor,
    spheres,
    techniques : list,
    importaceSampler,
    num_samples : int):
    """Sample points from a mesh.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        F (torch.Tensor): #F, 3 array of indices
        techniques (list[str]): list of techniques to sample with
        num_samples (int): points to sample per technique
    """


    if 'trace' in techniques or 'near' in techniques:
        # Precompute face distribution
        distrib = area_weighted_distribution(V, F)


    samples = []
    # 在这里 importance是NI的采接近表面的采样方法
    # near是nglod的采接近表面的采样方法
    for technique in techniques:
        if technique =='trace':
            samples.append(sample_surface(V, F, num_samples, distrib=distrib)[0])
        elif technique == 'near':
            samples.append(sample_near_surface(V, F, num_samples, distrib=distrib))
        elif technique == 'rand':
            samples.append(sample_uniform(num_samples).to(V.device))
        elif technique == 'importance':
            samples.append(torch.from_numpy(importaceSampler.sample(num_samples)))


    samples = torch.cat(samples, dim=0)
    samples = samples.to("cuda")
    spheres[:,3]  = abs(spheres[:,3])
    valid_idx = spc_render.point_is_in_sphere(spheres,samples)
    samples = samples[torch.squeeze(valid_idx)==1]# 保留符合条件的采样点，在安全球内的。


    # sam_cord = samples.view(-1,1,3).repeat(1,spheres.shape[0],1)
    # # oc = (sam_cord - spheres[:,:3])
    # oc_2 = torch.sum((sam_cord - spheres[:,:3])*(sam_cord - spheres[:,:3]),axis=-1)
    # minus = oc_2 - spheres[:,3]**2
    # in_sphere_bool = minus <= 0
    # pairs = torch.nonzero(in_sphere_bool==True)
    # # test = spc_render.point_its_max_sphere(spheres,samples)
    # new_samples = torch.empty((0,3),device="cuda")
    # sphere_list = torch.empty(0,device="cuda")
    # for i in range(pairs.shape[0]):
    #     print(i)
    #     new_samples  =  torch.cat([new_samples,torch.unsqueeze(samples[pairs[i,0]],dim=0)])
    #     sphere_list = torch.cat([sphere_list,torch.unsqueeze(pairs[i,1],dim=0)])
    # a = 0
    # sphere_list = torch.unsqueeze(sphere_list,dim=1)


    return samples
    # return new_samples .detach().cpu().numpy() ,sphere_list.detach().cpu().numpy()


