# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from kaolin import _C

def unbatched_raytrace(octree, point_hierarchy, pyramid, exsum, origin, direction, level):
    r"""Apply ray tracing over an unbatched SPC structure.

    The SPC model will be always normalized between -1 and 1 for each axis.

    Args:
        octree (torch.ByteTensor): the octree structure,
                                   of shape :math:`(\text{num_bytes})`.
        point_hierarchy (torch.ShortTensor): the point hierarchy associated to the octree,
                                             of shape :math:`(\text{num_points}, 3)`.
        pyramid (torch.IntTensor): the pyramid associated to the octree,
                                   of shape :math:`(2, \text{max_level} + 2)`.
        exsum (torch.IntTensor): the prefix sum associated to the octree.
                                 of shape :math:`(\text{num_bytes} + \text{batch_size})`.
        origin (torch.FloatTensor): the origins of the rays,
                                    of shape :math:`(\text{num_rays}, 3)`.
        direction (torch.FloatTensor): the directions of the rays,
                                       of shape :math:`(\text{num_rays}, 3)`.
        level (int): level to use from the octree.

    Returns:
        (torch.IntTensor): Nuggets of intersections sorted by depth,
                           of shape :math:`(\text{num_intersection}, 2)` representing pairs
                           :math:`(\text{index_to_ray}, \text{index_to_points})`.
    """
    return _C.render.spc.raytrace(
        octree.contiguous(),
        point_hierarchy.contiguous(),
        pyramid.contiguous(),
        exsum.contiguous(),
        origin.contiguous(),
        direction.contiguous(),
        level)

def mark_first_hit(nuggets):
    r"""Mark the first hit in the nuggets.

    The nuggets are a packed tensor containing correspondences from ray index to point index, sorted
    within each ray pack by depth. This will mark True for each first hit (by depth) for a pack of
    nuggets.

    Returns:
        first_hits (torch.BoolTensor): the boolean mask marking the first hit by depth.
    """
    # TODO(cfujitsang): directly output boolean
    return _C.render.spc.mark_first_hit(nuggets.contiguous()).bool()

def unbatched_ray_aabb(nuggets, point_hierarchy, ray_o, ray_d, level,
                       info=None, info_idxes=None, mask=None):
    r"""Ray AABB intersection with points.

    Raytrace will already get correspondences, but this will additionally compute distances.

    .. note::

      This function is likely to be folded into raytrace in a future version.

    Args:
        nuggets (torch.IntTensor): the ray-point correspondences,
                                   of shape :math:`(\text{num_nuggets}, 2)`.
        point_hierarchy (torch.ShortTensor): the point_hierarchy associated to the octree,
                                             of shape :math:`(\text{num_points}, 3)`.
        ray_o (torch.FloatTensor): ray origins, of shape :math:`(\text{num_rays}, 3)`.
        ray_d (torch.FloatTensor): ray directions, of shape :math:`(\text{num_rays}, 3)`.
        level (int): level of the SPC to trace.
        info (torch.BoolTensor): First hits. Default: Computed internally.
        info_idxes (torch.IntTensor): Packed indices of first hits.
                                      Default: Computed internally.
        mask (torch.BoolTensor): Mask to determine if the ray is still alive.
                                 Default: zeros mask.

    Returns:
        (torch.FloatTensor, torch.LongTensor, torch.BoolTensor):


            - Distance from ray origin to ray-aabb intersection,
              of shape :math:`(\text{num_rays}, 1)`.

            - Corresponding point index, of shape :math:`(\text{num_rays})`.

            - New mask. :math:`(\text{num_rays})`.
    """
    if info is None:
        info = mark_first_hit(nuggets.contiguous())
    if info_idxes is None:
        info_idxes = torch.nonzero(info, as_tuple=False).int()
    init = mask is None
    if mask is None:
        num_rays = ray_o.shape[0]
        mask = torch.zeros(num_rays, dtype=torch.bool, device=ray_o.device)
    return _C.render.spc.ray_aabb(nuggets.contiguous(), point_hierarchy.contiguous(),
                                  ray_o.contiguous(), ray_d.contiguous(), level,
                                  info.contiguous().int(), info_idxes.contiguous(),
                                  mask.contiguous(), init)
def unbatched_ray_sphere(nuggets,ray_o,ray_d,x , y, z,r,info=None, info_idxes=None , mask=None):
    if info is None:
        info = mark_first_hit(nuggets.contiguous())
    if info_idxes is None:
        info_idxes = torch.nonzero(info, as_tuple=False).int()
    init = mask is None
    if mask is None:
        num_rays = ray_o.shape[0]
        mask = torch.zeros(num_rays, dtype=torch.bool, device=ray_o.device)
    return _C.render.spc.ray_sphere(nuggets.contiguous(),ray_o.contiguous(), 
                                    ray_d.contiguous(),x.contiguous(),y.contiguous(),z.contiguous(),r.contiguous(),
                                    info.contiguous().int(), info_idxes.contiguous(),mask.contiguous(),init)
def compute_safe_dist(innerSpheres,points):
    return _C.render.spc.compute_safe_dis(innerSpheres.contiguous(),points.contiguous())
def safe_ray_sphere_step(ray_o,ray_d,cond,safe_spheres):
    return _C.render.spc.ray_safe_sphere(ray_o.contiguous(),ray_d.contiguous(),cond.contiguous(),safe_spheres.contiguous())
def near_sphere(innerSpheres,points):
    return _C.render.spc.compute_nearest_sphere(innerSpheres.contiguous(),points.contiguous())

def point_is_in_sphere(spheres,points):
    return _C.render.spc.point_is_in_sphere(spheres.contiguous(),points.contiguous())
def near_sphere_new(innerSpheres,points,point_cond):
    return _C.render.spc.compute_nearest_sphere_new(innerSpheres.contiguous(),points.contiguous(),point_cond.contiguous())
def point_its_max_sphere(safeSpheres,points):
    return _C.render.spc.point_its_max_sphere(safeSpheres.contiguous(),points.contiguous())
def get_clip(ray_o,ray_d,cond,safe_spheres,clip):
    return _C.render.spc.get_clip(ray_o.contiguous(),ray_d.contiguous(),cond.contiguous(),safe_spheres.contiguous(),clip.contiguous())
def sphere_raytrace(ray_o,ray_d,cond,safe_spheres,inner_spheres,max_num):
    return _C.render.spc.sphere_raytrace(ray_o.contiguous(),ray_d.contiguous(),cond.contiguous(),safe_spheres.contiguous(),inner_spheres.contiguous(),max_num)
def find_min_t(step_type1,cond_type1,max_num):
    return _C.render.spc.find_min_t(step_type1.contiguous(),cond_type1.contiguous(),max_num)
def test_raytrace(ray_o,ray_d,cond,safe_spheres,inner_spheres,max_num):
    return _C.render.spc.test_raytrace(ray_o.contiguous(),ray_d.contiguous(),cond.contiguous(),safe_spheres.contiguous(),inner_spheres.contiguous(),max_num)
def test_raytrace_new(ray_o,ray_d,cond,safe_spheres,inner_spheres,max_num):
    return _C.render.spc.test_raytrace_new(ray_o.contiguous(),ray_d.contiguous(),cond.contiguous(),safe_spheres.contiguous(),inner_spheres.contiguous(),max_num)
def find_min_t_new(step_type1,cond_type1,nonzero_cnts,prefix_sum):
    return _C.render.spc.find_min_t_new(step_type1.contiguous(),cond_type1.contiguous(),nonzero_cnts.contiguous(),prefix_sum.contiguous())
def find_nice_ball(compressed_d1,step_type1,cond_type1,nonzero_cnts,prefix_sum,epsilon):
    return _C.render.spc.find_nice_ball(compressed_d1.contiguous(),step_type1.contiguous(),cond_type1.contiguous(),nonzero_cnts.contiguous(),prefix_sum.contiguous(),epsilon)