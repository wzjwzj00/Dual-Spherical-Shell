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
import math

import numpy as np
import torch
from tqdm import tqdm

from .sample_near_surface import sample_near_surface
from .sample_surface import sample_surface
from .sample_uniform import sample_uniform
from .area_weighted_distribution import area_weighted_distribution
import kaolin.render.spc as spc_render


def point_sample(
        V: torch.Tensor,
        F: torch.Tensor,
        spheres,
        inner_spheres,
        techniques: list,
        importaceSampler,
        num_samples: int):
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
    # spheres[:, 3] = spheres[:, 3] + 1.0 * math.sqrt(3) / (64 * 3.0)  # 扩大一圈以计算梯度
    # 首先克隆原始张量
    spheres_copy = spheres.clone()

    # 然后在克隆的张量上修改半径
    spheres_copy[:, 3] = spheres_copy[:, 3] + 1.0 * math.sqrt(3) / (64 * 3.0)
    # 在这里 importance是NI的采接近表面的采样方法
    # near是nglod的采接近表面的采样方法
    for technique in techniques:
        if technique == 'trace':
            samples.append(sample_surface(V, F, num_samples, distrib=distrib)[0])
        elif technique == 'near':
            samples.append(sample_near_surface(V, F, num_samples, distrib=distrib))
        elif technique == 'rand':
            samples_curr = sample_uniform(num_samples).to(V.device).to("cuda")
            in_safe = spc_render.point_is_in_sphere(spheres_copy, samples_curr)
            in_inner = spc_render.point_is_in_sphere(inner_spheres, samples_curr)
            in_interior = in_safe & (~in_inner)
            samples_curr = samples_curr[torch.squeeze(in_interior) == 1].to("cpu")
            samples.append(samples_curr)

            # sam_points_numpy = samples[0].detach().cpu().numpy()
            # np.savetxt("/home/wzj/project/dcc_copy/new_safe_spheres/sam_point100.txt", sam_points_numpy)
            # a = 0

            # samples.append(sample_uniform(num_samples).to(V.device))
        elif technique == 'importance':
            samples.append(torch.from_numpy(importaceSampler.sample(num_samples)))
            print("importance sampled")
            # sam_points_numpy = samples[0].detach().cpu().numpy()
            # np.savetxt("/home/wzj/project/dcc_copy/new_safe_spheres/sam_point100.txt", sam_points_numpy)

    samples = torch.cat(samples, dim=0).to(torch.float32)
    samples = samples.to("cuda")
    spheres_copy[:, 3] = abs(spheres_copy[:, 3])
    valid_idx = spc_render.point_is_in_sphere(spheres_copy, samples)
    # samples = samples[torch.squeeze(valid_idx) == 1].to("cpu") # 保留符合条件的采样点，在安全球内的。
    samples = samples[torch.squeeze(valid_idx) == 1]

    final_samples_indices = []
    final_sphere_indices = []
    spheres_expand = spheres_copy.unsqueeze(0)
    batch_size = 10000
    split = torch.split(samples, batch_size)
    batch_idx = 0
    print("分批次计算采样点和球的对")
    for u in tqdm(split):
        batch_samples_expand = u.unsqueeze(1)
        distance = torch.norm(batch_samples_expand[..., :3] - spheres_expand[..., :3], dim=2)
        is_inside = distance < spheres_copy[..., 3].unsqueeze(0)
        inside_indices = is_inside.nonzero(as_tuple=True)
        batch_sample_indices = inside_indices[0] + batch_idx * batch_size  # 调整索引以适应原始samples
        batch_sphere_indices = inside_indices[1]
        final_samples_indices.append(batch_sample_indices)
        final_sphere_indices.append(batch_sphere_indices)
        batch_idx = batch_idx + 1
    final_samples_indices = torch.cat(final_samples_indices)
    final_sphere_indices = torch.cat(final_sphere_indices)
    final_samples = samples[final_samples_indices]

    # 解决小球采样点数量不足问题

    # 初始化一个计数器张量，用于记录每个球的采样点数量
    counts = torch.zeros(len(spheres), dtype=torch.long, device=final_sphere_indices.device)

    # 更新计数器：对于 final_sphere_indices 中的每个索引，增加相应球的计数
    counts.index_add_(0, final_sphere_indices, torch.ones_like(final_sphere_indices))

    # 设置每个球应该拥有的最少采样点数
    min_samples_per_sphere = 500

    # 遍历所有球，检查是否需要额外采样
    for sphere_idx in tqdm(range(len(spheres))):
        # 获取当前球的中心和内外半径
        center = spheres_copy[sphere_idx, :3]
        inner_radius = inner_spheres[sphere_idx, 3]
        outer_radius = spheres_copy[sphere_idx, 3]

        # 计算当前球还需要多少采样点
        num_samples_needed = min_samples_per_sphere - counts[sphere_idx].item()

        # 如果需要额外采样
        if num_samples_needed > 0:
            # 在当前球内随机采样直到达到所需的采样点数
            extra_samples = sample_points_between_spheres(center, inner_radius, outer_radius, num_samples_needed)

            # 将额外的采样点添加到总的采样点集合中
            final_samples = torch.cat([final_samples, extra_samples], dim=0)

            # 更新对应的球索引集合
            new_sphere_indices = torch.full((num_samples_needed,), sphere_idx, dtype=torch.long, device=samples.device)
            final_sphere_indices = torch.cat([final_sphere_indices, new_sphere_indices], dim=0)

    num_samples_on_sphere = 300
    for sphere_idx in tqdm(range(len(spheres))):
        # 获取当前球的中心和内外半径
        center = spheres_copy[sphere_idx, :3]
        outer_radius = spheres_copy[sphere_idx, 3]
        extra_samples = sample_points_on_sphere(num_samples_on_sphere,outer_radius,center)
        final_samples = torch.cat([final_samples, extra_samples], dim=0)

        # 更新对应的球索引集合
        new_sphere_indices = torch.full((num_samples_on_sphere,), sphere_idx, dtype=torch.long, device=samples.device)
        final_sphere_indices = torch.cat([final_sphere_indices, new_sphere_indices], dim=0)
    # sphere_point_count = torch.bincount(sphere_indices)
    t = 0







    # sam_points_numpy = final_samples.detach().cpu().numpy()
    # np.savetxt("/home/wzj/PycharmProjects/sphere_resconstruct/my_sam_points/look_sample.txt", sam_points_numpy)
    # print("sam_point saved")
    #
    # return final_samples

    return final_samples.cpu().numpy(), final_sphere_indices.unsqueeze(1).cpu().numpy()


def random_sampling_on_custom_sphere(n, x_center, y_center, z_center, radius):
    points = torch.zeros((n, 3))

    for i in range(n):
        # Generate random values for theta and phi
        theta = 2 * math.pi * torch.rand(1)
        phi = torch.acos(2 * torch.rand(1) - 1)

        # Calculate the Cartesian coordinates of the point in the local coordinates
        x_local = radius * torch.sin(phi) * torch.cos(theta)
        y_local = radius * torch.sin(phi) * torch.sin(theta)
        z_local = radius * torch.cos(phi)

        # Apply center transformation to get global coordinates
        x = x_local + x_center
        y = y_local + y_center
        z = z_local + z_center

        points[i] = torch.tensor([x, y, z])

    return points


# def between_in_and_safe(safe_sphere,innere_sphere,points):
#
def sample_points_between_spheres(center, inner_radius, outer_radius, num_samples):
    # 确保内半径小于外半径
    assert inner_radius < outer_radius, "Inner radius must be smaller than outer radius."

    # 生成随机半径，位于内球半径和外球半径之间
    r = torch.rand(num_samples, device=center.device) * (outer_radius - inner_radius) + inner_radius

    # 生成随机角度
    theta = torch.acos(1 - 2 * torch.rand(num_samples, device=center.device))  # 仰角
    phi = torch.rand(num_samples, device=center.device) * 2 * math.pi  # 方位角

    # 转换为笛卡尔坐标
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    # 调整到球心位置
    points = torch.stack([x, y, z], dim=1) + center

    return points


# import numpy as np


def sample_points_on_sphere(num_points, radius=1.0, center=(0, 0, 0)):
    """
    使用 PyTorch 在球壳上均匀采样点，考虑球心不在原点的情况。

    参数:
        num_points: 采样点的数量。
        radius: 球的半径，默认为 1。
        center: 球心的坐标，默认为原点 (0, 0, 0)。

    返回:
        points: 形状为 (num_points, 3) 的张量，每行是球壳上的一个点的 xyz 坐标。
    """
    # 使用余弦分布采样倾角 theta，以确保在球面上均匀分布
    theta = torch.acos(1 - 2 * torch.rand(num_points)).to(device=center.device)
    phi = 2 * math.pi * torch.rand(num_points,device=center.device)  # 方位角 phi 均匀采样

    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)

    # 将 x, y, z 组合成点的坐标
    points = torch.stack([x, y, z], dim=1)

    # 将点平移至新的球心
    center_tensor = torch.tensor(center, dtype=torch.float32)
    points = points + center_tensor

    return points
