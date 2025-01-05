// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KAOLIN_OPS_RENDER_SPC_RAYTRACE_H_
#define KAOLIN_OPS_RENDER_SPC_RAYTRACE_H_

#ifdef WITH_CUDA
#include "../../spc_math.h"
#include <cmath>
#endif

#include <ATen/ATen.h>

namespace kaolin
{

    std::vector<at::Tensor> generate_primary_rays(
        uint imageH,
        uint imageW,
        at::Tensor Eye,
        at::Tensor At,
        at::Tensor Up,
        float fov,
        at::Tensor World);

    at::Tensor spc_raytrace(
        at::Tensor octree,
        at::Tensor points,
        at::Tensor pyramid,
        at::Tensor exsum,
        at::Tensor Org,
        at::Tensor Dir,
        uint targetLevel);

    at::Tensor remove_duplicate_rays(
        at::Tensor nuggets);

    at::Tensor mark_first_hit(
        at::Tensor nuggets);

    std::vector<torch::Tensor> spc_ray_aabb(
        torch::Tensor nuggets,
        torch::Tensor points,
        torch::Tensor ray_query,
        torch::Tensor ray_d,
        uint targetLevel,
        torch::Tensor info,
        torch::Tensor info_idxes,
        torch::Tensor cond,
        bool init);

    std::vector<at::Tensor> generate_shadow_rays(
        at::Tensor Org,
        at::Tensor Dir,
        at::Tensor Light,
        at::Tensor Plane);
    std::vector<torch::Tensor> spc_ray_sphere( // date 2022.9
        torch::Tensor nuggets,
        torch::Tensor ray_query,
        torch::Tensor ray_d,
        torch::Tensor x,
        torch::Tensor y,
        torch::Tensor z,
        torch::Tensor r,
        torch::Tensor info,
        torch::Tensor info_indexs,
        torch::Tensor cond,
        bool init);
    std::vector<torch::Tensor> compute_nearest_sphere(
        torch::Tensor InnerSpheres,
        torch::Tensor Points);
    at::Tensor compute_safe_distance( // date 2023.3.3
        torch::Tensor InnerSpheres,
        torch::Tensor Points);
    at::Tensor point_is_in_sphere(
        torch::Tensor Spheres,
        torch::Tensor Points);
    at::Tensor point_its_max_sphere(
        torch::Tensor Spheres,
        torch::Tensor Points);

    std::vector<torch::Tensor> compute_nearest_sphere_new(
        torch::Tensor InnerSpheres,
        torch::Tensor Points,
        torch::Tensor Point_cond);
    std::vector<torch::Tensor> ray_safe_sphere( // date 2023.3.9
        torch::Tensor ray_o,
        torch::Tensor ray_d,
        torch::Tensor cond,
        torch::Tensor SafeSpheres);
    at::Tensor get_clip(
        torch::Tensor ray_o,
        torch::Tensor ray_d,
        torch::Tensor cond,
        torch::Tensor SafeSpheres,
        torch::Tensor clip);
    std::vector<at::Tensor> sphere_raytrace(
        at::Tensor ray_o,
        at::Tensor ray_d,
        at::Tensor cond,
        at::Tensor SafeSpheres,
        at::Tensor InnerSpheres,
        int max_num);
    std::vector<at::Tensor> find_min_t(
        at::Tensor step,
        at::Tensor cond_type1,
        int max_num);
    std::vector<at::Tensor> test_raytrace(
        at::Tensor ray_o,
        at::Tensor ray_d,
        at::Tensor cond,
        at::Tensor SafeSpheres,
        at::Tensor InnerSpheres,
        int max_num);
    std::vector<at::Tensor> test_raytrace_new(
        at::Tensor ray_o,
        at::Tensor ray_d,
        at::Tensor cond,
        at::Tensor SafeSpheres,
        at::Tensor InnerSpheres,
        int max_num);
    std::vector<at::Tensor> find_min_t_new(
        at::Tensor step,
        at::Tensor cond_type1,
        at::Tensor nonzero_cnt,
        at::Tensor prefix_sum);
    std::vector<at::Tensor> find_nice_ball(
        at::Tensor compressed_d1,
        at::Tensor step,
        at::Tensor cond_type1,
        at::Tensor nonzero_cnt,
        at::Tensor prefix_sum,
        float epsilon);    
}

// namespace kaolin

#endif // KAOLIN_OPS_RENDER_SPC_RAYTRACE_H_
