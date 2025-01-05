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

#define CUB_NS_PREFIX \
  namespace kaolin    \
  {
#define CUB_NS_POSTFIX }

#include <stdio.h>
#include <ATen/ATen.h>
#include <cmath>
#define CUB_STDERR
#include <cub/device/device_scan.cuh>

#include "../../spc_math.h"
#include "spc_render_utils.h"
#include <sys/time.h>
namespace kaolin
{

  using namespace cub;
  using namespace std;
  using namespace at::indexing;

  __constant__ uint Order[8][8] = {
      {0, 1, 2, 4, 3, 5, 6, 7},
      {1, 0, 3, 5, 2, 4, 7, 6},
      {2, 0, 3, 6, 1, 4, 7, 5},
      {3, 1, 2, 7, 0, 5, 6, 4},
      {4, 0, 5, 6, 1, 2, 7, 3},
      {5, 1, 4, 7, 0, 3, 6, 2},
      {6, 2, 4, 7, 0, 3, 5, 1},
      {7, 3, 5, 6, 1, 2, 4, 0}};

  uint64_t GetStorageBytes(void *d_temp_storage, uint *d_Info, uint *d_PrefixSum,
                           uint max_total_points)
  {
    uint64_t temp_storage_bytes = 0;
    CubDebugExit(DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, d_Info,
        d_PrefixSum, max_total_points));
    return temp_storage_bytes;
  }

  __global__ void
  d_InitNuggets(uint num, uint2 *nuggets)
  {
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      nuggets[tidx].x = tidx; // ray idx
      nuggets[tidx].y = 0;
    }
  }

  __device__ bool
  d_FaceEval(ushort i, ushort j, float a, float b, float c)
  {
    float result[4];

    result[0] = a * i + b * j + c;
    result[1] = result[0] + a;
    result[2] = result[0] + b;
    result[3] = result[1] + b;

    float min = 1;
    float max = -1;

    for (int i = 0; i < 4; i++)
    {
      if (result[i] < min)
        min = result[i];
      if (result[i] > max)
        max = result[i];
    }

    return (min <= 0.0f && max >= 0.0f);
  }

  // This function will iterate over the nuggets (ray intersection proposals) and determine if they
  // result in an intersection. If they do, the info tensor is populated with the # of child nodes
  // as determined by the input octree.
  __global__ void
  d_Decide(uint num, point_data *points, float3 *rorg, float3 *rdir,
           uint2 *nuggets, uint *info, uchar *O, uint Level, uint notDone)
  {
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      uint ridx = nuggets[tidx].x;
      uint pidx = nuggets[tidx].y;
      point_data p = points[pidx];
      float3 o = rorg[ridx];
      float3 d = rdir[ridx];

      // Radius of voxel
      float s1 = 1.0 / ((float)(0x1 << Level));

      // Transform to [-1, 1]
      const float3 vc = make_float3(
          fmaf(s1, fmaf(2.0, p.x, 1.0), -1.0f),
          fmaf(s1, fmaf(2.0, p.y, 1.0), -1.0f),
          fmaf(s1, fmaf(2.0, p.z, 1.0), -1.0f));

      // Compute aux info (precompute to optimize)
      float3 sgn = ray_sgn(d);
      float3 ray_inv = make_float3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);

      // Perform AABB check
      if (ray_aabb(o, d, ray_inv, sgn, vc, s1) > 0.0)
      {
        // Count # of occupied voxels for expansion, if more levels are left
        info[tidx] = notDone ? __popc(O[pidx]) : 1;
      }
      else
      {
        info[tidx] = 0;
      }
    }
  }

  __global__ void
  d_Subdivide(uint num, uint2 *nuggetsIn, uint2 *nuggetsOut, float3 *rorg,
              point_data *points, uchar *O, uint *S, uint *info,
              uint *prefix_sum, uint Level)
  {
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num && info[tidx])
    {
      uint ridx = nuggetsIn[tidx].x;
      int pidx = nuggetsIn[tidx].y;
      point_data p = points[pidx];

      uint IdxBase = prefix_sum[tidx];

      uchar o = O[pidx];
      uint s = S[pidx];

      float scale = 1.0 / ((float)(0x1 << Level));
      float3 org = rorg[ridx];
      float x = (0.5f * org.x + 0.5f) - scale * ((float)p.x + 0.5);
      float y = (0.5f * org.y + 0.5f) - scale * ((float)p.y + 0.5);
      float z = (0.5f * org.z + 0.5f) - scale * ((float)p.z + 0.5);

      uint code = 0;
      if (x > 0)
        code = 4;
      if (y > 0)
        code += 2;
      if (z > 0)
        code += 1;

      for (uint i = 0; i < 8; i++)
      {
        uint j = Order[code][i];
        if (o & (0x1 << j))
        {
          uint cnt = __popc(o & ((0x2 << j) - 1)); // count set bits up to child - inclusive sum
          nuggetsOut[IdxBase].y = s + cnt;
          nuggetsOut[IdxBase++].x = ridx;
        }
      }
    }
  }

  __global__ void
  d_RemoveDuplicateRays(uint num, uint2 *nuggets, uint *info)
  {
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      if (tidx == 0)
        info[tidx] = 1;
      else
        info[tidx] = nuggets[tidx - 1].x == nuggets[tidx].x ? 0 : 1;
    }
  }

  __global__ void
  d_Compactify(uint num, uint2 *nuggetsIn, uint2 *nuggetsOut,
               uint *info, uint *prefix_sum)
  {
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num && info[tidx])
      nuggetsOut[prefix_sum[tidx]] = nuggetsIn[tidx];
  }

  uint spc_raytrace_cuda(
      uchar *d_octree,
      uint Level,
      uint targetLevel,
      point_data *d_points,
      uint *h_pyramid,
      uint *d_exsum,
      uint num,
      float3 *d_Org,
      float3 *d_Dir,
      uint2 *d_NuggetBuffers,
      uint *d_Info,
      uint *d_PrefixSum,
      void *d_temp_storage,
      uint64_t temp_storage_bytes)
  {

    uint *PyramidSum = h_pyramid + Level + 2;

    uint2 *d_Nuggets[2];
    d_Nuggets[0] = d_NuggetBuffers;
    d_Nuggets[1] = d_NuggetBuffers + KAOLIN_SPC_MAX_POINTS;

    int osize = PyramidSum[Level];

    d_InitNuggets<<<(num + 1023) / 1024, 1024>>>(num, d_Nuggets[0]);

    uint cnt, buffer = 0;

    // set first element to zero
    CubDebugExit(cudaMemcpy(d_PrefixSum, &buffer, sizeof(uint),
                            cudaMemcpyHostToDevice));

    for (uint l = 0; l <= targetLevel; l++)
    {
      d_Decide<<<(num + 1023) / 1024, 1024>>>(
          num, d_points, d_Org, d_Dir, d_Nuggets[buffer], d_Info, d_octree, l,
          targetLevel - l);
      CubDebugExit(DeviceScan::InclusiveSum(
          d_temp_storage, temp_storage_bytes, d_Info,
          d_PrefixSum + 1, num)); // start sum on second element
      cudaMemcpy(&cnt, d_PrefixSum + num, sizeof(uint), cudaMemcpyDeviceToHost);

      if (cnt == 0 || cnt > KAOLIN_SPC_MAX_POINTS)
        break; // either miss everything, or exceed memory allocation

      if (l < targetLevel)
      {
        d_Subdivide<<<(num + 1023) / 1024, 1024>>>(
            num, d_Nuggets[buffer], d_Nuggets[(buffer + 1) % 2], d_Org, d_points,
            d_octree, d_exsum, d_Info, d_PrefixSum, l);
      }
      else
      {
        d_Compactify<<<(num + 1023) / 1024, 1024>>>(
            num, d_Nuggets[buffer], d_Nuggets[(buffer + 1) % 2],
            d_Info, d_PrefixSum);
      }

      CubDebugExit(cudaGetLastError());

      buffer = (buffer + 1) % 2;
      num = cnt;
    }

    return cnt;
  }

  uint remove_duplicate_rays_cuda(
      uint num,
      uint2 *d_Nuggets0,
      uint2 *d_Nuggets1,
      uint *d_Info,
      uint *d_PrefixSum,
      void *d_temp_storage,
      uint64_t temp_storage_bytes)
  {
    uint cnt = 0;

    d_RemoveDuplicateRays<<<(num + 1023) / 1024, 1024>>>(num, d_Nuggets0, d_Info);
    CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_Info, d_PrefixSum, num + 1));
    cudaMemcpy(&cnt, d_PrefixSum + num, sizeof(uint), cudaMemcpyDeviceToHost);
    d_Compactify<<<(num + 1023) / 1024, 1024>>>(num, d_Nuggets0, d_Nuggets1, d_Info, d_PrefixSum);

    return cnt;
  }

  void mark_first_hit_cuda(
      uint num,
      uint2 *d_Nuggets,
      uint *d_Info)
  {
    d_RemoveDuplicateRays<<<(num + 1023) / 1024, 1024>>>(num, d_Nuggets, d_Info);
  }

  ////////// generate rays //////////////////////////////////////////////////////////////////////////

  __global__ void
  d_generate_rays(uint num, uint imageW, uint imageH, float4x4 mM,
                  float3 *rayorg, float3 *raydir)
  {
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      uint px = tidx % imageW;
      uint py = tidx / imageW;

      float4 a = mul4x4(make_float4(0.0f, 0.0f, 1.0f, 0.0f), mM);
      float4 b = mul4x4(make_float4(px, py, 0.0f, 1.0f), mM);
      // float3 org = make_float3(M.m[3][0], M.m[3][1], M.m[3][2]);

      rayorg[tidx] = make_float3(a.x, a.y, a.z);
      raydir[tidx] = make_float3(b.x, b.y, b.z);
    }
  }

  void generate_primary_rays_cuda(uint imageW, uint imageH, float4x4 &mM,
                                  float3 *d_Org, float3 *d_Dir)
  {
    uint num = imageW * imageH;

    d_generate_rays<<<(num + 1023) / 1024, 1024>>>(num, imageW, imageH, mM, d_Org, d_Dir);
  }

  ////////// generate shadow rays /////////

  __global__ void
  d_plane_intersect_rays(uint num, float3 *d_Org, float3 *d_Dir,
                         float3 *d_Dst, float4 plane, uint *info)
  {
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      float3 org = d_Org[tidx];
      float3 dir = d_Dir[tidx];

      float a = org.x * plane.x + org.y * plane.y + org.z * plane.z + plane.w;
      float b = dir.x * plane.x + dir.y * plane.y + dir.z * plane.z;

      if (fabs(b) > 1e-3)
      {
        float t = -a / b;
        if (t > 0.0f)
        {
          d_Dst[tidx] = make_float3(org.x + t * dir.x, org.y + t * dir.y, org.z + t * dir.z);
          info[tidx] = 1;
        }
        else
        {
          info[tidx] = 0;
        }
      }
      else
      {
        info[tidx] = 0;
      }
    }
  }

  __global__ void
  d_Compactify2(uint num, float3 *pIn, float3 *pOut, uint *map,
                uint *info, uint *prefix_sum)
  {
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num && info[tidx])
    {
      pOut[prefix_sum[tidx]] = pIn[tidx];
      map[prefix_sum[tidx]] = tidx;
    }
  }

  __global__ void
  d_SetShadowRays(uint num, float3 *src, float3 *dst, float3 light)
  {
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (tidx < num)
    {
      dst[tidx] = normalize(src[tidx] - light);
      src[tidx] = light;
    }
  }

  uint generate_shadow_rays_cuda(
      uint num,
      float3 *org,
      float3 *dir,
      float3 *src,
      float3 *dst,
      uint *map,
      float3 &light,
      float4 &plane,
      uint *info,
      uint *prefixSum,
      void *d_temp_storage,
      uint64_t temp_storage_bytes)
  {
    uint cnt = 0;
    d_plane_intersect_rays<<<(num + 1023) / 1024, 1024>>>(
        num, org, dir, dst, plane, info);
    CubDebugExit(DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, info, prefixSum, num));
    cudaMemcpy(&cnt, prefixSum + num - 1, sizeof(uint), cudaMemcpyDeviceToHost);
    d_Compactify2<<<(num + 1023) / 1024, 1024>>>(
        num, dst, src, map, info, prefixSum);
    d_SetShadowRays<<<(cnt + 1023) / 1024, 1024>>>(cnt, src, dst, light);

    return cnt;
  }

  // This kernel will iterate over Nuggets, instead of iterating over rays
  __global__ void ray_aabb_kernel(
      const float3 *__restrict__ query,   // ray query array
      const float3 *__restrict__ ray_d,   // ray direction array
      const float3 *__restrict__ ray_inv, // inverse ray direction array
      const int2 *__restrict__ nuggets,   // nugget array (ray-aabb correspondences)
      const float3 *__restrict__ points,  // 3d coord array
      const int *__restrict__ info,       // binary array denoting beginning of nugget group
      const int *__restrict__ info_idxes, // array of active nugget indices
      const float r,                      // radius of aabb
      const bool init,                    // first run?
      float *__restrict__ d,              // distance
      bool *__restrict__ cond,            // true if hit
      int *__restrict__ pidx,             // index of 3d coord array
      const int num_nuggets,              // # of nugget indices
      const int n                         // # of active nugget indices
  )
  {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > n)
      return;

    for (int _i = idx; _i < n; _i += stride)
    {
      // Get index of corresponding nugget
      int i = info_idxes[_i];

      // Get index of ray
      uint ridx = nuggets[i].x;

      // If this ray is already terminated, continue
      if (!cond[ridx] && !init)
        continue;

      bool _hit = false;

      // Sign bit
      const float3 sgn = ray_sgn(ray_d[ridx]);

      int j = 0;
      // In order traversal of the voxels
      do
      {
        // Get the vc from the nugget
        uint _pidx = nuggets[i].y; // Index of points

        // Center of voxel
        const float3 vc = make_float3(
            fmaf(r, fmaf(2.0, points[_pidx].x, 1.0), -1.0f),
            fmaf(r, fmaf(2.0, points[_pidx].y, 1.0), -1.0f),
            fmaf(r, fmaf(2.0, points[_pidx].z, 1.0), -1.0f));

        float _d = ray_aabb(query[ridx], ray_d[ridx], ray_inv[ridx], sgn, vc, r);

        if (_d != 0.0)
        {
          _hit = true;
          pidx[ridx] = _pidx;
          cond[ridx] = _hit;
          if (_d > 0.0)
          {
            d[ridx] = _d;
          }
        }

        ++i;
        ++j;

      } while (i < num_nuggets && info[i] != 1 && _hit == false);

      if (!_hit)
      {
        // Should only reach here if it misses
        cond[ridx] = false;
        d[ridx] = 100;
      }
    }
  }

  void ray_aabb_cuda(
      const float3 *query,   // ray query array
      const float3 *ray_d,   // ray direction array
      const float3 *ray_inv, // inverse ray direction array
      const int2 *nuggets,   // nugget array (ray-aabb correspondences)
      const float3 *points,  // 3d coord array
      const int *info,       // binary array denoting beginning of nugget group
      const int *info_idxes, // array of active nugget indices
      const float r,         // radius of aabb
      const bool init,       // first run?
      float *d,              // distance
      bool *cond,            // true if hit
      int *pidx,             // index of 3d coord array
      const int num_nuggets, // # of nugget indices
      const int n)
  { // # of active nugget indices

    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    ray_aabb_kernel<<<blocks, threads>>>(
        query, ray_d, ray_inv, nuggets, points, info, info_idxes, r, init, d, cond, pidx, num_nuggets, n);
  }
  __global__ void ray_sphere_kernel(
      const float3 *__restrict__ query, // ray query array
      const float3 *__restrict__ ray_d, // ray direction array
      // const float3* __restrict__ ray_inv,   // inverse ray direction array
      const int2 *__restrict__ nuggets, // nugget array (ray-aabb correspondences)
      // const float3* __restrict__ points,    // 3d coord array

      const float *__restrict__ x,
      const float *__restrict__ y,
      const float *__restrict__ z,
      const float *__restrict__ r,
      // array of active nugget indices
      // radius of sphere
      // const bool init,                      // first run?
      float *__restrict__ d,
      const int *__restrict__ info, // binary array denoting beginning of nugget group
      const int *__restrict__ info_idxes,
      bool init,
      bool *__restrict__ cond, // true if hit
      // int* __restrict__ pidx,               // index of 3d coord array
      const int num_nuggets,
      const int n // # of nugget indices
  )
  {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > n)
      return;

    for (int _i = idx; _i < n; _i += stride)
    {
      // Get index of corresponding nugget
      int i = info_idxes[_i];

      // Get index of ray
      uint ridx = nuggets[i].x;
      // If this ray is already terminated, continue
      if (!cond[ridx] && !init)
        continue;

      bool _hit = false;

      // Sign bit
      // const float3 sgn = ray_sgn(ray_d[ridx]);

      int j = 0;
      // In order traversal of the voxels
      do
      {

        uint _sidx = nuggets[i].y; // Index of sphere

        float _d = ray_sphere(query[ridx], ray_d[ridx], x[_sidx], y[_sidx], z[_sidx], r[_sidx]);

        if (_d != 0.0)
        {
          _hit = true;

          cond[ridx] = _hit;
          if (_d > 0.0)
          {
            d[ridx] = _d;
          }
        }

        ++i;
        ++j;

      } while (i < num_nuggets && info[i] != 1 && _hit == false);

      if (!_hit)
      {
        // Should only reach here if it misses
        // cond[ridx] = false;
        // d[ridx] = 100;
        d[ridx] = 0;
      }
    }
  }

  void ray_sphere_cuda(
      const float3 *query, // ray query array
      const float3 *ray_d, // ray direction array
      const int2 *nuggets, // nugget array (ray-aabb correspondences)
                           // array of active nugget indices
      const float *x,
      const float *y,
      const float *z,
      const float *r,
      float *d,
      const int *info, // binary array denoting beginning of nugget group
      const int *info_idxes,
      bool init,             // distance
      bool *cond,            // true if hit
      const int num_nuggets, // # of nugget indices
      const int n)
  { // # of active nugget indices

    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    ray_sphere_kernel<<<blocks, threads>>>(
        query, ray_d, nuggets, x, y, z, r, d, info, info_idxes, init, cond, num_nuggets, n);
  }
  __global__ void safe_dis_kernel(
      const float4 *__restrict__ spheres,
      const float3 *__restrict__ points,
      const int ns,
      const int np,
      float *__restrict__ safe_dis_list)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > np)
    {
      return;
    }

    for (int _i = idx; _i < np; _i += stride)
    {
      float min_i = 100.0;
      for (int j = 0; j < ns; j++)
      {
        float dis_ij = sqrt((points[_i].x - spheres[j].x) * (points[_i].x - spheres[j].x) +
                            (points[_i].y - spheres[j].y) * (points[_i].y - spheres[j].y) +
                            (points[_i].z - spheres[j].z) * (points[_i].z - spheres[j].z)) -
                       spheres[j].w;
        if (dis_ij < min_i)
          min_i = dis_ij;
      }
      safe_dis_list[_i] = min_i;
      // if(min_i>max_all)
      //     max_all = min_i;
    }
    // safe_dist = max_all;
    // safe_dis[0] = safe_dist;

    // uint _i = blockDim.x * blockIdx.x + threadIdx.x;
    // safe_dis_list[_i]=0.88;
    // // if(_i<np){
    // //   float min_i = 100.0;
    // //     for(int j = 0;j < ns; j++){
    // //         float dis_ij = sqrt( (points[_i].x-spheres[j].x)*(points[_i].x-spheres[j].x)+
    // //                              (points[_i].y-spheres[j].y)*(points[_i].y-spheres[j].y)+
    // //                              (points[_i].z-spheres[j].z)*(points[_i].z-spheres[j].z)  )-spheres[j].w;
    // //         if(dis_ij<min_i)
    // //             min_i = dis_ij;
    // //     }
    // //     safe_dis_list[_i] = min_i;
    // //     // if(min_i>max_all)
    // //     //     max_all = min_i;
    // // }
  }

  void safe_dist_cuda(
      const float4 *spheres,
      const float3 *points,
      const int ns,
      const int np,
      float *safe_dis_list)
  {

    const int threads = 128;
    const int blocks = (np + threads - 1) / threads;
    struct timeval time1;
    gettimeofday(&time1, NULL);
    printf("us: %ld\n", time1.tv_usec);

    safe_dis_kernel<<<blocks, threads>>>(spheres, points, ns, np, safe_dis_list);

    struct timeval time2;
    gettimeofday(&time2, NULL);

    printf("us: %ld\n", time2.tv_usec);
  }
  __global__ void safe_ray_sphere_kernel(

      const float4 *__restrict__ spheres,
      const float3 *__restrict__ ray_o,
      const float3 *__restrict__ ray_d,
      const int ns,
      const int nr,
      float *__restrict__ d,
      bool *__restrict__ cond)
  {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > nr)
    {
      return;
    }
    for (int _i = idx; _i < nr; _i += stride)
    {
      if (!cond[_i])
      { // yasuo
        continue;
      }
      float min_dis = 100.0;
      bool hit = false;
      for (int j = 0; j < ns; j++)
      {

        float t1 = ray_sphere_out(ray_o[_i], ray_d[_i], spheres[j].x, spheres[j].y, spheres[j].z, spheres[j].w);
        // printf("%f",t1);
        if (t1 != 0.0)
        { // 1e-7
          // printf("%f ",t1);
          hit = true;
          cond[_i] = hit;
          if (t1 < min_dis && t1 > 0.0)
          {
            min_dis = t1;
          }
        }
      }
      if (hit == true)
      {
        d[_i] = min_dis + 1e-6f;
      }
      else
      {
        cond[_i] = false;
        d[_i] = 0.0;
      }
    }
  }
  void safe_ray_sphere_cuda(
      const float4 *spheres,
      const float3 *ray_o,
      const float3 *ray_d,
      const int ns,
      const int nr,
      float *d,
      bool *cond)
  {

    const int threads = 128;
    const int blocks = (nr + threads - 1) / threads;
    safe_ray_sphere_kernel<<<blocks, threads>>>(spheres, ray_o, ray_d, ns, nr, d, cond);
  }
  // namespace kaolin

  __global__ void near_and_dist_kernel(
      const float4 *__restrict__ spheres,
      const float3 *__restrict__ points,
      const int ns,
      const int np,
      int *__restrict__ safe_dis_list,
      float *__restrict__ distance_list)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > np)
    {
      return;
    }
    for (int _i = idx; _i < np; _i += stride)
    {
      float min_d = 100.0;
      for (int j = 0; j < ns; j++)
      {
        float dis_ij = sqrt((points[_i].x - spheres[j].x) * (points[_i].x - spheres[j].x) +
                            (points[_i].y - spheres[j].y) * (points[_i].y - spheres[j].y) +
                            (points[_i].z - spheres[j].z) * (points[_i].z - spheres[j].z)) -
                       spheres[j].w;
        if (dis_ij < min_d)
        {
          min_d = dis_ij;
          safe_dis_list[_i] = j;
        }
      }
      distance_list[_i] = min_d;
    }
  }

  __global__ void near_and_dist_new_kernel(
      const float4 *__restrict__ spheres,
      const float3 *__restrict__ points,
      const int ns,
      const int np,
      int *__restrict__ safe_dis_list,
      float *__restrict__ distance_list,
      const bool *__restrict__ point_cond)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > np)
    {
      return;
    }
    for (int _i = idx; _i < np; _i += stride)
    {
      if (point_cond[_i])
      {
        float min_d = 100.0;
        for (int j = 0; j < ns; j++)
        {
          float dis_ij = sqrt((points[_i].x - spheres[j].x) * (points[_i].x - spheres[j].x) +
                              (points[_i].y - spheres[j].y) * (points[_i].y - spheres[j].y) +
                              (points[_i].z - spheres[j].z) * (points[_i].z - spheres[j].z)) -
                         spheres[j].w;
          if (dis_ij < min_d)
          {
            min_d = dis_ij;
            safe_dis_list[_i] = j;
          }
        }
        distance_list[_i] = min_d;
      }
    }
  }
  void return_points_nearest_sphere_cuda(
      const float4 *spheres,
      const float3 *points,
      const int ns,
      const int np,
      int *near_sphere_list,
      float *distance_list)
  {
    const int threads = 128;
    const int blocks = (np + threads - 1) / threads;
    near_and_dist_kernel<<<blocks, threads>>>(spheres, points, ns, np, near_sphere_list, distance_list);
  }
  void return_points_nearest_sphere_new_cuda(
      const float4 *spheres,
      const float3 *points,
      const int ns,
      const int np,
      int *near_sphere_list,
      float *distance_list,
      bool *point_cond)
  {
    const int threads = 128;
    const int blocks = (np + threads - 1) / threads;
    near_and_dist_new_kernel<<<blocks, threads>>>(spheres, points, ns, np, near_sphere_list, distance_list, point_cond);
  }
  __global__ void is_in_kernel(
      const float4 *__restrict__ spheres,
      const float3 *__restrict__ points,
      const int ns,
      const int np,
      int *__restrict__ is_in_list)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > np)
    {
      return;
    }
    for (int _i = idx; _i < np; _i += stride)
    {

      for (int j = 0; j < ns; j++)
      {
        float oc = (points[_i].x - spheres[j].x) * (points[_i].x - spheres[j].x) +
                   (points[_i].y - spheres[j].y) * (points[_i].y - spheres[j].y) +
                   (points[_i].z - spheres[j].z) * (points[_i].z - spheres[j].z);
        if (oc <= spheres[j].w * spheres[j].w)
        {

          is_in_list[_i] = 1;
          break;
        }
      }
    }
  }
  void is_in_cuda(
      const float4 *spheres,
      const float3 *points,
      const int ns,
      const int np,
      int *is_in_list)
  {
    const int threads = 128;
    const int blocks = (np + threads - 1) / threads;
    is_in_kernel<<<blocks, threads>>>(spheres, points, ns, np, is_in_list);
  }
  __global__ void is_in_kernel_new(
      const float4 *__restrict__ spheres,
      const float3 *__restrict__ points,
      const int ns,
      const int np,
      int *__restrict__ is_in_list)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > np)
    {
      return;
    }
    for (int _i = idx; _i < np; _i += stride)
    {

      float min_radius = 100.0;
      int min_sphere_idx = -1;
      for (int j = 0; j < ns; j++)
      {
        float oc = (points[_i].x - spheres[j].x) * (points[_i].x - spheres[j].x) +
                   (points[_i].y - spheres[j].y) * (points[_i].y - spheres[j].y) +
                   (points[_i].z - spheres[j].z) * (points[_i].z - spheres[j].z);
        if (oc <= spheres[j].w * spheres[j].w)
        {

          if (spheres[j].w < min_radius)
          {
            min_radius = spheres[j].w;
            min_sphere_idx = j;
          }
        }
      }
      is_in_list[_i] = min_sphere_idx;
    }
  }
  void is_in_cuda_new(
      const float4 *spheres,
      const float3 *points,
      const int ns,
      const int np,
      int *is_in_list)
  {
    const int threads = 128;
    const int blocks = (np + threads - 1) / threads;
    is_in_kernel_new<<<blocks, threads>>>(spheres, points, ns, np, is_in_list);
  }
  __global__ void get_clip_kernel(

      const float4 *__restrict__ spheres,
      const float3 *__restrict__ ray_o,
      const float3 *__restrict__ ray_d,
      const int ns,
      const int nr,
      float *__restrict__ clip,
      bool *__restrict__ cond)
  {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > nr)
    {
      return;
    }
    for (int _i = idx; _i < nr; _i += stride)
    {
      if (!cond[_i])
      { // yasuo
        continue;
      }
      float max_dis = -1.0;
      bool hit = false;
      for (int j = 0; j < ns; j++)
      {

        float t2 = ray_sphere_out1(ray_o[_i], ray_d[_i], spheres[j].x, spheres[j].y, spheres[j].z, spheres[j].w);
        // printf("%f",t1);
        if (t2 != 0.0)
        { // 1e-7
          // printf("%f ",t1);
          hit = true;
          if (t2 > max_dis && t2 > 0.0)
          {
            max_dis = t2;
          }
        }
      }
      if (hit == true)
      {
        clip[_i] = max_dis;
      }
    }
  }

  void get_clip_cuda(
      const float4 *spheres,
      const float3 *ray_o,
      const float3 *ray_d,
      const int ns,
      const int nr,
      float *clip,
      bool *cond)
  {

    const int threads = 128;
    const int blocks = (nr + threads - 1) / threads;
    get_clip_kernel<<<blocks, threads>>>(spheres, ray_o, ray_d, ns, nr, clip, cond);
  }
  __global__ void list_kernel(
      const float4 *__restrict__ safespheres,
      const float4 *__restrict__ innerspheres,
      const float3 *__restrict__ ray_o,
      const float3 *__restrict__ ray_d,
      bool *__restrict__ cond,
      const int ns,
      const int nr,
      const int max_num,
      int *__restrict__ list,
      int *__restrict__ ray_type,
      bool *__restrict__ list_cond,
      float *__restrict__ list_d1,
      float *__restrict__ list_d2,
      int *__restrict__ type2_index,
      float *__restrict__ type2_d)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > nr)
    {
      return;
    }
    for (int _i = idx; _i < nr; _i += stride)
    {
      bool has_positive_t1 = false;
      int curr_intersect_num = 0;
      bool safe_sphere_intersection = false;
      bool first_inner_sphere_intersection = false;
      float min_out_d = 100.0;
      int min_out_index = -1; // first_safe_sphere_index
      for (int j = 0; j < ns; j++)
      {
        // judge safesphere
        float3 oc = make_float3(ray_o[_i].x - safespheres[j].x, ray_o[_i].y - safespheres[j].y, ray_o[_i].z - safespheres[j].z);
        float l = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z;
        float b = (oc.x * ray_d[_i].x + oc.y * ray_d[_i].y + oc.z * ray_d[_i].z) * 2.0;
        float c = l - safespheres[j].w * safespheres[j].w;
        if (b * b - 4 * c >= 0)
        {
          float t1 = (-b) * 0.5 - 0.5 * sqrt(b * b - 4.0 * c);
          float t2 = (-b) * 0.5 + 0.5 * sqrt(b * b - 4.0 * c);
          if (t1 > 0)
          {
            safe_sphere_intersection = true;
            list[_i * max_num + curr_intersect_num] = j;
            list_cond[_i * max_num + curr_intersect_num] = true;
            list_d1[_i * max_num + curr_intersect_num] = t1 + 1e-6f;
            list_d2[_i * max_num + curr_intersect_num] = t2 - 1e-6f;
            has_positive_t1 = true;
            curr_intersect_num++;
          }
          if (t1 < min_out_d && t1 > 0)
          {
            min_out_d = t1;
            min_out_index = j;
          }
        }
      }

      if (safe_sphere_intersection)
      {
        float min_inner_d = 100.0;
        int min_inner_index = -2;
        for (int j = 0; j < ns; j++)
        {
          // judge innersphere
          float3 oc = make_float3(ray_o[_i].x - innerspheres[j].x, ray_o[_i].y - innerspheres[j].y, ray_o[_i].z - innerspheres[j].z);
          float l = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z;
          float b = (oc.x * ray_d[_i].x + oc.y * ray_d[_i].y + oc.z * ray_d[_i].z) * 2.0;
          float c = l - innerspheres[j].w * innerspheres[j].w;
          if (b * b - 4 * c >= 0)
          {
            float t1 = (-b) * 0.5 - 0.5 * sqrt(b * b - 4.0 * c);
            if (t1 < min_inner_d && t1 > 0)
            {
              min_inner_d = t1;
              min_inner_index = j;
            }
          }
        }
        if (min_inner_index == min_out_index)
        {
          first_inner_sphere_intersection = true;
          type2_index[_i] = min_inner_index;
          type2_d[_i] = min_out_d + 1e-6f;
        }
      }

      if (safe_sphere_intersection && first_inner_sphere_intersection)
      {
        ray_type[_i] = 2;
        cond[_i] = true;
      }
      else if ((!first_inner_sphere_intersection) && safe_sphere_intersection)
      {
        ray_type[_i] = 1;
        cond[_i] = true;
      }
      else if ((!first_inner_sphere_intersection) && (!safe_sphere_intersection))
      {
        ray_type[_i] = 0;
        cond[_i] = false;
      }
    }
  }
  void get_list_and_cond_cuda(
      const float4 *safespheres,
      const float4 *innerspheres,
      const float3 *ray_o,
      const float3 *ray_d,
      bool *cond,
      const int ns,
      const int nr,
      const int max_num,
      int *list,
      int *ray_type,
      bool *list_cond,
      float *list_d1,
      float *list_d2,
      int *type2_index,
      float *type2_d)
  {
    const int threads = 128;
    const int blocks = (nr + threads - 1) / threads;
    list_kernel<<<blocks, threads>>>(safespheres, innerspheres, ray_o, ray_d, cond, ns, nr, max_num, list, ray_type, list_cond, list_d1, list_d2, type2_index, type2_d);
  }

  __global__ void get_min_kernel(
      const int max_num,
      const int n,
      const float *__restrict__ step,
      const bool *__restrict__ cond_type1,
      bool *__restrict__ list_cond,
      float *__restrict__ min_t,
      int *__restrict__ list_of_sphere)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > n)
    {
      return;
    }
    for (int _i = idx; _i < n; _i += stride)
    {
      float min_tt = 100.0;
      bool flag = false;
      int mint_sphere_index = -1;
      for (int j = 0; j < max_num; j++)
      {
        if (!cond_type1[_i * max_num + j])
          continue;
        if (step[_i * max_num + j] < min_tt)
        {
          min_tt = step[_i * max_num + j];
          flag = true;
          mint_sphere_index = j;
        }
      }
      min_t[_i] = min_tt;
      list_cond[_i] = flag;
      list_of_sphere[_i] = mint_sphere_index;
    }
  }
  void get_min_cuda(
      const int max_num,
      const int n,
      const float *step,
      const bool *cond_type1,
      bool *list_cond,
      float *min_t,
      int *list_of_sphere)
  {
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    get_min_kernel<<<blocks, threads>>>(max_num, n, step, cond_type1, list_cond, min_t, list_of_sphere);
  }

  __global__ void list_test_kernel(
      const float4 *__restrict__ safespheres,
      const float4 *__restrict__ innerspheres,
      const float3 *__restrict__ ray_o,
      const float3 *__restrict__ ray_d,
      bool *__restrict__ cond,
      const int ns,
      const int nr,
      const int max_num,
      int *__restrict__ list,
      int *__restrict__ ray_type,
      bool *__restrict__ list_cond,
      float *__restrict__ list_d1,
      float *__restrict__ list_d2,
      int *__restrict__ type2_in_index,
      int *__restrict__ type2_out_index,
      float *__restrict__ type2_d,
      float *__restrict__ type2_t2,
      float *__restrict__ type2_in_d)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > nr)
    {
      return;
    }
    for (int _i = idx; _i < nr; _i += stride)
    {
      bool has_positive_t1 = false;
      int curr_intersect_num = 0;
      bool safe_sphere_intersection = false;
      bool inner_sphere_intersection = false;
      float min_out_d = 100.0;
      float min_out_t2 = 100.0;
      int min_out_index = -1; // first_safe_sphere_index
      for (int j = 0; j < ns; j++)
      {
        // judge safesphere
        float3 oc = make_float3(ray_o[_i].x - safespheres[j].x, ray_o[_i].y - safespheres[j].y, ray_o[_i].z - safespheres[j].z);
        float l = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z;
        float b = (oc.x * ray_d[_i].x + oc.y * ray_d[_i].y + oc.z * ray_d[_i].z) * 2.0;
        float c = l - safespheres[j].w * safespheres[j].w;
        if (b * b - 4 * c >= 0)
        {
          float t1 = (-b) * 0.5 - 0.5 * sqrt(b * b - 4.0 * c);
          float t2 = (-b) * 0.5 + 0.5 * sqrt(b * b - 4.0 * c);

          if (t1 > 0)
          {
            safe_sphere_intersection = true;
            list[_i * max_num + curr_intersect_num] = j;
            list_cond[_i * max_num + curr_intersect_num] = true;
            list_d1[_i * max_num + curr_intersect_num] = t1 + 1e-6f;
            list_d2[_i * max_num + curr_intersect_num] = t2 - 1e-6f;
            has_positive_t1 = true;
            curr_intersect_num++;
          }
          if (t1 < min_out_d && t1 > 0)
          {
            min_out_d = t1;
            min_out_index = j;
          }
          if (t2 < min_out_t2 && t2 > 0)
          {
            min_out_t2 = t2;
          }
        }
      }
      float min_inner_d = 100.0;
      int min_inner_index = -2;
      if (safe_sphere_intersection)
      {
        // float min_inner_d = 100.0;
        // int min_inner_index = -2;
        for (int j = 0; j < ns; j++)
        {
          // judge innersphere
          float3 oc = make_float3(ray_o[_i].x - innerspheres[j].x, ray_o[_i].y - innerspheres[j].y, ray_o[_i].z - innerspheres[j].z);
          float l = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z;
          float b = (oc.x * ray_d[_i].x + oc.y * ray_d[_i].y + oc.z * ray_d[_i].z) * 2.0;
          float c = l - innerspheres[j].w * innerspheres[j].w;
          if (b * b - 4 * c >= 0)
          {

            float t1 = (-b) * 0.5 - 0.5 * sqrt(b * b - 4.0 * c);
            if (t1 < min_inner_d && t1 > 0)
            {
              min_inner_d = t1;
              min_inner_index = j;
              inner_sphere_intersection = true;
            }
          }
        }

      }
      if (inner_sphere_intersection)
      {
        type2_out_index[_i] = min_out_index;
        type2_d[_i] = min_out_d + 1e-6f;
        type2_t2[_i] = min_out_t2 - 1e-6f;
        type2_in_index[_i] = min_inner_index;
        type2_in_d[_i] = min_inner_d + 1e-6f;
      }

      if (safe_sphere_intersection && inner_sphere_intersection)
      {
        ray_type[_i] = 2;
        cond[_i] = true;
      }
      else if ((!inner_sphere_intersection) && safe_sphere_intersection)
      {
        ray_type[_i] = 1;
        cond[_i] = true;
      }
      else if ((!inner_sphere_intersection) && (!safe_sphere_intersection))
      {
        ray_type[_i] = 0;
        cond[_i] = false;
      }
    }
  }
  __global__ void list_test_new_kernel(
      const float4 *__restrict__ safespheres,
      const float4 *__restrict__ innerspheres,
      const float3 *__restrict__ ray_o,
      const float3 *__restrict__ ray_d,
      bool *__restrict__ cond,
      const int ns,
      const int nr,
      const int max_num,
      int *__restrict__ list,
      bool *__restrict__ list_cond,
      float *__restrict__ list_d1,
      float *__restrict__ list_d2)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > nr)
    {
      return;
    }
    for (int _i = idx; _i < nr; _i += stride) //[_i] is for a ray
    {
      float min_inner_t1 = 100.0;
      for (int j = 0; j < ns; j++)
      { // get min inner_sphere_t1, if not intersect,make it 100.0
        float3 oc = make_float3(ray_o[_i].x - innerspheres[j].x, ray_o[_i].y - innerspheres[j].y, ray_o[_i].z - innerspheres[j].z);
        float l = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z;
        float b = (oc.x * ray_d[_i].x + oc.y * ray_d[_i].y + oc.z * ray_d[_i].z) * 2.0;
        float c = l - innerspheres[j].w * innerspheres[j].w;
        if (b * b - 4 * c >= 0)
        {
          float t1 = (-b) * 0.5 - 0.5 * sqrt(b * b - 4.0 * c);
          if (t1 < min_inner_t1 && t1 > 0)
            {
              min_inner_t1 = t1;
            }
        }
      }
      bool intersect = false;
      int curr_intersect_num = 0;
      for(int j = 0;j < ns; j++)//for every safe_sphere
      {
        float3 oc = make_float3(ray_o[_i].x - safespheres[j].x, ray_o[_i].y - safespheres[j].y, ray_o[_i].z - safespheres[j].z);
        float l = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z;
        float b = (oc.x * ray_d[_i].x + oc.y * ray_d[_i].y + oc.z * ray_d[_i].z) * 2.0;
        float c = l - safespheres[j].w * safespheres[j].w;
        if (b * b - 4 * c >= 0)
        {
          float t1 = (-b) * 0.5 - 0.5 * sqrt(b * b - 4.0 * c);
          float t2 = (-b) * 0.5 + 0.5 * sqrt(b * b - 4.0 * c);
          if(t1<min_inner_t1&&t1>0){//t1<min_inner_t1&&t1>0
            intersect=true;
            list_cond[_i * max_num + curr_intersect_num] = true;
            list[_i * max_num + curr_intersect_num] = j;//sphere idx
            list_d1[_i * max_num + curr_intersect_num] = t1 + 1e-6f;
            list_d2[_i * max_num + curr_intersect_num] = t2 - 1e-6f;
            curr_intersect_num++;
          }
        }
      }
      if(intersect){
        cond[_i] = true;
      }else{
        cond[_i] = false;
      }
    }
  }
  void get_list_and_cond_cuda_test_new(
      const float4 *safespheres,
      const float4 *innerspheres,
      const float3 *ray_o,
      const float3 *ray_d,
      bool *cond,
      const int ns,
      const int nr,
      const int max_num,
      int *list,
      bool *list_cond,
      float *list_d1,
      float *list_d2
 )
  {
    const int threads = 128;
    const int blocks = (nr + threads - 1) / threads;
    list_test_new_kernel<<<blocks, threads>>>(safespheres, innerspheres, ray_o, ray_d, cond, ns, nr, max_num, list, list_cond, list_d1, list_d2);
  }
  void get_list_and_cond_cuda_test(
      const float4 *safespheres,
      const float4 *innerspheres,
      const float3 *ray_o,
      const float3 *ray_d,
      bool *cond,
      const int ns,
      const int nr,
      const int max_num,
      int *list,
      int *ray_type,
      bool *list_cond,
      float *list_d1,
      float *list_d2,
      int *type2_in_index,
      int *type2_out_index,
      float *type2_d,
      float *type2_t2,
      float *type2_in_d)
  {
    const int threads = 128;
    const int blocks = (nr + threads - 1) / threads;
    list_test_kernel<<<blocks, threads>>>(safespheres, innerspheres, ray_o, ray_d, cond, ns, nr, max_num, list, ray_type, list_cond, list_d1, list_d2, type2_in_index, type2_out_index, type2_d, type2_t2, type2_in_d);
  }
  __global__ void get_min_kernel_new(
      const int *nonzero_cnt,
      const int *prefix_sum,
      const int n,
      const float *__restrict__ step,
      const bool *__restrict__ cond_type1,
      bool *__restrict__ list_cond,
      float *__restrict__ min_t,
      int *__restrict__ list_of_sphere)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > n)
    {
      return;
    }
    for (int _i = idx; _i < n; _i += stride)
    {
      float min_tt = 100.0;
      bool flag = false;
      int mint_sphere_index = -1;
      for (int j = 0; j < nonzero_cnt[_i]; j++)
      {
        if (!cond_type1[prefix_sum[_i] + j])
          continue;
        if (step[prefix_sum[_i] + j] < min_tt)
        {
          min_tt = step[prefix_sum[_i] + j];
          flag = true;
          mint_sphere_index = j;
        }
      }
      min_t[_i] = min_tt;
      list_cond[_i] = flag;
      list_of_sphere[_i] = mint_sphere_index;
    }
  }
  void get_min_cuda_new(
      const int *nonzero_cnt,
      const int *prefix_sum,
      const int n,
      const float *step,
      const bool *cond_type1,
      bool *list_cond,
      float *min_t,
      int *list_of_sphere)
  {
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    get_min_kernel_new<<<blocks, threads>>>(nonzero_cnt, prefix_sum, n, step, cond_type1, list_cond, min_t, list_of_sphere);
  }
  __global__ void get_nice_kernel_new(
      const float *compressed_d1,
      const int *nonzero_cnt,
      const int *prefix_sum,
      const int n,
      const float *__restrict__ step,
      const bool *__restrict__ cond_type1,
      bool *__restrict__ list_cond,
      float *__restrict__ nice_t,
      int *__restrict__ list_of_sphere,
      const float epsilon)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (idx > n)
    {
      return;
    }
    for (int _i = idx; _i < n; _i += stride)
    {
      float min_tt = 100.0;
      bool flag = false;
      int mint_sphere_index = -1;

      for (int j = 0; j < nonzero_cnt[_i]; j++)
      {
        if (!cond_type1[prefix_sum[_i] + j])
          continue;
        if (step[prefix_sum[_i] + j] < min_tt)
        {
          min_tt = step[prefix_sum[_i] + j];
          flag = true;
          mint_sphere_index = j;
        }
      }
      float nice_dis1 = -100.0;
      float nice_tt = min_tt;
      int nice_sphere_index = mint_sphere_index;
      if (flag)
      {
        for (int j = 0; j < nonzero_cnt[_i]; j++)
        {
          if (!cond_type1[prefix_sum[_i] + j])
            continue;
          if (abs(min_tt - step[prefix_sum[_i] + j]) < epsilon)
          {
            float dis1 = step[prefix_sum[_i] + j] - compressed_d1[prefix_sum[_i] + j];
            float dis2 = min_tt - compressed_d1[prefix_sum[_i] + mint_sphere_index];
            if (dis1 > dis2)
            {
              if (dis1 > nice_dis1)
              {
                nice_dis1 = dis1;
                nice_sphere_index = j;
                nice_tt = step[prefix_sum[_i] + j];
              }
            }
          }
        }
      }

      nice_t[_i] = nice_tt;
      list_cond[_i] = flag;
      list_of_sphere[_i] = nice_sphere_index;
    }
  }
  void get_nice_cuda_new(
      const float *compressed_d1,
      const int *nonzero_cnt,
      const int *prefix_sum,
      const int n,
      const float *step,
      const bool *cond_type1,
      bool *list_cond,
      float *nice_t,
      int *list_of_sphere,
      const float epsilon)
  {
    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    get_nice_kernel_new<<<blocks, threads>>>(compressed_d1, nonzero_cnt, prefix_sum, n, step, cond_type1, list_cond, nice_t, list_of_sphere, epsilon);
  }
}