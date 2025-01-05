import torch
from torch_geometric.nn import knn
from utils.spherelatent import computePropotion, getDistance, EuclideanDistance
from torch import nn

def onenn(spheres, x):
    spheredis = EuclideanDistance(spheres, x).squeeze(dim=-1)  # bs*spherenum
    index = torch.argmin(spheredis, dim=1)
    col = torch.arange(0, x.shape[0])
    return col, index

def getSphereM(spheres, spherelatent, sphereR, k, bs=10000):
    sphere_num = spheres.shape[0]
    fdim = spherelatent.shape[-1]
    col, index = knn(spheres, spheres, k)
    spheres = torch.repeat_interleave(spheres[None, :, :], sphere_num, dim=0)[
        col, index].view(sphere_num, k, 3)
    sphereM = torch.repeat_interleave(spheres[None, :, :, :], bs, dim=0)
    spherelatent = torch.repeat_interleave(spherelatent[None, :, :], sphere_num, dim=0)[
        col, index].view(sphere_num, k, fdim)
    spherelatentM = torch.repeat_interleave(
        spherelatent[None, :, :, :], bs, dim=0)
    sphereR = torch.repeat_interleave(sphereR[None, :], sphere_num, dim=0)[
        col, index].view(sphere_num, k)
    sphereRM = torch.abs(torch.repeat_interleave(
        sphereR[None, :, :], bs, dim=0))
    return sphereM, spherelatentM, sphereRM


def getNearKspheresByone(x,spheres,sphereM,spherelatentM,sphereRM):
    bs = x.shape[0]
    spheres = spheres[:bs, :, :]
    sphereM = sphereM[:bs]
    spherelatentM = spherelatentM[:bs]
    sphereRM = sphereRM[:bs]
    x = x[:, None, :]
    col, index = onenn(spheres, x)
    spheres = sphereM[col, index]
    sphereR = sphereRM[col, index]
    spheredis = (sphereR**2)/(EuclideanDistance(spheres,x).squeeze(dim=-1))
    #spheredis = EuclideanDistance(spheres, x).squeeze(dim=-1)
    pro = nn.functional.normalize(spheredis,p=1).unsqueeze(dim=-1)
    sp = spherelatentM[col,index]
    return torch.sum(sp * pro, dim=1)

def get_near_k_spheres_local_features_by_pp(x, sphere_latent, vs_dist, vs_indexes):
    # print('sphere_latent', sphere_latent.shape, sphere_latent)
    x1 = torch.trunc((x/(2/63) + 31.5)).long()
    x1 = x1[:, 0] * 64 * 64 + x1[:, 1] * 64 + x1[:, 2]
    sphere_index = vs_indexes[x1,:]
    sphere_latent_features = sphere_latent[sphere_index]
    sphere_dist = vs_dist[x1, :]
    pro = nn.functional.normalize(sphere_dist, p=1).unsqueeze(dim=-1)
    return torch.sum(sphere_latent_features * pro, dim=1, dtype=torch.float)


def getNearKspheres(x, spheres, spherelatent, k):

    col, index = knn(spheres, x, k)
    spheredis = torch.unsqueeze(getDistance(
        x, spheres), dim=-1)[col, index].view(x.shape[0], k)
    pro = nn.functional.normalize(spheredis, p=1).unsqueeze(dim=-1)
    spherelatent = torch.repeat_interleave(
        spherelatent.unsqueeze(dim=0), x.shape[0], dim=0)
    sp = spherelatent[col, index].view(x.shape[0], k, -1)
    return sp, pro


def defined_distance(src, dst, sphere_r):

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
    dist_max = torch.max(dist, dim=-1)[0].view(N,1).repeat(1, M)
    dist = torch.exp(-(dist/dist_max))
    return dist




def query_k_point(k, xyz, new_xyz, sphere_r):

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

    # group_idx = torch.arange(N, dtype=torch.long).view(N, 1).repeat([1, k])
    # print(group_idx)
    dists = defined_distance(xyz, new_xyz, sphere_r)
    # group_idx[sqrdists > radius ** 2] = N
    # nearst_dis = torch.topk(dists, k=k, dim=-1, largest=False)[1]
    nearst_dis, nearst_idx = torch.topk(dists, k=k, dim=-1, largest=True)
    nearst_info = torch.cat([nearst_dis.unsqueeze(1), nearst_idx.unsqueeze(1)], dim=1)
    
    '''
    # vertify the index was right
    print('nearst_idx', nearst_idx)
    print(dists[group_idx, nearst_idx])
    print(nearst_idx.shape)
    print('inside_sphere_idx', inside_sphere_idx)
    print(dists_inside_sphere[group_idx, nearst_idx])
    print(inside_sphere_idx.shape)
    '''

    return nearst_info


def getNearKspheres_new(x, spheres, spherelatent, k):
    nearst_info = query_k_point(k=k, xyz=x, new_xyz=spheres[:, :3], sphere_r=spheres[:, 3])
    knn_sphere_index, spheredis = nearst_info[:,1,:].long(), nearst_info[:, 0, :]
    pro = nn.functional.normalize(spheredis, p=1).unsqueeze(dim=-1)
    spherelatent_p = spherelatent[knn_sphere_index]
    return spherelatent_p, pro
