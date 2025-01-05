import time

from torch import nn
import torch
from torch_geometric.nn import knn
from utils.EmbedderHelper import get_embedder, get_embedder_sp
# from spherelatent import computePropotion, getDistance, EuclideanDistance
from PCT_model import Pctn, Pct
from tqdm import tqdm
import utils.geometry as geo
import numpy as np
from utils.distance_utils import *
from torch import nn
import kaolin.render.spc as spc_render

emb, out_dim = get_embedder(6)


class OverfitSDF(nn.Module):
    def __init__(self, H, N, sphere32,innersphere):
        super().__init__()
        assert (N > 0)
        assert (H > 0)
        self.sphere32 = sphere32
        self.innersphere = innersphere
        self.sphere32_num = self.sphere32.shape[0]
        # self.spherenet = nn.Linear(4,29)
        # self.spherenet = nn.Linear(4,29)

        fts = torch.zeros(self.sphere32_num, 32)
        self.spherefeatures = nn.Parameter(torch.rand_like(fts))  # 这里乘0.01是nglod这么做的，原因未知，后面问

        net = [nn.Linear(64, N), nn.LeakyReLU(0.1)]
        self.training = True
        for _ in range(H - 1):
            net += [nn.Linear(N, N), nn.LeakyReLU(0.1)]
        net += [nn.Linear(N, 1)]
        self.model = nn.Sequential(*net)

    def forward(self, x):
        return self.sdf(x)
        # # spherelatent = self.spherenet(self.sphere32)
        # sphere_id = self.get_every_safe_sphere(x, self.sphere32)
        # theta, phi, dis = self.get_theta_and_phi_and_dis(x, sphere_id, self.sphere32)
        # feature = self.get_ultimate_feature(x, theta, phi, dis, self.spherefeatures, sphere_id)
        # # co_feature = compute_co_feature(x, self.sphere32[:,:3], spherelatent)
        # # pos_feature = torch.sin(10*self.poslatent(x))
        # # pos_feature = emb(x)[:, :29]
        # # print(pos_feature.shape)
        # # PRINT
        # # overall_feature = co_feature + pos_feature
        # # x_train = overall_feature
        # # x_train = torch.cat((x, overall_feature),dim=1)
        # '''if self.training == True:
        #     x1 = torch.cat((self.sphere32[:,:3],spherelatent),dim=1)
        #     x = torch.cat((x1,x),dim=0)'''
        # x = self.model(feature)
        # output = torch.tanh(x)
        # return output


    def sdf(self,x):
         if x.shape[1] == 3:
            sphere_id = self.get_every_safe_sphere(x, self.sphere32)
            # theta, phi, dis = self.get_theta_and_phi_and_dis(x, sphere_id, self.sphere32)
            new_vec = self.get_dir_vec_and_dis(x,sphere_id,self.sphere32)
            aux_feature = emb(new_vec)[:,:32]
            sphere_latent = self.get_point_sphere_latent(self.spherefeatures,sphere_id)
            # feature = torch.cat([sphere_latent,aux_feature],dim=1)
            # feature = sphere_latent+aux_feature
            # pos_feature = emb(x)[:, :32]
            # feature = self.get_ultimate_feature(x, theta, phi, dis, self.spherefeatures, sphere_id)
            # feature = feature[:,6:]
            # feature = pos_feature + sphere_latent
            feature = torch.cat([sphere_latent, aux_feature], dim=1)
            # time111 = time.time()
            x = self.model(feature)
            # print(time.time()-time111)
            output = torch.tanh(x)
            return output
         else:
             sphere_latent  = self.get_point_sphere_latent(self.spherefeatures,x[:,3])
             # pos_feature = emb(x)[:, :32]
             new_vec = self.get_dir_vec_and_dis(x[:,:3], x[:,3], self.sphere32)
             aux_feature = emb(new_vec)[:, :32]
             feature = torch.cat([sphere_latent, aux_feature], dim=1)
             # theta, phi, dis = self.get_theta_and_phi_and_dis(x[:,:3], x[:,3], self.sphere32)
             # feature = self.get_ultimate_feature(x[:,:3], theta, phi, dis, self.spherefeatures, x[:,3])
             # feature = torch.cat([feature[:,0:3],feature[:,6:]],dim=1)
             x = self.model(feature)
             output = torch.tanh(x)
             return output
    def sdf1(self,x,sphere_index):

        sphere_latent = self.get_point_sphere_latent(self.spherefeatures,sphere_index)
        new_vec = self.get_dir_vec_and_dis(x[:, :3], sphere_index, self.sphere32)
        # time111 = time.time()
        aux_feature = emb(new_vec)[:, :32]
        # print(time.time() - time111)
        feature = torch.cat([sphere_latent, aux_feature], dim=1)

        x = self.model(feature)

        output = torch.tanh(x)
        return output
    def get_every_safe_sphere(self, x, safe_spheres):
        # 找出在哪个球里，并返回最大半径球编号，cuda写， returm x.shape[0]*1
        #改成最小球了，但是函数名称没改

        return spc_render.point_its_max_sphere(safe_spheres, x)

    def get_theta_and_phi_and_dis(self, x, sphere_id, sphere_xyzr):
        its_sphere = sphere_xyzr[sphere_id.long()].squeeze(dim=1)
        d = torch.sqrt((x[:, 0] - its_sphere[:, 0]) ** 2 + (x[:, 1] - its_sphere[:, 1]) ** 2 + (
                x[:, 2] - its_sphere[:, 2]) ** 2)
        dis_divide = d / its_sphere[:, 3]
        # x_theta = torch.acos((x[:, 2] - its_sphere[:, 2]) / d)
        x_theta = torch.atan2((x[:, 2] - its_sphere[:, 2]), torch.sqrt(
            (x[:, 0] - its_sphere[:, 0]) ** 2 + (x[:, 1] - its_sphere[:, 1]) ** 2))  # 纬度角 负二分之一pi到二分之一pi
        x_phi = torch.atan2(x[:, 1] - its_sphere[:, 1], x[:, 0] - its_sphere[:, 0])  # 负pi到pi
        return x_theta, x_phi, dis_divide
    def get_dir_vec_and_dis(self,x,sphere_id,sphere_xyzr):
        its_sphere = sphere_xyzr[sphere_id.long()].squeeze(dim=1)
        vec = x-its_sphere[:,:3]
        d = torch.sqrt((x[:, 0] - its_sphere[:, 0]) ** 2 + (x[:, 1] - its_sphere[:, 1]) ** 2 + (
                x[:, 2] - its_sphere[:, 2]) ** 2)
        dis_divide = d / its_sphere[:, 3]
        dis_divide = torch.unsqueeze(dis_divide,dim=1)
        new_vec = torch.cat([vec,dis_divide],dim=1)
        return  new_vec
    def get_ultimate_feature(self, x, theta, phi, dis_divide, sphere_latent, sphere_id):
        normalized_theta = torch.unsqueeze(torch.sin(theta), dim=1)
        normalized_phi = torch.unsqueeze(torch.cos(phi), dim=1)
        dis_divide = torch.unsqueeze(dis_divide, dim=1)
        point_sphere = sphere_latent[torch.squeeze(sphere_id).long()]
        if point_sphere.dim()==1:
            point_sphere = torch.unsqueeze(point_sphere,dim=0)

        ultimate_feature = torch.cat([x, normalized_theta, normalized_phi, dis_divide, point_sphere], dim=1)
        return ultimate_feature
    def get_point_sphere_latent(self,sphere_latent,sphere_id):
        point_sphere = sphere_latent[torch.squeeze(sphere_id).long()]
        if point_sphere.dim() == 1:
            point_sphere = torch.unsqueeze(point_sphere, dim=0)
        return point_sphere

def buildGenMesh(sdfModel, output, res=256):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cubeMarcher = geo.CubeMarcher()
    rGrid = torch.from_numpy(cubeMarcher.createGrid(res).astype(np.float32)).cuda()
    sdfModel.eval()
    rGrid_tmp = torch.split(rGrid, 512, dim=0)
    # rGrid_pct = torch.split(rGrid_pct, 512_64_60, dim=0)
    # sdfModel.eval()
    with torch.no_grad():
        S = sdfModel(rGrid_tmp[0])
    for i in tqdm(range(len(rGrid_tmp) - 1)):
        with torch.no_grad():
            temp = sdfModel(rGrid_tmp[i + 1])
        S = torch.cat((S, temp), dim=0)
    # with torch.no_grad():
    #     S = sdfModel(rGrid).cpu()
    cubeMarcher.march(rGrid.cpu().numpy(), S.cpu().numpy())
    marchedMesh = cubeMarcher.getMesh()
    marchedMesh.save(output)

# def compute_co_feature(x, sphere32_xyz, spherelatent):
#     spheredis = get_distance(x, sphere32_xyz)
#     pro = spheredis / torch.unsqueeze(torch.sum(spheredis, dim=1), dim=1)
#     # pro = nn.functional.normalize(spheredis, p=1).unsqueeze(dim=-1)
#     lt = spherelatent.t().mm(pro.t()).t()#矩阵乘法与转置
#     return lt

# def get_distance(a, b):
#     sq_a = a**2
#     sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)
#     sq_b = b**2
#     sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)
#     bt = b.t()
#     return torch.sqrt(torch.abs(sum_sq_a+sum_sq_b-2*a.mm(bt)))
def buildGenMesh_with_sphere(sdfModel, output, res=256):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cubeMarcher = geo.CubeMarcher()
    rGrid = torch.from_numpy(cubeMarcher.createGrid(res).astype(np.float32)).cuda()
    sdfModel.eval()
    rGrid_tmp = torch.split(rGrid, 512, dim=0)
    # rGrid_pct = torch.split(rGrid_pct, 512_64_60, dim=0)
    # sdfModel.eval()
    with torch.no_grad():
        S = sdfModel(rGrid_tmp[0])
    for i in tqdm(range(len(rGrid_tmp) - 1)):
        with torch.no_grad():
            temp = sdfModel(rGrid_tmp[i + 1])
        S = torch.cat((S, temp), dim=0)
    # with torch.no_grad():
    #     S = sdfModel(rGrid).cpu()
    cubeMarcher.march(rGrid.cpu().numpy(), S.cpu().numpy())
    marchedMesh = cubeMarcher.getMesh()
    marchedMesh.save(output)