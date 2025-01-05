import math

import h5py
import numpy as np

import utils.geometry as geo


# print(sampleVoxels[0:10])
# print(np.true_divide(sampleVoxels,4)[0:10])
# this class is responsible for read mesh, spheres, computing sdf
class readSampler():
    # def __init__(self,meshfile,spherefile,outspherefile,samplenum,spherenum=32,sdftype=None,res=128, M=None, W=None):
    def __init__(self, mesh_file=None, sphere_file=None, out_sphere_file=None, sample_num = None, sphere_num=32, sdf_type=None,res=128, sample_method=None):
    
        self.mesh_file = mesh_file
        self.sphere_file = sphere_file
        self.out_sphere_file = out_sphere_file
        if not self.mesh_file==None:
            if sdf_type==None:
                self.mesh = geo.Mesh_nglod(self.mesh_file,doNormalize=False)
                self.sdf = geo.SDF(self.mesh_file,self.mesh,signType='fast_winding_number')
                # self.sdf = geo.SDF(self.mesh_file, self.mesh, signType='instant')
            else:
                self.mesh = geo.Mesh(self.mesh_file)
                self.sdf = geo.SDF(self.mesh_file,self.mesh)
        self.sphere_num = sphere_num
        self.res = res
        if not sample_method == None:
            self.importanceSampler = geo.ImportanceSampler(self.mesh_file,
                mesh = self.mesh,
                M = int(sample_num /sample_method['ratio']),
                W = sample_method['weight'])

    '''def computeDisConstant(self,a,sphere32,spherenum=32):
        sphere32_xyz = sphere32[:, :3]
        sphere32_r = sphere32[:, 3]
        sphere32_r = sphere32_r[:, np.newaxis]
        temp_dis = sphere32_xyz - a
        d = np.sqrt(np.square(temp_dis[:, 0]) + np.square(temp_dis[:, 1]) + np.square(temp_dis[:, 2]))
        d = d[:, np.newaxis]
        t_dis = np.concatenate((sphere32_r, d), axis=1)

        #t_dis = t_dis[t_dis[:, 1].argsort()]
        #t_dis = t_dis[:spherenum, :]
        #r1 = t_dis[:, 0]
        r2 = t_dis[:, 1]
        #r3 = r1 / r2
        # gra_dis =  (r3 - np.min(r3)) / (np.max(r3) - np.min(r3))
        return np.concatenate((a, r2))

    def computeDis(self,a, sphere32,spherenum):
        sphere32_xyz = sphere32[:,:3]
        temp_dis = sphere32_xyz - a
        d = np.sqrt(np.square(temp_dis[:,0])+np.square(temp_dis[:,1])+np.square(temp_dis[:,2]))
        d = d[:,np.newaxis]
        temp_dis = np.concatenate((temp_dis,d),axis=1)
        temp_dis = temp_dis[temp_dis[:, 3].argsort()]
        temp_dis = temp_dis[:spherenum,:3]
        return np.concatenate((a,np.reshape(temp_dis,(-1,))), axis=0)'''

    def getSphereR(self,num):
        sphereR = self.sphere32[-1*self.sphere32[:,3].argsort()]
        return sphereR[:num,3]

    def computeDisandRadius(self,a, sphere32,spherenum):
        sphere32_xyz = sphere32[:,:3]
        sphere32_r = sphere32[:,3]
        sphere32_r = sphere32_r[:,np.newaxis]
        temp_dis = sphere32_xyz - a
        d = np.sqrt(np.square(temp_dis[:,0])+np.square(temp_dis[:,1])+np.square(temp_dis[:,2]))
        d = d[:,np.newaxis]
        t_dis = np.concatenate((sphere32_r,d),axis=1)

        #t_dis = t_dis[t_dis[:, 1].argsort()]
        t_dis = t_dis[:spherenum,:]
        r1 = t_dis[:,0]
        r2 = t_dis[:,1]+1e-6
        r3 = r1/r2
       # gra_dis =  (r3 - np.min(r3)) / (np.max(r3) - np.min(r3))
        return np.concatenate((a,r3))

    #
    # def computeDis(a, sphere32):
    #     sphere32_xyz = sphere32[:,:3]
    #     temp_dis = sphere32_xyz - a
    #     return np.concatenate((a,np.reshape(temp_dis,(-1,))), axis=0)

    '''def normalSamples(self,samples):
        _range = np.max(abs(samples))
        res = np.true_divide(samples,_range)
        return res'''

    def compute32(self,sampleVoxels,sdf_samples):

        samples = []
        if sdf_samples.shape[0]!=1:
            for i in range(sampleVoxels.shape[0]):

                samples.append(np.append(self.computeDis(sampleVoxels[i],self.sphere32,self.spherenum),sdf_samples[i]))
        else:
            for i in range(sampleVoxels.shape[0]):
                samples.append(self.computeDis(sampleVoxels[i], self.sphere32, self.spherenum))
        return np.array(samples)


    def getSDF(self,item):
        try:
            sdf_value = float(item)
            if sdf_value >= 0:
                return math.sqrt(sdf_value)
            else:
                return -1 * math.sqrt(math.fabs(sdf_value))
        except ValueError:
            print(item)

    '''
    def getSurfaceSamples(self,ratio,std,surfacenum):
        surfaceSampler = geo.PointSampler(self.mesh,ratio,std)
        surfacePoints = surfaceSampler._surfaceSamples(surfacenum)
        surfaceSdfs = self.sdf.query(surfacePoints)
        surfaceSamples = self.compute32(surfacePoints,surfaceSdfs)

        return surfaceSamples


    
    # def getVoxels(file):
    #     with open(file, 'r') as f:
    #         sdfstr = f.read()
    #     sdfstr = sdfstr.split(" ")
    #     sdfs = [getSDF(item) for item in sdfstr]
    #     voxels = np.zeros((128*128*128,5))
    #     for x in range(128):
    #         for y in range(128):
    #             for k in range(128):
    #                 index = x*128*128+y*128+k
    #                 voxels[index,0] = x
    #                 voxels[index,1] = y
    #                 voxels[index,2] = k
    #                 if sdfs[index]>=0:
    #                     voxels[index,3] = 1
    #                     voxels[index, 4] = sdfs[index]
    #                 else:
    #                     voxels[index,3] = -1
    #                     voxels[index,4] = -1 * sdfs[index]
    #
    #
    #
    #     return voxels[np.lexsort(voxels.T)]
    
    def getVoxels(self,res=128):


        cuber = geo.CubeMarcher()
        grid = cuber.createGrid(res)
        gridsdfs = self.sdf.query(grid)
        voxels = np.concatenate((grid, gridsdfs), axis=1)
        signv = voxels[:, 3] / np.fabs(voxels[:, 3])
        signv = signv[:, np.newaxis]
        voxels[:, 3] = np.fabs(voxels[:, 3])
        voxels = np.concatenate((voxels, signv), axis=1)
        voxels = voxels[voxels[:, 3].argsort()]
        #voxels = voxels.astype(np.float32)
        return voxels
    '''



    def getSpheres(self):

        with open(self.sphere_file, 'r') as f:
            tspheres = f.readlines()
        spherenum = int(tspheres[0])
        if not self.out_sphere_file==None:
            sphere32 = np.zeros([spherenum*2, 4], dtype=np.float32)
            with open(self.sphere_file,'r') as f:
                tspheres = f.readlines()
                for i in range(1,spherenum+1):
                    t = tspheres[i].split(' ')
                    sphere32[i-1] = [float(item) for item in t][:4]
                # sphere32[:,:3] = (sphere32[:,:3]-64)/64
                # sphere32[:,3] = sphere32[:,3]/64
                self.sphere32 = sphere32
            with open(self.out_sphere_file,'r') as f:
                tspheres = f.readlines()
                for i in range(1,spherenum+1):
                    t = tspheres[i].split(' ')
                    sphere32[i+spherenum-1] = [float(item) for item in t][:4]
                sphere32[:,:3] = (sphere32[:,:3]-64)/64
                sphere32[:,3] = sphere32[:,3]/64
                self.sphere32 = sphere32
        else:
            sphere32 = np.zeros([spherenum , 4], dtype=np.float32)
            with open(self.sphere_file,'r') as f:
                tspheres = f.readlines()
                for i in range(1,spherenum+1):
                    t = tspheres[i].split(' ')
                    sphere32[i-1] = [float(item) for item in t][:4]
                sphere32[:,:3] = (sphere32[:,:3]-64)/64
                sphere32[:,3] = sphere32[:,3]/64
                # self.sphere32 = sphere32

        return sphere32

    def getMySpheres(self):
        f = h5py.File(self.sphere_file,'r')
        f.keys()
        sphere32 = np.array(f['data'])
        f.close()
        return sphere32
    def getSpheres32(self):
        self.sphere32 = self.getMySpheres()
        # print('self.shphere32', self.sphere32)
        # sphere32R = self.sdf.query(self.sphere32[:, :3].copy())
        # # print(self.sphere32[:, :3].shape)
        # # print(sphere32R.shape)
        # self.sphere32 = np.concatenate((self.sphere32[:, :3], sphere32R), axis=1)
        # self.sphere32 = self.sphere32.astype(np.float32)
        return self.sphere32



