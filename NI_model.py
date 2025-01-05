import logging
import os
import sys
import time

import h5py
import matplotlib
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
# from torchsummary import summary
import meshDataset as md
# from meshDataset import MeshDataset_define as md_define
import readTest
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import utils.geometry as geo
# from oversdf_trans_star import OverfitSDF as TransStar
# from oversdf_trans_star import OverfitSDF_no_pp as TransStarNoPP
# from oversdf_trans_star import buildGenMesh as buildGenMeshTransStar
#
# from oversdf_trans import OverfitSDF as Trans
# from oversdf_trans import OverfitSDFCat as TransCat
# from oversdf_trans import buildGenMesh as buildGenMeshTrans
#
# from oversdf_default import OverfitSDF as DefaultNetwork
from net_and_latent.oversdf_default_wylPE import OverfitSDF as DefaultNetwork
from oversdf_default import buildGenMesh as buildGenMeshDefault

# from oversdf_two import OverfitSDF as DefaultTwo
# from oversdf_two import buildGenMesh as buildGenMeshTwo

from tqdm import tqdm

class NeuralImplicit():
    def __init__(self, H=6, N=32, epochs=150, args=None):
        self.args = args
        # self.key = key
        self.N = N
        self.H = H
        self.epochs = self.args.max_epoch
        # self.epochs
        self.lr = 3e-3
        self.batch_size = 512
        self.trained = False
        self.sphere32 = None
        self.sphere_num = self.args.sphere_num
        self.sample_num = self.args.sample_num

    # Supported mesh file formats are .obj and .stl
    # Sampler selects oversample_ratio * num_sample points around the mesh, keeping only num_sample most
    # important points as determined by the importance metric
    def encode(self, 
                # sphere_num, 
                sphere_file,
                inner_sphere_file,
                mesh_file, 
                out_sphere_file=None, #这个选项和下面那个选项都是历史遗留问题，暂时不处理
                sdf_type=None,
                # sample_num=1000000, 
                verbose=True):
        # buildGenMeshTrans(None, None, None)

        
        self.sphere_file = sphere_file
        self.inner_sphere_file = inner_sphere_file
        self.mesh_file = mesh_file
        self.verbose = verbose
        self.mesh_name = self.mesh_file.split('/')[-1].split('.')[0]


        if (verbose and not logging.getLogger().hasHandlers()):
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
            logging.getLogger().setLevel(logging.INFO)

        print("显卡是否可用", torch.cuda.is_available())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("所用设备为: ", self.device)

        # get mesh_name
        self.mesh_name = os.path.splitext(os.path.split(mesh_file)[1])[0]

        #initialize

        self.set_dataset()
        # return
        self.set_model()
        self.set_optimizer()
        self.set_scheduler()

        self.losslist = []
        self.errorlist = []
        loss_func = nn.L1Loss(reduction='mean')
        
        print('========start train========')
        for e in tqdm(range(self.epochs)):
            if self.args.RESAMPLE_FLAG and e>1 and e % self.args.resample_epoch==0:
                print('========dataset resampling...========')
                self.dataset.resample()

                total_size = len(self.dataset)
                train_size = int(0.8 * total_size)
                test_size = total_size - train_size
                self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset,
                                                                                      [train_size, test_size])
                self.train_dataloader = DataLoader(
                    dataset=self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    # num_workers=8,
                    pin_memory=True)
                # return dataloader
                self.test_loader = self.train_dataloader = DataLoader(
                    dataset=self.test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    # num_workers=8,
                    pin_memory=True)
                # self.dataloader = DataLoader(
                #                     dataset=self.dataset,
                #                     batch_size=self.batch_size,
                #                     shuffle=True,
                #                     # num_workers=8,
                #                     pin_memory=True)
                print('========resample is done=========')
            # if e > 0 and e % 50 == 0:
            #     dataset.resample()
            #     dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size,
            #                             shuffle=True, num_workers=8, pin_memory=True)

            batch_idx = 0
            time0 = time.time()
            epoch_loss = 0
            epoch_error = 0
            self.model.train(True)
            # for batch_idx, (x_train, y_train) in enumerate(tqdm(dataloader)):
            for batch_idx, (x_train, y_train) in enumerate(self.train_dataloader):
                # x_train, y_train, nearst_info = x_train.to(self.device), y_train.to(self.device), nearst_info.to(self.device)
                x_train, y_train= x_train.to(self.device), y_train.to(self.device)
                #y_train = torch.cat((self.sphere32[:, 3], y_train), dim=0)
                self.optimizer.zero_grad()
                y_pred = self.model(x_train).squeeze(-1)
                loss = loss_func(y_pred , y_train)
                # error = self.my_loss(y_pred, y_train)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                # epoch_error += error.item()
                # if e == self.epochs - 1 or e == 0:
                #     with open('./spheretrainlog.txt', 'a') as f:
                #         f.write(msg)
                #         f.write('\n')
            with torch.no_grad():
                for batch_idx ,(x_test,y_test) in enumerate(self.test_loader):
                    x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                    y_pred = self.model(x_test).squeeze(-1)
                    error = loss_func(y_pred, y_test)
                    epoch_error += error.item()

            # if (early_stop and epoch_loss < early_stop):
            #     break
            time1 = time.time()
            epoch_loss = epoch_loss / (batch_idx + 1)
            epoch_error = epoch_error /(batch_idx + 1)
            self.scheduler.step(epoch_error)

            
            self.losslist.append(epoch_loss)
            self.errorlist.append(epoch_error)
            if e % 10==0:
                msg = '{}\tEpoch: {} sample: {} epoch_loss: {:.6f}\t epoch_error: {:.6f}\t elapse: {:.2f}'.format(
                    self.mesh_name,
                    e + 1,
                    self.sample_num,
                    epoch_loss,
                    epoch_error,
                    (time1 - time0))
                logging.info(msg)
            if  e%50==0:
                self.save_log()
            
                    
        self.save_log()

        # create a results folder for this mesh
        ####

    def save_log(self):
        outputdir = os.path.join(self.args.output_dir_path, self.mesh_name)
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        model_file = os.path.join(outputdir, str(self.sphere_num) + 'inout_' + self.mesh_name + '.pth')
        image_file = os.path.join(outputdir, str(self.sphere_num) + 'inout_' + self.mesh_name + '.png')
        genmesh_file = os.path.join(outputdir, str(self.sphere_num) + 'inout_' + self.mesh_name + '.obj')
        
        # if self.key =='default' or self.key=='two':
        #     pass
        # else:
        #     self.model.changeEval()
        torch.save(self.model, model_file)
        plotTrainResults(self.losslist, self.errorlist,image_file)
        # if self.key=='trans' or self.key=='trans_cat':
        #     buildGenMeshTrans(self.model, genmesh_file, self.sphere32)
        # elif self.key=='trans_star' or self.key=='trans_star_no_pp':
        #     buildGenMeshTransStar(self.model, genmesh_file, self.sphere32)
        # if self.key=='default':
        # buildGenMeshDefault(self.model, genmesh_file)
        # elif self.key=='two':
        #     buildGenMeshTwo(self.model, genmesh_file)
        #     # pass
        # else:
        #     raise NotImplementedError

            
    def set_dataset(self):

        def getMySpheres(sphere_file):
            f = h5py.File(sphere_file, 'r')
            f.keys()
            sphere32 = np.array(f['data'])
            f.close()
            return sphere32
        safe_sphere = getMySpheres(self.sphere_file)
        inner_sphere = getMySpheres(self.inner_sphere_file)[:,:4]
        max_safe_dis = np.max(safe_sphere[:,3]-abs(inner_sphere[:,3]))
        if self.args.FRESH_SAMPLE_FLAG:
            sample_method = {
                'weight': 100,
                'ratio': 0.1,
                'type': 'Importance'
            }
            sphereSampler = readTest.readSampler(
                mesh_file = self.mesh_file, \
                sphere_file= self.sphere_file, \
                # out_sphere_file = out_sphere_file, \
                sample_num = self.sample_num, \
                sphere_num = self.sphere_num, \
                sdf_type = "ni",
                sample_method = sample_method)

            # importance_sampler =  sphereSampler.importanceSampler()
            self.sphere32 = sphereSampler.getSpheres32()
            self.innersphere = inner_sphere
            # np.save('./sphere32.npy', self.sphere32)
            np.save('data1/npy/'+self.mesh_name+'_'+str(self.sphere_num)+'_spheres.npy', self.sphere32)
            # return
            # PRINT
            self.sphere32 = torch.from_numpy(self.sphere32).to(self.device)
            self.innersphere = torch.from_numpy(self.innersphere).to(self.device).to(torch.float32)
            self.innersphere[:,3] = abs(self.innersphere[:,3])
            # self.sphere32 = self.sphere32
            self.dataset = md.MeshDataset(sphereSampler, self.sample_num, self.verbose,self.sphere32, self.innersphere)
        elif self.args.RESAMPLE_FLAG:
            print('PLEASE use fresh sample flag so that it can resample')
            exit(-1)
        else:
            print('==========check '+self.mesh_name+'===========')
            self.sphere32 = torch.as_tensor(np.load('data/npy/'+self.mesh_name+'_'+str(self.sphere_num)+'_spheres.npy'), device = self.device)
            self.dataset = md.MeshDataset_define(file_path='data/npy/'+self.mesh_name+'.npy')
        # print(self.sphere32.shape)
        # print(dataset.shape)
        # PRINt
        total_size = len(self.dataset)
        train_size = int(0.8*total_size)
        test_size = total_size-train_size
        self.train_dataset , self.test_dataset = torch.utils.data.random_split(self.dataset,[train_size,test_size])
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # num_workers=8,
            pin_memory=True)
        # return dataloader
        self.test_loader = self.train_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # num_workers=8,
            pin_memory=True)

    def set_model(self):
        # if self.key=='trans':
        #     self.model = Trans(self.H, self.N, self.sphere32,TRANS_FLAG=self.args.TRANS_FLAG)
        # elif self.key=='trans_star':
        #     self.model = TransStar(self.H, self.N, self.sphere32)
        # elif self.key=='trans_cat':
        #     self.model = TransCat(self.H, self.N, self.sphere32, TRANS_FLAG=self.args.TRANS_FLAG)
        # elif self.key=='trans_star_no_pp':
        #     self.model = TransStarNoPP(self.H, self.N, self.sphere32)
        # elif self.key=='default':
        self.model = DefaultNetwork(self.H, self.N, self.sphere32,self.innersphere)
        # elif self.key=='two':
        #     self.model = DefaultTwo(self.H, self.N, self.sphere32)

        # else:
        #     raise NotImplementedError
        self.model.to(self.device)
        # summary(self.model, (3,))

    def set_optimizer(self):
        trans_params = []
        sdf_params = []
        sphere_params = []
        lw_params = []
        for pname, p in self.model.named_parameters():
            if pname.startswith('spherefeatures'):
                sphere_params += [p]
            elif pname.startswith('spherenet'):
                trans_params += [p]
            elif pname.startswith('lw'):
                lw_params+=[p]
            else:
                sdf_params+=[p]

        self.optimizer = optim.Adam(
            [
                {
                    "params": trans_params,
                    "lr": self.lr,
                },
                {
                    "params": sdf_params,
                    "lr": self.lr,
                },
                {
                    "params": sphere_params,
                    "lr": self.lr,
                },
                {
                    "params": lw_params,
                    "lr": self.lr,
                }
            ]
        )
    def set_scheduler(self):
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                   factor=0.7,
                                                   verbose=True,
                                                   min_lr=1e-6,
                                                   threshold=1e-4,
                                                   threshold_mode='abs',
                                                   patience=10,cooldown=5)

    def my_loss(self, ypred, ytrue):
        # l1 = 1+torch.log(1+50*torch.abs(ytrue-ypred))
        l2 = torch.abs(ytrue - ypred)
        loss = torch.mean(l2)
        return loss

def plotTrainResults(losslist, error,output, show=False, save=True):
    legend = ['Train','Test']
    loss_history = losslist
    plt.plot(loss_history)
    plt.plot(error)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    msg = 'final loss=' + str(losslist[-1]) + '\n'

    plt.text(len(loss_history) * 0.8, loss_history[-1], msg)
    plt.legend(legend, loc='upper left')

    if save:
        plt.savefig(output)
    if show:
        plt.show()

    plt.close()