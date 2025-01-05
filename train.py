import os
import  time
import  argparse
from NI_model import NeuralImplicit
class MyTrain():
    def run(self,args):
        self.mesh_dir_path = args.mesh_dir_path
        self.safe_sphere_dir_path = args.safe_sphere_dir_path
        self.inner_sphere_dir_path = args.inner_sphere_dir_path
        self.mesh = NeuralImplicit(H=1, N=128, args = args)
        self.main()
    def main(self):
        print('-----start training-------')

        for mesh_path in os.listdir(self.mesh_dir_path):
            mname = os.path.splitext(mesh_path)[0]
            mfile = os.path.join(self.mesh_dir_path,mname+'.obj')
            # spfile = os.path.join(self.safe_sphere_dir_path, mname + '.txt')
            spfile = os.path.join(self.safe_sphere_dir_path,mname+'.h5')
            self.inner_sphere_file = os.path.join(self.inner_sphere_dir_path,mname+'.h5')
            time0 = time.time()

            self.mesh.encode(
                # sphere_num=self.sphere_num,
                sphere_file=spfile,
                inner_sphere_file=self.inner_sphere_file,
                # out_sphere_file=None,
                mesh_file=mfile,
                # sample_num=self.num_samples,
                # sdf_type=None
            )
            time1 = time.time()
            msg = '训练时间为：{:.2f}\n'.format(
                (time1 - time0))
            print(msg)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir_path', default='/home/wzj/PycharmProjects/sphere_resconstruct/thingi32_obj'
                                                   '', help='Log dir [default: data/obj/mesh]')
    parser.add_argument('--safe_sphere_dir_path', default='/home/wzj/PycharmProjects/sphere_resconstruct/thingi32_safe512/', help='sphere dir [default: data/sphere/mesh]')
    parser.add_argument('--inner_sphere_dir_path',default='/home/wzj/PycharmProjects/sphere_resconstruct/thingi32_inner512/',help='innersphere')
    parser.add_argument('--output_dir_path', default='results/thingi32_512', help='output_dir_path [default: results/trans]')
    parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 256]')
    parser.add_argument('--sample_num', type=int, default=200000, help='Sampled point num')
    parser.add_argument('--sphere_num', type=int, default=512, help='Max sphere num [default:256]')
    parser.add_argument('--TRANS_FLAG', action="store_true", help='Whether Trans is used here [if set, True; if not, False.]')
    parser.add_argument('--FRESH_SAMPLE_FLAG', default=True,action="store_true", help='Whether fresh sample is used here [if set, True; if not, False.]')
    parser.add_argument('--RESAMPLE_FLAG', default=True,action="store_true", help='Whether resample during training is used here [if set, True; if not, False.]')
    parser.add_argument('--resample_epoch', type=int, default=50, help='resample after some epoches [default:110]')

    args = parser.parse_args()
    return args
if __name__ == "__main__":
    # key = 'two'
    key ='default'
    args = get_args()
    print(args.TRANS_FLAG)
    '''args.mesh_dir_path = 'data/hardmesh'
    args.sphere_dir_path='data/sphere/hard_sphere'
    args.num_samples=200000
    args.sphere_num=256
    args.max_epoch=150
    args.output_dir_path='results/no_trans_hard'''
    os.system('cp {} {}'.format(os.path.join('oversdf_'+'default.py'), args.output_dir_path))
    train = MyTrain()
    train.run(args)