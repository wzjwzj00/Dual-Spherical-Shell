import h5py
import  torch
from scipy.spatial import   cKDTree as KDTree
from tracer.test_tracer_test_choose_far_sphere_0327 import SSTracer
from utils.lib.options import parse_options
from kaolin.ops.mesh import sample_points
import  readTest
from utils.lib.loadObjNglod import load_obj
from meshDataset import *
import meshDataset as md
def do_compute_cd(mesh_path,model_path,args):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    net = torch.load(model_path)
    net.to(device)
    my_tracer = SSTracer(args)
    sphere_sampler = readTest.readSampler(
        mesh_file=mesh_path,
        sdf_type='ni'
    )
    pred_pts = my_tracer.sample_surface(131072,net,sphere_sampler)
    pred_pts = pred_pts.cpu().numpy()
    V ,F = load_obj(mesh_path)
    V = V.unsqueeze(0)
    gt_pts , _= sample_points(V,F,131072)
    gt_pts = gt_pts.squeeze()
    gt_pts = gt_pts.cpu().numpy()
    pred_pts_kd_tree = KDTree(pred_pts)
    one_dis ,one_vertex_ids = pred_pts_kd_tree.query(gt_pts)
    gt_to_temp = np.square(one_dis)
    gt_to_pre_chamfer = np.mean(gt_to_temp)
    np.savetxt("mypoints.txt",pred_pts)
    np.savetxt("gtpoints.txt",gt_pts)
    gt_pts_kd_tree = KDTree(gt_pts)
    two_dis , two_vertex_ids = gt_pts_kd_tree.query(pred_pts)
    pre_to_gt_temp = np.square(two_dis)
    pre_to_gt_chamfer = np.mean(pre_to_gt_temp)
    cd = gt_to_pre_chamfer + pre_to_gt_chamfer
    print(cd)
    return cd

def computeIOU(gvoxels,tvoxels):
    t1 = gvoxels|tvoxels
    t2 = np.sum(t1)#union
    t3 = np.sum(gvoxels & tvoxels)#intersect
    return t3/t2


def compute_iou1(dist_gt, dist_pr):
    """Intersection over Union.

    Args:
        dist_gt (torch.Tensor): Groundtruth signed distances
        dist_pr (torch.Tensor): Predicted signed distances
    """

    occ_gt = (dist_gt < 0).byte()
    occ_pr = (dist_pr < 0).byte()

    area_union = torch.sum((occ_gt | occ_pr).float())
    area_intersect = torch.sum((occ_gt & occ_pr).float())

    iou = area_intersect / area_union
    return 100. * iou
def compute_iou(mesh_path,model_path,sphere_file,inner_sphere_file):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    sample_method = {
        'weight': 100,
        'ratio': 0.1,
        'type': 'Importance'
    }

    def getMySpheres(sphere_file):
        f = h5py.File(sphere_file, 'r')
        f.keys()
        sphere32 = np.array(f['data'])
        f.close()
        return sphere32

    safe_sphere = getMySpheres(sphere_file)
    inner_sphere = getMySpheres(inner_sphere_file)[:, :4]
    inner_sphere = torch.from_numpy(inner_sphere).to(device).to(torch.float32)
    inner_sphere[:,3] = abs(inner_sphere[:,3])
    sphereSampler = readTest.readSampler(mesh_path, sphere_file= sphere_file,sample_num= 1000000,sample_method=sample_method)
    sphere = sphereSampler.getSpheres32()
    sphere = torch.from_numpy(sphere).to(device)
    dataset = md.MeshDataset(sphereSampler,sphere32=sphere,inner_sphere=inner_sphere,samplenum=1000000)
    sdf = sphereSampler.sdf

    net = torch.load(model_path)
    net.to(device)
    gt_pts = dataset.trainData[:,:3].to(device)
    idx = dataset.trainData[:,3].long().to(device)

    with torch.no_grad():
        pred_sdf  = net.sdf1(gt_pts,idx)
    gt_pts = gt_pts.detach().cpu().numpy()
    pred_sdf = pred_sdf.detach().cpu().numpy()
    gts  =  dataset.trainData[:,4]
    pred_sdf = np.where(pred_sdf<0,1,0)
    gts = gts.reshape(-1,1)
    gts = np.where(gts<0,1,0)
    iou = computeIOU(pred_sdf,gts)
    a = 0





if __name__ == "__main__":
    parser = parse_options(return_parser=True)
    args = parser.parse_args()
    mesh_path = "/home/wzj/PycharmProjects/sphere_resconstruct/thingi_32_09/398259.obj"
    model_path = "/home/wzj/PycharmProjects/sphere_resconstruct/aaa_new_pe_results/39_32_1128/398259/512inout_398259.pth"
    mesh_dir = "/home/wzj/PycharmProjects/sphere_resconstruct/thingi_32_09"
    model_dir = "/home/wzj/PycharmProjects/sphere_resconstruct/aaa_thingi32_new_results/128_64_36/1_128"
    # model_dir = "/home/wzj/PycharmProjects/sphere_resconstruct/aaa_thingi32_new_results/512_64_36/4_128"
    # sphere_file = "/home/wzj/PycharmProjects/sphere_resconstruct/thingi32_safe512/441708.h5"
    # inner_sphere_file = "/home/wzj/PycharmProjects/sphere_resconstruct/thingi32_inner512/441708.h5"
    cd = 0
    mnum = 0
    # compute_iou(mesh_path,model_path,sphere_file,inner_sphere_file)
    # do_compute_cd(mesh_path,model_path,args)
    for mesh_path in sorted(os.listdir(mesh_dir)):


        mname = os.path.splitext(mesh_path)[0]
        print("mname = {}".format(mname))
        mfile = os.path.join(mesh_dir, mname + '.obj')
        model_file_1 = os.path.join(model_dir,mname)
        model_file_2 = os.path.join(model_file_1,"128inout_"+mname+".pth")
        cd_curr = do_compute_cd(mfile, model_file_2, args)
        print("{} cd:{}".format(mname,cd_curr))
        cd = cd + cd_curr
        mnum = mnum + 1
    print(cd/mnum)
