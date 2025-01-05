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


import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math


from PIL import Image
import numpy as np
import torch

from scipy.spatial.transform import Rotation as R
import pyexr

from render_gt_utils.renderer import Renderer
# from utils.lib.tracer import SphereTracer
# from SSTracer import SSTracerr

from tracer.gt_tracer import SphereTracer
# from ray_type_tracer import SSTracer
from utils.lib.options import parse_options
from utils.lib.geoutils import sample_fib_sphere
from time import time

def write_exr(path, data):
    pyexr.write(path, data,
                channel_names={'normal': ['X', 'Y', 'Z'],
                               'x': ['X', 'Y', 'Z'],
                               'view': ['X', 'Y', 'Z']},
                precision=pyexr.HALF)


if __name__ == '__main__':

    # Parse
    parser = parse_options(return_parser=True)
    app_group = parser.add_argument_group('app')
    app_group.add_argument('--img-dir', type=str, default='_results/render_app/imgs',
                           help='Directory to output the rendered images')
    app_group.add_argument('--render-2d', action='store_true',
                           help='Render in 2D instead of 3D')
    app_group.add_argument('--exr', action='store_true',
                           help='Write to EXR')
    app_group.add_argument('--r360', action='store_true',
                           help='Render a sequence of spinning images.')
    app_group.add_argument('--rsphere', action='store_true',
                           help='Render around a sphere.')
    app_group.add_argument('--nb-poses', type=int, default=64,
                           help='Number of poses to render for sphere rendering.')
    app_group.add_argument('--cam-radius', type=float, default=4.0,
                           help='Camera radius to use for sphere rendering.')
    app_group.add_argument('--disable-aa', action='store_true',
                           help='Disable anti aliasing.')
    app_group.add_argument('--export', type=str, default=None,
                           help='Export model to C++ compatible format.')
    app_group.add_argument('--rotate', type=float, default=None,
                           help='Rotation in degrees.')
    app_group.add_argument('--depth', type=float, default=0.0,
                           help='Depth of 2D slice.')
    args = parser.parse_args()

    # Pick device
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    #get mesh file
    # meshfile = '/media/cscvlab/data/project/Yuanzhan/pctTemp/temp/472139.stl'
    # spsampler = readTest.readSampler(meshfile, None,NotImplementedError,int(1000000*0.001), 256)
    # net_o = spsampler.sdf.query
    # def net(x):
    #     re = torch.from_numpy(net_o(x.cpu().numpy()).astype(np.float32)).cuda()
    #     return re
    model_dir = 'aaa_thingi32_new_results/128_64_36/1_128'
    mesh_dir = 'aaa_my3/mesh1'
    key = 'star'
    # key=None
    for m in sorted(os.listdir(mesh_dir)):
        import readTest
        mesh_file = os.path.join(mesh_dir,m)
        sphere_sampler = readTest.readSampler(
            mesh_file=mesh_file,
            sdf_type='ni'
        )

        modelfile = "/home/wzj/PycharmProjects/sphere_resconstruct/results/thingi32_64_6_64/441708/64inout_441708.pth"
        net = torch.load(modelfile)
        net.to(device)
        net.eval()
        # if key == 'star':
        #     net.getPreM()


        print("Total number of network parameters: {}".format(sum(p.numel() for p in net.parameters())))


        render_dir = '/home/wzj/PycharmProjects/sphere_resconstruct/aaa_my3/render'
        # Make output directory
        # outputdir = os.path.join(model_dir,m)
        outputdir = os.path.join(render_dir, m)


        name = m+'_image_'+str(args.rotate)
        ins_dir = os.path.join(outputdir, name)
        if not os.path.exists(ins_dir):
            os.makedirs(ins_dir)

        for t in ['normal', 'rgb', 'exr']:
            _dir = os.path.join(ins_dir, t)
            if not os.path.exists(_dir):
                os.makedirs(_dir)

        tracer = SphereTracer(args)
        renderer = Renderer(tracer, args=args, device=device, sdf_net=net)
        if args.rotate is not None:
            rad = np.radians(args.rotate)
            pre_model_matrix = torch.FloatTensor(R.from_rotvec(rad * np.array([1, 0, 0])).as_matrix())
            model_matrix = torch.FloatTensor(R.from_rotvec(rad * np.array([1, 0, 0])).as_matrix())
        else: # test_r = torch.cat((test_r.cuda(), step), 1)
                # test_x = torch.cat((test_x.cuda(), torch.unsqueeze(pos[:, 0], dim=1)), 1)
                # test_y = torch.cat((test_y.cuda(), torch.unsqueeze(pos[:, 1], dim=1)), 1)
                # test_z = torch.cat((test_z.cuda(), torch.unsqueeze(pos[:, 2], dim=1)), 1)
            pre_model_matrix = torch.eye(3)
            model_matrix = torch.eye(3)

        if args.r360:
            time0 = time()
            for angle in np.arange(0, 360, 90):
                print('---------------------'+str(angle)+'------------------')
                rad = np.radians(angle)
                model_matrix = torch.FloatTensor(R.from_rotvec(rad * np.array([0, 1, 0])).as_matrix())
                model_matrix = torch.mm(model_matrix,pre_model_matrix)

                #'--camera-origin', type=float, nargs=3, default=[-2.8, 2.8, -2.8]
                #'--camera-lookat', type=float, nargs=3, default=[0, 0, 0]
                # --camera-fov', type=float, default=30
                # --aa True
                # model_matrix

                out = renderer.shade_images(net=net,
                                            sampler=sphere_sampler,
                                            f=args.camera_origin,
                                            t=args.camera_lookat,
                                            fov=args.camera_fov,
                                            aa=not args.disable_aa,
                                            mm=model_matrix)
            #print('elapsed time: {}'.format(time()-time0))

                data = out.float().numpy().exrdict()

                idx = int(math.floor(100 * angle))

                if args.exr:
                    write_exr('{}/exr/{:06d}.exr'.format(ins_dir, idx), data)

                img_out = out.image().byte().numpy()
                Image.fromarray(img_out.rgb).save('{}/rgb/{:06d}.png'.format(ins_dir, idx), mode='RGB')
                Image.fromarray(img_out.normal).save('{}/normal/{:06d}.png'.format(ins_dir, idx), mode='RGB')

        elif args.rsphere:
            views = sample_fib_sphere(args.nb_poses)
            cam_origins = args.cam_radius * views
            for p, cam_origin in enumerate(cam_origins):
                out = renderer.shade_images(f=cam_origin,
                                            t=args.camera_lookat,
                                            fov=args.camera_fov,
                                            aa=not args.disable_aa,
                                            mm=model_matrix)

                data = out.float().numpy().exrdict()

                if args.exr:
                    write_exr('{}/exr/{:06d}.exr'.format(ins_dir, p), data)

                img_out = out.image().byte().numpy()
                Image.fromarray(img_out.rgb).save('{}/rgb/{:06d}.png'.format(ins_dir, p), mode='RGB')
                Image.fromarray(img_out.normal).save('{}/normal/{:06d}.png'.format(ins_dir, p), mode='RGB')

        else:

            out = renderer.shade_images(net=net,
                                        f=args.camera_origin,
                                        t=args.camera_lookat,
                                        fov=args.camera_fov,
                                        aa=not args.disable_aa,
                                        mm=model_matrix)

            data = out.float().numpy().exrdict()

            if args.render_2d:
                depth = args.depth
                data['sdf_slice'] = renderer.sdf_slice(depth=depth)
                data['rgb_slice'] = renderer.rgb_slice(depth=depth)
                data['normal_slice'] = renderer.normal_slice(depth=depth)

            if args.exr:
                write_exr(f'{ins_dir}/out.exr', data)

            img_out = out.image().byte().numpy()

            Image.fromarray(img_out.rgb).save('{}/{}_rgb.png'.format(ins_dir, name), mode='RGB')
            Image.fromarray(img_out.depth).save('{}/{}_depth.png'.format(ins_dir, name), mode='RGB')
            Image.fromarray(img_out.normal).save('{}/{}_normal.png'.format(ins_dir, name), mode='RGB')
            Image.fromarray(img_out.hit).save('{}/{}_hit.png'.format(ins_dir, name), mode='L')
