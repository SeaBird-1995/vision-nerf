'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-05-02 09:49:55
Email: haimingzhang@link.cuhk.edu.cn
Description: Extract the point cloud from the NeRF.
'''

import os
import numpy as np
import torch
from tqdm import tqdm
from einops import repeat, rearrange
from collections import OrderedDict
import imageio
from models.model import VisionNerfModel
from models.projection import Projector
from eval import config_parser, SRNRenderDataset
from models.sample_ray import RaySamplerSingleImage


def infer_nerf(batch_dict, model, projector, featmaps):
    pts = batch_dict['xyz_batch']
    dir = batch_dict['dir_batch']
    batch, N_rays, N_samples = pts.shape[:3]

    ## Get the feature maps and the points coordinate in the camera coordinate system
    rgb_feat, xyz_c = projector.compute_pixel(pts, batch_dict['src_rgbs'],
                                              batch_dict['src_intrinsics'],
                                              batch_dict['src_c2w_mats'],
                                              featmaps=featmaps)  # [batch, N_rays, N_samples, N_views, x]
    xyz_c = xyz_c[:, 0, ..., :3] # HACK only use the first view
    rgb_feat = rgb_feat.squeeze(3) # HACK consider only one camera now so remove the axis

    dir_c = projector.compute_directions(dir, batch_dict['src_c2w_mats'])
    dir_c = dir_c[:, 0, ..., :3] # HACK only use the first view

    feat = torch.cat([rgb_feat, dir_c], -1)
    raw_output = model.net_fine(xyz_c.flatten(0, 1), feat.flatten(0, 1))   # [batch*N_rays*N_samples, 4]
    raw_output = raw_output.reshape([batch, N_rays, N_samples, 4])
    return raw_output


def get_sigma_from_nerf(args, ckpt_path, N=128):
    chunk = 1024 * 32

    # Create VisionNeRF model
    model = VisionNerfModel(args, False, False)
    model.switch_to_eval()

    # Define the projector
    projector = Projector(args)

    size = 1.2      # default 1.2
    xmin, xmax = -size, size  # left/right range
    ymin, ymax = -size, size  # forward/backward range
    zmin, zmax = -size, size  # up/down range

    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()

    print('Predicting occupancy ...')
    with torch.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            batch, N_rays, N_samples = 1, 1, xyz_.shape[0]
            xyz_batch = xyz_[i:i + chunk]
            dir_batch = dir_[i:i + chunk]

            xyz_batch = rearrange(xyz_batch, '(b nr ns) Dim3 -> b nr ns Dim3', b=batch, nr=N_rays, ns=N_samples)
            dir_batch = repeat(dir_batch, '(b nr ns) Dim3 -> b nr ns Dim3', b=batch, nr=N_rays, ns=N_samples)

            batch_dict['xyz_batch'] = xyz_batch
            batch_dict['dir_batch'] = dir_batch

            out_chunks += [infer_nerf(batch_dict, model, projector, featmaps)]
        rgbsigma = torch.cat(out_chunks, 0)

    sigma = rgbsigma[:, -1].cpu().numpy()
    sigma = np.maximum(sigma, 0)
    sigma = sigma.reshape(N, N, N)
    return sigma


def get_point_cloud_from_nerf(batch_dict,
                              model,
                              projector,
                              featmaps, 
                              chunk_size):
    N_points = batch_dict['xyz_batch'].shape[0]

    all_chunks = []
    for i in range(0, N_points, chunk_size):
        chunk = OrderedDict()
        for k in batch_dict:
            if k in ['intrinsics', 'c2w_mat', 'depth_range',
                     'src_rgbs', 'src_intrinsics', 'src_c2w_mats']:
                chunk[k] = batch_dict[k]
            elif batch_dict[k] is not None:
                chunk[k] = batch_dict[k][None, i:i+chunk_size]
            else:
                chunk[k] = None
            
        ret = infer_nerf(chunk, model, projector, featmaps)  # to (b, N, 1, 4)
        all_chunks.append(ret)
    
    rgbsigma = torch.cat(all_chunks, dim=1)
    rgbsigma = rgbsigma.squeeze()  # to (N, 4)
    sigma = rgbsigma[:, -1].cpu().numpy()
    sigma = np.maximum(sigma, 0)
    return sigma


def main(args, N=128):
    device = "cuda"
    print(f"checkpoints reload from {args.ckptdir}")

    dataset = SRNRenderDataset(args)

    # Create VisionNeRF model
    model = VisionNerfModel(args, False, False)
    # create projector
    projector = Projector(device=device)
    model.switch_to_eval()

    if args.use_data_index:
        data_index = args.data_indices
    else:
        # how many object do you want to render
        data_index = np.arange(args.data_range[0], args.data_range[1])

    ## Define the sampled points in the space
    size = 1.2      # default 1.2
    xmin, xmax = -size, size  # left/right range
    ymin, ymax = -size, size  # forward/backward range
    zmin, zmax = -size, size  # up/down range

    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3))
    dir_ = torch.zeros_like(xyz_)

    xyz_ = xyz_[:, None, :].to(device)  # to (N, 1, 3)
    dir_ = dir_[:, None, :].to(device)  # to (N, 1, 3)

    for d_idx in tqdm(data_index):
        out_folder = os.path.join(args.outdir, args.expname, f'{d_idx:06d}')
        os.makedirs(out_folder, exist_ok=True)
        
        sample = dataset[d_idx]
        pose_index = args.pose_index
        data_input = dict(
            rgb_path=sample['rgb_path'],
            img_id=sample['img_id'],
            img_hw=sample['img_hw'],
            tgt_intrinsic=sample['src_intrinsics'][0:1],
            src_masks=sample['src_masks'][pose_index][None, None, :],
            src_rgbs=sample['src_rgbs'][pose_index][None, None, :],
            src_c2w_mats=sample['src_c2w_mats'][pose_index][None, None, :],
            src_intrinsics=sample['src_intrinsics'][pose_index][None, None, :],
            depth_range=sample['depth_range'][None, :]
        )

        input_im = sample['src_rgbs'][pose_index].cpu().numpy() * 255.
        input_im = input_im.astype(np.uint8)
        filename = os.path.join(out_folder, 'input.png')
        imageio.imwrite(filename, input_im)

        # load training rays
        data_input['tgt_c2w_mat'] = data_input['src_c2w_mats'][0].clone()
        ray_sampler = RaySamplerSingleImage(data_input, device, render_stride=1)
        ray_batch = ray_sampler.get_all()

        with torch.no_grad():
            # Extract the feature maps from the src image
            featmaps = model.encode(ray_batch['src_rgbs'])

            ray_batch['xyz_batch'] = xyz_
            ray_batch['dir_batch'] = dir_

            sigma = get_point_cloud_from_nerf(batch_dict=ray_batch,
                                              model=model,
                                              projector=projector,
                                              featmaps=featmaps,
                                              chunk_size=1024*32)
        sigma = sigma.reshape(N, N, N)
        sigma_threshold = 0.5
        points = np.argwhere(sigma > sigma_threshold)
        pc_fp = os.path.join(out_folder, f"pc_{d_idx}.xyz")
        np.savetxt(pc_fp, points)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)