import sys
sys.path.append("./")
sys.path.append("../")

import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import time
import numpy as np
import torchvision
import cv2
from PIL import Image
from omegaconf import OmegaConf
import torch

import opt


parser = opt.config_parser()
args = parser.parse_args()
print(args)


def test_SRNDataset():
    """You can run this with the following command:
    python test/test_dataset.py --config configs/train_srn_chairs_ours.txt --debug
    """
    from data.srn import SRNDataset

    dataset = SRNDataset(args, "train")
    print(len(dataset))

    entry = dataset[0]
    print(entry.keys())

    for k, v in entry.items():
        try:
            print(k, v.shape)
        except:
            print(k, v)
    
    print("tgt_intrinsic:", entry['tgt_intrinsic'])
    print("tgt_c2w_mat:", entry['tgt_c2w_mat'])


def test_RaySamplerSingleImage():
    from models.sample_ray import RaySamplerSingleImage
    from data.srn import SRNDataset

    args.debug = True

    dataset = SRNDataset(args, "train")
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               shuffle=True)
    
    print(len(train_loader))

    device = torch.device("cuda:0")
    for data_dict in train_loader:
        tmp_ray_sampler = RaySamplerSingleImage(data_dict, device, render_stride=args.render_stride)


if __name__ == "__main__":
    test_SRNDataset()
    # test_RaySamplerSingleImage()

