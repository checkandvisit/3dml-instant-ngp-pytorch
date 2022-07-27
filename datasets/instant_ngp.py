import json
from typing import Any, Dict, List
import torch
import glob
import numpy as np
import os
from PIL import Image
from einops import rearrange
from tqdm import tqdm

from .ray_utils import *

from .base import BaseDataset


class InstantNGPDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_meta(split)

    def read_meta(self, split):

        with open(os.path.join(self.root_dir, 'transform.json'), "r", encoding="utf-8") as _f:
            transform_json = json.load(_f)

        # Step 1: read and scale intrinsics (same for all images)
        w = int(transform_json["w"]*self.downsample)
        h = int(transform_json["h"]*self.downsample)
        self.img_wh = (w, h)

        fx = transform_json["fl_x"]*self.downsample
        fy = transform_json["fl_y"]*self.downsample
        cx = transform_json["cx"]*self.downsample
        cy = transform_json["cy"]*self.downsample
        self.K = torch.FloatTensor([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,   1]])

        self.directions = get_ray_directions(h, w, self.K)

        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        self.poses = np.array([np.array(x["transform_matrix"])[:3] for x in transform_json["frames"]])
        self.poses[..., 1:3] *= -1
        self.poses[..., 3] /= float(transform_json["aabb_scale"])

        self.rays = []
        if split == 'test_traj':
            self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
            self.poses = torch.FloatTensor(self.poses)
            return

        # use every 8th image as test set
        if split=='train':
            img_paths = [x["file_path"] for i, x in enumerate(transform_json["frames"]) if i%8!=0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i%8!=0])
        elif split=='test':
            img_paths = [x["file_path"] for i, x in enumerate(transform_json["frames"]) if i%8==0]
            self.poses = np.array([x for i, x in enumerate(self.poses) if i%8==0])

        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            path = os.path.join(self.root_dir, img_path)
            img = Image.open(path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (c, h, w)
            img = rearrange(img, 'c h w -> (h w) c')
            self.rays += [img]

        self.rays = torch.stack(self.rays)
        self.poses = torch.FloatTensor(self.poses)