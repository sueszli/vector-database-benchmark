import glob
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from .ray_utils import *

def normalize(v):
    if False:
        i = 10
        return i + 15
    'Normalize a vector.'
    return v / np.linalg.norm(v)

def average_poses(poses):
    if False:
        return 10
    "\n    Calculate the average pose, which is then used to center all poses\n    using @center_poses. Its computation is as follows:\n    1. Compute the center: the average of pose centers.\n    2. Compute the z axis: the normalized average z axis.\n    3. Compute axis y': the average y axis.\n    4. Compute x' = y' cross product z, then normalize it as the x axis.\n    5. Compute the y axis: z cross product x.\n\n    Note that at step 3, we cannot directly use y' as y axis since it's\n    not necessarily orthogonal to z axis. We need to pass from x to y.\n    Inputs:\n        poses: (N_images, 3, 4)\n    Outputs:\n        pose_avg: (3, 4) the average pose\n    "
    center = poses[..., 3].mean(0)
    z = normalize(poses[..., 2].mean(0))
    y_ = poses[..., 1].mean(0)
    x = normalize(np.cross(z, y_))
    y = np.cross(x, z)
    pose_avg = np.stack([x, y, z, center], 1)
    return pose_avg

def center_poses(poses, blender2opencv):
    if False:
        print('Hello World!')
    '\n    Center the poses so that we can use NDC.\n    See https://github.com/bmild/nerf/issues/34\n    Inputs:\n        poses: (N_images, 3, 4)\n    Outputs:\n        poses_centered: (N_images, 3, 4) the centered poses\n        pose_avg: (3, 4) the average pose\n    '
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg
    pose_avg_homo = pose_avg_homo
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))
    poses_homo = np.concatenate([poses, last_row], 1)
    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo
    poses_centered = poses_centered[:, :3]
    return (poses_centered, pose_avg_homo)

def viewmatrix(z, up, pos):
    if False:
        return 10
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    if False:
        for i in range(10):
            print('nop')
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    if False:
        while True:
            i = 10
    c2w = average_poses(c2ws_all)
    up = normalize(c2ws_all[:, :3, 1].sum(0))
    dt = 0.75
    (close_depth, inf_depth) = (near_fars.min() * 0.9, near_fars.max() * 5.0)
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views)
    return np.stack(render_poses)

class LLFFDataset(Dataset):

    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8):
        if False:
            while True:
                i = 10
        '\n        spheric_poses: whether the images are taken in a spheric inward-facing manner\n                       default: False (forward-facing)\n        val_num: number of val images (used for multigpu training, validate same image for all gpus)\n        '
        self.root_dir = datadir
        self.split = split
        self.hold_every = hold_every
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()
        self.blender2opencv = np.eye(4)
        self.read_meta()
        self.white_bg = False
        self.near_far = [0.0, 1.0]
        self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_meta(self):
        if False:
            print('Hello World!')
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images_4/*')))
        if self.split in ['train', 'test']:
            assert len(poses_bounds) == len(self.image_paths), 'Mismatch between number of images and number of poses! Please rerun COLMAP!'
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        self.near_fars = poses_bounds[:, -2:]
        (H, W, self.focal) = poses[0, :, -1]
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        (self.poses, self.pose_avg) = center_poses(poses, self.blender2opencv)
        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor
        N_views = 120
        self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views)
        (W, H) = self.img_wh
        self.directions = get_ray_directions_blender(H, W, self.focal)
        i_test = np.arange(0, self.poses.shape[0], self.hold_every)
        img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))
        self.all_rays = []
        self.all_rgbs = []
        for i in img_list:
            image_path = self.image_paths[i]
            c2w = torch.FloatTensor(self.poses[i])
            img = Image.open(image_path).convert('RGB')
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)
            img = img.view(3, -1).permute(1, 0)
            self.all_rgbs += [img]
            (rays_o, rays_d) = get_rays(self.directions, c2w)
            (rays_o, rays_d) = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1], 3)

    def define_transforms(self):
        if False:
            while True:
                i = 10
        self.transform = T.ToTensor()

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        sample = {'rays': self.all_rays[idx], 'rgbs': self.all_rgbs[idx]}
        return sample

    def get_render_pose(self, N_cameras=120):
        if False:
            print('Hello World!')
        return get_spiral(self.poses, self.near_fars, N_views=N_cameras)