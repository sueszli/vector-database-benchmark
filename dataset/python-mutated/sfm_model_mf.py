import random
import torch.nn as nn
from modelscope.models.cv.video_depth_estimation.geometry.pose import Pose
from modelscope.models.cv.video_depth_estimation.utils.image import flip_lr as flip_lr_img
from modelscope.models.cv.video_depth_estimation.utils.image import flip_lr_intr, flip_mf_model, interpolate_scales
from modelscope.models.cv.video_depth_estimation.utils.misc import make_list

class SfmModelMF(nn.Module):
    """
    Model class encapsulating a pose and depth networks.

    Parameters
    ----------
    depth_net : nn.Module
        Depth network to be used
    pose_net : nn.Module
        Pose network to be used
    rotation_mode : str
        Rotation mode for the pose network
    flip_lr_prob : float
        Probability of flipping when using the depth network
    upsample_depth_maps : bool
        True if depth map scales are upsampled to highest resolution
    kwargs : dict
        Extra parameters
    """

    def __init__(self, depth_net=None, pose_net=None, rotation_mode='euler', flip_lr_prob=0.0, upsample_depth_maps=False, min_depth=0.1, max_depth=100, **kwargs):
        if False:
            return 10
        super().__init__()
        self.depth_net = depth_net
        self.pose_net = pose_net
        self.rotation_mode = rotation_mode
        self.flip_lr_prob = flip_lr_prob
        self.upsample_depth_maps = upsample_depth_maps
        self.min_depth = min_depth
        self.max_depth = max_depth
        self._logs = {}
        self._losses = {}
        self._network_requirements = {'depth_net': True, 'pose_net': False, 'percep_net': False}
        self._train_requirements = {'gt_depth': False, 'gt_pose': False}

    @property
    def logs(self):
        if False:
            return 10
        'Return logs.'
        return self._logs

    @property
    def losses(self):
        if False:
            for i in range(10):
                print('nop')
        'Return metrics.'
        return self._losses

    def add_loss(self, key, val):
        if False:
            for i in range(10):
                print('nop')
        'Add a new loss to the dictionary and detaches it.'
        self._losses[key] = val.detach()

    @property
    def network_requirements(self):
        if False:
            i = 10
            return i + 15
        '\n        Networks required to run the model\n\n        Returns\n        -------\n        requirements : dict\n            depth_net : bool\n                Whether a depth network is required by the model\n            pose_net : bool\n                Whether a depth network is required by the model\n        '
        return self._network_requirements

    @property
    def train_requirements(self):
        if False:
            i = 10
            return i + 15
        '\n        Information required by the model at training stage\n\n        Returns\n        -------\n        requirements : dict\n            gt_depth : bool\n                Whether ground truth depth is required by the model at training time\n            gt_pose : bool\n                Whether ground truth pose is required by the model at training time\n        '
        return self._train_requirements

    def add_depth_net(self, depth_net):
        if False:
            return 10
        'Add a depth network to the model'
        self.depth_net = depth_net

    def add_pose_net(self, pose_net):
        if False:
            return 10
        'Add a pose network to the model'
        self.pose_net = pose_net

    def compute_inv_depths(self, image, ref_imgs, intrinsics):
        if False:
            print('Hello World!')
        'Computes inverse depth maps from single images'
        flip_lr = random.random() < self.flip_lr_prob if self.training else False
        if flip_lr:
            intrinsics = flip_lr_intr(intrinsics, width=image.shape[3])
        inv_depths_with_poses = flip_mf_model(self.depth_net, image, ref_imgs, intrinsics, flip_lr)
        (inv_depths, poses) = inv_depths_with_poses
        inv_depths = make_list(inv_depths)
        if flip_lr:
            inv_depths = [flip_lr_img(inv_d) for inv_d in inv_depths]
        if self.upsample_depth_maps:
            inv_depths = interpolate_scales(inv_depths, mode='nearest', align_corners=None)
        return (inv_depths, poses)

    def compute_poses(self, image, contexts, intrinsics, depth):
        if False:
            print('Hello World!')
        'Compute poses from image and a sequence of context images'
        pose_vec = self.pose_net(image, contexts, intrinsics, depth)
        if pose_vec is None:
            return None
        if pose_vec.shape[2] == 6:
            return [Pose.from_vec(pose_vec[:, i], self.rotation_mode) for i in range(pose_vec.shape[1])]
        else:
            return [Pose(pose_vec[:, i]) for i in range(pose_vec.shape[1])]

    def forward(self, batch, return_logs=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Processes a batch.\n\n        Parameters\n        ----------\n        batch : dict\n            Input batch\n        return_logs : bool\n            True if logs are stored\n\n        Returns\n        -------\n        output : dict\n            Dictionary containing predicted inverse depth maps and poses\n        '
        (inv_depths, pose_vec) = self.compute_inv_depths(batch['rgb'], batch['rgb_context'], batch['intrinsics'])
        if pose_vec.shape[2] == 6:
            poses = [Pose.from_vec(pose_vec[:, i], self.rotation_mode) for i in range(pose_vec.shape[1])]
        elif pose_vec.shape[2] == 4 and pose_vec.shape[3] == 4:
            poses = [Pose(pose_vec[:, i]) for i in range(pose_vec.shape[1])]
        else:
            poses = []
            for i in range(pose_vec.shape[1]):
                poses_view = []
                for j in range(pose_vec.shape[2]):
                    poses_view.append(Pose.from_vec(pose_vec[:, i, j], self.rotation_mode))
                poses.append(poses_view)
        return {'inv_depths': inv_depths, 'poses': poses}