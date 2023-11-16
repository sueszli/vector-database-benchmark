from modelscope.models.cv.video_depth_estimation.models.model_utils import merge_outputs
from modelscope.models.cv.video_depth_estimation.models.sfm_model_mf import SfmModelMF
from modelscope.models.cv.video_depth_estimation.utils.depth import depth2inv

class SupModelMF(SfmModelMF):
    """
    Model that inherits a depth and pose network from SfmModel and
    includes the photometric loss for self-supervised training.

    Parameters
    ----------
    kwargs : dict
        Extra parameters
    """

    def __init__(self, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self._network_requirements = {'depth_net': True, 'pose_net': False, 'percep_net': False}
        self._train_requirements = {'gt_depth': True, 'gt_pose': True}

    @property
    def logs(self):
        if False:
            i = 10
            return i + 15
        'Return logs.'
        return {**super().logs, **self._photometric_loss.logs}

    def supervised_loss(self, image, ref_images, inv_depths, gt_depth, gt_poses, poses, intrinsics, return_logs=False, progress=0.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates the self-supervised photometric loss.\n\n        Parameters\n        ----------\n        image : torch.Tensor [B,3,H,W]\n            Original image\n        ref_images : list of torch.Tensor [B,3,H,W]\n            Reference images from context\n        inv_depths : torch.Tensor [B,1,H,W]\n            Predicted inverse depth maps from the original image\n        poses : list of Pose\n            List containing predicted poses between original and context images\n        intrinsics : torch.Tensor [B,3,3]\n            Camera intrinsics\n        return_logs : bool\n            True if logs are stored\n        progress :\n            Training progress percentage\n\n        Returns\n        -------\n        output : dict\n            Dictionary containing a "loss" scalar a "metrics" dictionary\n        '
        return self._loss(image, ref_images, inv_depths, depth2inv(gt_depth), gt_poses, intrinsics, intrinsics, poses, return_logs=return_logs, progress=progress)

    def forward(self, batch, return_logs=False, progress=0.0):
        if False:
            print('Hello World!')
        '\n        Processes a batch.\n\n        Parameters\n        ----------\n        batch : dict\n            Input batch\n        return_logs : bool\n            True if logs are stored\n        progress :\n            Training progress percentage\n\n        Returns\n        -------\n        output : dict\n            Dictionary containing a "loss" scalar and different metrics and predictions\n            for logging and downstream usage.\n        '
        output = super().forward(batch, return_logs=return_logs)
        if not self.training:
            return output
        else:
            if output['poses'] is None:
                return None
            self_sup_output = self.supervised_loss(batch['rgb_original'], batch['rgb_context_original'], output['inv_depths'], batch['depth'], batch['pose_context'], output['poses'], batch['intrinsics'], return_logs=return_logs, progress=progress)
            return {'loss': self_sup_output['loss'], **merge_outputs(output, self_sup_output)}