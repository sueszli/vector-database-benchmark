from ..builder import DETECTORS
from .single_stage_mono3d import SingleStageMono3DDetector

@DETECTORS.register_module()
class SMOKEMono3D(SingleStageMono3DDetector):
    """SMOKE <https://arxiv.org/abs/2002.10111>`_ for monocular 3D object
        detection.

    """

    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None, pretrained=None):
        if False:
            for i in range(10):
                print('nop')
        super(SMOKEMono3D, self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)