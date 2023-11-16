import os
import os.path as osp
from typing import Any, Dict, List, Union
import numpy as np
import torch
from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
logger = get_logger()
__all__ = ['DepeDetect']

@MODELS.register_module(Tasks.object_detection_3d, module_name=Models.depe)
class DepeDetect(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        if False:
            print('Hello World!')
        'DEPE is a simple and pure DETR-like 3D detector with transformers,\n        for more information please refer to:\n        https://www.modelscope.cn/models/damo/cv_object-detection-3d_depe/summary\n\n        initialize the 3d object detection model from the `model_dir` path.\n        Args:\n            model_dir (str): the model path.\n        '
        super().__init__(model_dir, *args, **kwargs)
        from mmcv.runner import load_checkpoint
        import modelscope.models.cv.object_detection_3d.depe.mmdet3d_plugin
        from modelscope.models.cv.object_detection_3d.depe.mmdet3d_plugin.models.detectors import Petr3D
        config_path = osp.join(model_dir, ModelFile.CONFIGURATION)
        config = Config.from_file(config_path)
        detector = Petr3D(**config.model.network_param)
        model_file = kwargs.get('model_file', ModelFile.TORCH_MODEL_BIN_FILE)
        ckpt_path = osp.join(model_dir, model_file)
        logger.info(f'loading model from {ckpt_path}')
        load_checkpoint(detector, ckpt_path, map_location='cpu')
        detector.eval()
        self.detector = detector
        logger.info('load model done')

    def forward(self, img: Union[torch.Tensor, List[torch.Tensor]], img_metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if False:
            print('Hello World!')
        '\n        Args:\n            img (`torch.Tensor`): batched image tensor or list of batched image tensor,\n                shape of each tensor is [B, N, C, H, W], N is 12 for 6 views from current\n                and history frame.\n            img_metas (` List[Dict[str, Any]`): image meta info.\n        Return:\n            result (`List[Dict[str, Any]]`): list of detection results.\n        '
        if isinstance(img, torch.Tensor):
            img = [img]
            img_metas = [img_metas]
        result = self.detector(return_loss=False, rescale=True, img=img, img_metas=img_metas)
        assert result is not None
        return result