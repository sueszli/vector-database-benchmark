import os.path as osp
from collections import OrderedDict
from typing import Any, Dict
import cv2
import numpy as np
import PIL
import torch
from modelscope.metainfo import Pipelines
from modelscope.models.cv.face_recognition.align_face import align_face
from modelscope.models.cv.face_recognition.torchkit.backbone.facemask_backbone import iresnet286
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from . import FaceProcessingBasePipeline
logger = get_logger()

@PIPELINES.register_module(Tasks.face_recognition, module_name=Pipelines.mask_face_recognition)
class MaskFaceRecognitionPipeline(FaceProcessingBasePipeline):

    def __init__(self, model: str, **kwargs):
        if False:
            print('Hello World!')
        '\n        use `model` to create a mask face recognition pipeline for prediction\n        Args:\n            model: model id on modelscope hub.\n        '
        super().__init__(model=model, **kwargs)
        face_model = iresnet286()
        state_dict = torch.load(osp.join(model, ModelFile.TORCH_MODEL_FILE))
        reviesed_state_dict = self._prefix_revision(state_dict)
        face_model.load_state_dict(reviesed_state_dict, strict=True)
        face_model = face_model.to(self.device)
        face_model.eval()
        self.face_model = face_model
        logger.info('face recognition model loaded!')

    def _prefix_revision(self, state_dict):
        if False:
            return 10
        new_state_dict = OrderedDict()
        for (k, v) in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        state = new_state_dict
        return state

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if False:
            return 10
        result = super().preprocess(input)
        align_img = result['img']
        face_img = align_img[:, :, ::-1]
        face_img = np.transpose(face_img, axes=(2, 0, 1))
        face_img = (face_img / 255.0 - 0.5) / 0.5
        face_img = face_img.astype(np.float32)
        result['img'] = face_img
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        assert input['img'] is not None
        img = input['img'].unsqueeze(0)
        emb = self.face_model(img).detach().cpu().numpy()
        emb /= np.sqrt(np.sum(emb ** 2, -1, keepdims=True))
        return {OutputKeys.IMG_EMBEDDING: emb}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return inputs