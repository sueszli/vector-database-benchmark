import os.path as osp
import numpy as np
import torch
from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.image_semantic_segmentation import pan_merge, vit_adapter
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks

@MODELS.register_module(Tasks.image_segmentation, module_name=Models.swinL_semantic_segmentation)
@MODELS.register_module(Tasks.image_segmentation, module_name=Models.vitadapter_semantic_segmentation)
class SemanticSegmentation(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'str -- model file root.'
        super().__init__(model_dir, **kwargs)
        from mmcv.runner import load_checkpoint
        import mmcv
        from mmdet.models import build_detector
        config = osp.join(model_dir, 'mmcv_config.py')
        cfg = mmcv.Config.fromfile(config)
        if 'pretrained' in cfg.model:
            cfg.model.pretrained = None
        elif 'init_cfg' in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None
        cfg.model.train_cfg = None
        self.model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        _ = load_checkpoint(self.model, model_path, map_location='cpu')
        self.CLASSES = cfg['CLASSES']
        self.PALETTE = cfg['PALETTE']
        self.num_classes = len(self.CLASSES)
        self.cfg = cfg

    def forward(self, Inputs):
        if False:
            i = 10
            return i + 15
        return self.model(**Inputs)

    def postprocess(self, Inputs):
        if False:
            return 10
        semantic_result = Inputs[0]
        ids = np.unique(semantic_result)[::-1]
        legal_indices = ids != self.model.num_classes
        ids = ids[legal_indices]
        segms = semantic_result[None] == ids[:, None, None]
        masks = [it.astype(int) for it in segms]
        labels_txt = np.array(self.CLASSES)[ids].tolist()
        results = {OutputKeys.MASKS: masks, OutputKeys.LABELS: labels_txt, OutputKeys.SCORES: [0.999 for _ in range(len(labels_txt))]}
        return results

    def inference(self, data):
        if False:
            print('Hello World!')
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **data)
        return results