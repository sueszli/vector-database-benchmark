import os.path as osp
import torch
from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks

@MODELS.register_module(Tasks.image_segmentation, module_name=Models.panoptic_segmentation)
class SwinLPanopticSegmentation(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        if False:
            return 10
        'str -- model file root.'
        super().__init__(model_dir, **kwargs)
        from mmcv.runner import load_checkpoint
        import mmcv
        from mmdet.models import build_detector
        config = osp.join(model_dir, 'config.py')
        cfg = mmcv.Config.fromfile(config)
        if 'pretrained' in cfg.model:
            cfg.model.pretrained = None
        elif 'init_cfg' in cfg.model.backbone:
            cfg.model.backbone.init_cfg = None
        cfg.model.train_cfg = None
        self.model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        checkpoint = load_checkpoint(self.model, model_path, map_location='cpu')
        self.CLASSES = checkpoint['meta']['CLASSES']
        self.num_classes = len(self.CLASSES)
        self.cfg = cfg

    def inference(self, data):
        if False:
            return 10
        'data is dict,contain img and img_metas,follow with mmdet.'
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **data)
        return results

    def forward(self, Inputs):
        if False:
            i = 10
            return i + 15
        return self.model(**Inputs)