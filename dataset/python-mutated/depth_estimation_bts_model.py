import os.path as osp
import torch
from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .networks.bts_model import BtsModel
logger = get_logger()
__all__ = ['DepthEstimationBtsModel']

@MODELS.register_module(Tasks.image_depth_estimation, module_name=Models.bts_depth_estimation)
class DepthEstimationBtsModel(TorchModel):
    """ Depth estimation model bts, implemented from paper https://arxiv.org/pdf/1907.10326.pdf.
        The network utilizes novel local planar guidance layers located at multiple stage in the decoding phase.
        The bts model is composed with encoder and decoder, an encoder for dense feature extraction and a decoder
        for predicting the desired depth.
    """

    def __init__(self, model_dir: str, **kwargs):
        if False:
            print('Hello World!')
        "initialize the bts model from the `model_dir` path.\n\n        Args:\n            model_dir (str): the model path.\n            focal: focal length, pictures that do not work are input according to\n                the camera setting value at the time of shooting\n            dataset: used to set focal value according dataset type, only support 'kitti'\n        "
        super().__init__(model_dir, **kwargs)
        self.focal = 715.0873
        if 'focal' in kwargs:
            self.focal = kwargs['focal']
        elif 'dataset' in kwargs:
            if kwargs['dataset'] == 'nyu':
                self.focal = 518.8579
            elif kwargs['dataset'] == 'kitti':
                self.focal = 715.0873
        self.model = BtsModel(focal=self.focal)
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        checkpoint = torch.load(model_path)
        state_dict = {}
        for k in checkpoint['model_state_dict'].keys():
            if k.startswith('module.'):
                state_dict[k[7:]] = checkpoint['model_state_dict'][k]
            else:
                state_dict[k] = checkpoint['model_state_dict'][k]
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        return self.model(inputs['imgs'])

    def postprocess(self, inputs):
        if False:
            return 10
        results = {OutputKeys.DEPTHS: inputs}
        return results

    def inference(self, data):
        if False:
            print('Hello World!')
        results = self.forward(data)
        return results