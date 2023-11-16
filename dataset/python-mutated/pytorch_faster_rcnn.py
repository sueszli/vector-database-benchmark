"""
This module implements the task specific estimator for Faster R-CNN v3 in PyTorch.
"""
import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector
if TYPE_CHECKING:
    import torch
    import torchvision
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor
logger = logging.getLogger(__name__)

class PyTorchFasterRCNN(PyTorchObjectDetector):
    """
    This class implements a model-specific object detector using Faster R-CNN and PyTorch following the input and output
    formats of torchvision.
    """

    def __init__(self, model: Optional['torchvision.models.detection.FasterRCNN']=None, input_shape: Tuple[int, ...]=(-1, -1, -1), optimizer: Optional['torch.optim.Optimizer']=None, clip_values: Optional['CLIP_VALUES_TYPE']=None, channels_first: Optional[bool]=True, preprocessing_defences: Union['Preprocessor', List['Preprocessor'], None]=None, postprocessing_defences: Union['Postprocessor', List['Postprocessor'], None]=None, preprocessing: 'PREPROCESSING_TYPE'=None, attack_losses: Tuple[str, ...]=('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'), device_type: str='gpu'):
        if False:
            i = 10
            return i + 15
        "\n        Initialization.\n\n        :param model: Faster R-CNN model. The output of the model is `List[Dict[str, torch.Tensor]]`, one for\n                      each input image. The fields of the Dict are as follows:\n\n                      - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and\n                        0 <= y1 < y2 <= H.\n                      - labels [N]: the labels for each image.\n                      - scores [N]: the scores of each prediction.\n        :param input_shape: The shape of one input sample.\n        :param optimizer: The optimizer for training the classifier.\n        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and\n               maximum values allowed for features. If floats are provided, these will be used as the range of all\n               features. If arrays are provided, each value will be considered the bound for a feature, thus\n               the shape of clip values needs to match the total number of features.\n        :param channels_first: Set channels first or last.\n        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.\n        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.\n        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be\n               used for data preprocessing. The first value will be subtracted from the input. The input will then\n               be divided by the second one.\n        :param attack_losses: Tuple of any combination of strings of loss components: 'loss_classifier', 'loss_box_reg',\n                              'loss_objectness', and 'loss_rpn_box_reg'.\n        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU\n                            if available otherwise run on CPU.\n        "
        import torchvision
        if model is None:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True)
        super().__init__(model=model, input_shape=input_shape, optimizer=optimizer, clip_values=clip_values, channels_first=channels_first, preprocessing_defences=preprocessing_defences, postprocessing_defences=postprocessing_defences, preprocessing=preprocessing, attack_losses=attack_losses, device_type=device_type)