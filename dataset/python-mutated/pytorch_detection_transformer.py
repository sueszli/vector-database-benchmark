"""
This module implements the task specific estimator for DEtection TRansformer (DETR) in PyTorch.

 | Paper link: https://arxiv.org/abs/2005.12872
"""
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Any
import numpy as np
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.pytorch import PyTorchEstimator
if TYPE_CHECKING:
    import torch
    from art.defences.postprocessor.postprocessor import Postprocessor
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
logger = logging.getLogger(__name__)

class PyTorchDetectionTransformer(ObjectDetectorMixin, PyTorchEstimator):
    """
    This class implements a model-specific object detector using DEtection TRansformer (DETR)
    and PyTorch following the input and output formats of torchvision.
    """
    MIN_IMAGE_SIZE = 800
    MAX_IMAGE_SIZE = 1333
    estimator_params = PyTorchEstimator.estimator_params + ['attack_losses']

    def __init__(self, model: 'torch.nn.Module'=None, input_shape: Tuple[int, ...]=(3, 800, 800), clip_values: Optional['CLIP_VALUES_TYPE']=None, channels_first: Optional[bool]=True, preprocessing_defences: Union['Preprocessor', List['Preprocessor'], None]=None, postprocessing_defences: Union['Postprocessor', List['Postprocessor'], None]=None, preprocessing: 'PREPROCESSING_TYPE'=None, attack_losses: Tuple[str, ...]=('loss_ce', 'loss_bbox', 'loss_giou'), device_type: str='gpu'):
        if False:
            return 10
        '\n        Initialization.\n\n        :param model: DETR model. The output of the model is `List[Dict[Tensor]]`, one for each input image. The\n                      fields of the Dict are as follows:\n\n                      - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values                         between 0 and H and 0 and W\n                      - labels (Tensor[N]): the predicted labels for each image\n                      - scores (Tensor[N]): the scores or each prediction\n        :param input_shape: Tuple of the form `(height, width)` of ints representing input image height and width\n        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and\n               maximum values allowed for features. If floats are provided, these will be used as the range of all\n               features. If arrays are provided, each value will be considered the bound for a feature, thus\n               the shape of clip values needs to match the total number of features.\n        :param channels_first: Set channels first or last.\n        :param preprocessing_defences: Preprocessing defence(s) to be applied by the estimator.\n        :param postprocessing_defences: Postprocessing defence(s) to be applied by the estimator.\n        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be\n               used for data preprocessing. The first value will be subtracted from the input. The input will then\n               be divided by the second one.\n        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU\n                            if available otherwise run on CPU.\n        '
        import torch
        from art.estimators.object_detection.detr import HungarianMatcher, SetCriterion, grad_enabled_forward
        if model is None:
            model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        func_type = type(model.forward)
        model.forward = func_type(grad_enabled_forward, model)
        super().__init__(model=model, clip_values=clip_values, channels_first=channels_first, preprocessing_defences=preprocessing_defences, postprocessing_defences=postprocessing_defences, preprocessing=preprocessing, device_type=device_type)
        if self.clip_values is not None:
            if not np.all(self.clip_values[0] == 0):
                raise ValueError('This estimator requires normalized input images with clip_vales=(0, 1).')
            if not np.all(self.clip_values[1] == 1):
                raise ValueError('This estimator requires normalized input images with clip_vales=(0, 1).')
        if self.postprocessing_defences is not None:
            raise ValueError('This estimator does not support `postprocessing_defences`.')
        self._input_shape = input_shape
        cost_class = 1.0
        cost_bbox = 5.0
        cost_giou = 2.0
        bbox_loss_coef = 5.0
        giou_loss_coef = 2.0
        eos_coef = 0.1
        self.max_norm = 0.1
        num_classes = 91
        matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
        self.weight_dict = {'loss_ce': 1, 'loss_bbox': bbox_loss_coef, 'loss_giou': giou_loss_coef}
        losses = ['labels', 'boxes', 'cardinality']
        self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=self.weight_dict, eos_coef=eos_coef, losses=losses)
        self._model.to(self._device)
        self._model.eval()
        self.attack_losses: Tuple[str, ...] = attack_losses

    @property
    def native_label_is_pytorch_format(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Are the native labels in PyTorch format [x1, y1, x2, y2]?\n        '
        return True

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if False:
            while True:
                i = 10
        '\n        Return the shape of one input sample.\n\n        :return: Shape of one input sample.\n        '
        return self._input_shape

    @property
    def device(self) -> 'torch.device':
        if False:
            return 10
        '\n        Get current used device.\n\n        :return: Current used device.\n        '
        return self._device

    def predict(self, x: np.ndarray, batch_size: int=128, **kwargs) -> List[Dict[str, np.ndarray]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform prediction for a batch of inputs.\n\n        :param x: Samples of shape (nb_samples, height, width, nb_channels).\n        :param batch_size: Batch size.\n        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict\n                 are as follows:\n\n                 - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                 - labels [N]: the labels for each image\n                 - scores [N]: the scores or each prediction.\n        '
        import torch
        from art.estimators.object_detection.detr import rescale_bboxes
        self._model.eval()
        (x_resized, _) = self._apply_resizing(x)
        (x_preprocessed, _) = self._apply_preprocessing(x_resized, y=None, fit=False)
        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0
        x_preprocessed_tensor = torch.from_numpy(x_preprocessed).to(self.device)
        x_preprocessed_tensor /= norm_factor
        model_output = self._model(x_preprocessed_tensor)
        predictions: List[Dict[str, np.ndarray]] = []
        for i in range(x_preprocessed_tensor.shape[0]):
            predictions.append({'boxes': rescale_bboxes(model_output['pred_boxes'][i, :, :].cpu(), (self._input_shape[2], self._input_shape[1])).detach().numpy(), 'labels': model_output['pred_logits'][i, :, :].unsqueeze(0).softmax(-1)[0, :, :-1].max(dim=1)[1].detach().cpu().numpy(), 'scores': model_output['pred_logits'][i, :, :].unsqueeze(0).softmax(-1)[0, :, :-1].max(dim=1)[0].detach().cpu().numpy()})
        return predictions

    def _get_losses(self, x: Union[np.ndarray, 'torch.Tensor'], y: List[Dict[str, Union[np.ndarray, 'torch.Tensor']]]) -> Tuple[Dict[str, 'torch.Tensor'], 'torch.Tensor', 'torch.Tensor']:
        if False:
            while True:
                i = 10
        '\n        Get the loss tensor output of the model including all preprocessing.\n\n        :param x: Samples of shape (nb_samples, nb_channels, height, width).\n        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The fields of the Dict are as\n                  follows:\n                  - boxes (FloatTensor[N, 4]): the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and\n                                               0 <= y1 < y2 <= H.\n                  - labels (Int64Tensor[N]): the labels for each image\n        :return: Loss gradients of the same shape as `x`.\n        '
        import torch
        self._model.train()
        self.set_dropout(False)
        self.set_multihead_attention(False)
        if self.all_framework_preprocessing:
            if y is not None and isinstance(y, list) and isinstance(y[0]['boxes'], np.ndarray):
                y_tensor = []
                for y_i in y:
                    y_t = {'boxes': torch.from_numpy(y_i['boxes']).type(torch.float).to(self.device), 'labels': torch.from_numpy(y_i['labels']).type(torch.int64).to(self.device)}
                    y_tensor.append(y_t)
            elif y is not None and isinstance(y, dict):
                y_tensor = []
                for i in range(y['boxes'].shape[0]):
                    y_t = {'boxes': y['boxes'][i], 'labels': y['labels'][i]}
                    y_tensor.append(y_t)
            else:
                y_tensor = y
            if isinstance(x, np.ndarray):
                if self.clip_values is not None:
                    norm_factor = self.clip_values[1]
                else:
                    norm_factor = 1.0
                x_grad = torch.from_numpy(x / norm_factor).to(self.device)
                x_grad.requires_grad = True
            else:
                x_grad = x.to(self.device)
                if x_grad.shape[2] < x_grad.shape[0] and x_grad.shape[2] < x_grad.shape[1]:
                    x_grad = torch.permute(x_grad, (2, 0, 1)).to(self.device)
            image_tensor_list_grad = x_grad
            (x_preprocessed, y_preprocessed) = self._apply_preprocessing(x_grad, y=y_tensor, fit=False, no_grad=False)
            inputs_t = x_preprocessed
        elif isinstance(x, np.ndarray):
            if y is not None and isinstance(y, list) and isinstance(y[0]['boxes'], np.ndarray):
                y_tensor = []
                for y_i in y:
                    y_t = {'boxes': torch.from_numpy(y_i['boxes']).type(torch.float).to(self.device), 'labels': torch.from_numpy(y_i['labels']).type(torch.int64).to(self.device)}
                    y_tensor.append(y_t)
            elif y is not None and isinstance(y[0]['boxes'], np.ndarray):
                y_tensor = []
                for y_i in y_preprocessed:
                    y_t = {'boxes': torch.from_numpy(y_i['boxes']).type(torch.float).to(self.device), 'labels': torch.from_numpy(y_i['labels']).type(torch.int64).to(self.device)}
                    y_tensor.append(y_t)
            else:
                y_tensor = y
            (x_preprocessed, y_preprocessed) = self._apply_preprocessing(x, y=y_tensor, fit=False, no_grad=True)
            if self.clip_values is not None:
                norm_factor = self.clip_values[1]
            else:
                norm_factor = 1.0
            x_grad = torch.from_numpy(x_preprocessed / norm_factor).to(self.device)
            x_grad.requires_grad = True
            image_tensor_list_grad = x_grad
            inputs_t = image_tensor_list_grad
        else:
            raise NotImplementedError('Combination of inputs and preprocessing not supported.')
        outputs = self._model(inputs_t)
        loss_components = self.criterion(outputs, y_preprocessed)
        return (loss_components, inputs_t, image_tensor_list_grad)

    def loss_gradient(self, x: Union[np.ndarray, 'torch.Tensor'], y: List[Dict[str, 'torch.Tensor']], **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Compute the gradient of the loss function w.r.t. `x`.\n\n        :param x: Samples of shape (nb_samples, nb_channels, height, width).\n        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The\n                  fields of the Dict are as follows:\n\n                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values                     between 0 and H and 0 and W\n                  - labels (Tensor[N]): the predicted labels for each image\n        :return: Loss gradients of the same shape as `x`.\n        '
        (x_resized, y_resized) = self._apply_resizing(x, y)
        (output, inputs_t, image_tensor_list_grad) = self._get_losses(x=x_resized, y=y_resized)
        loss = sum((output[k] * self.weight_dict[k] for k in output.keys() if k in self.weight_dict))
        self._model.zero_grad()
        loss.backward(retain_graph=True)
        if isinstance(x_resized, np.ndarray):
            if image_tensor_list_grad.grad is not None:
                grads = image_tensor_list_grad.grad.cpu().numpy().copy()
            else:
                raise ValueError('Gradient term in PyTorch model is `None`.')
        elif inputs_t.grad is not None:
            grads = inputs_t.grad.clone()
        else:
            raise ValueError('Gradient term in PyTorch model is `None`.')
        if self.clip_values is not None:
            grads = grads / self.clip_values[1]
        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x_resized, grads)
        return grads

    def get_activations(self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool=False) -> np.ndarray:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def fit(self, x: np.ndarray, y, batch_size: int=128, nb_epochs: int=20, **kwargs) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def compute_losses(self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, 'torch.Tensor']]]) -> Dict[str, np.ndarray]:
        if False:
            return 10
        '\n        Compute all loss components.\n\n        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,\n                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).\n        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The\n                  fields of the Dict are as follows:\n\n                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values                     between 0 and H and 0 and W\n                  - labels (Int64Tensor[N]): the predicted labels for each image\n                  - scores (Tensor[N]): the scores or each prediction.\n        :return: Dictionary of loss components.\n        '
        (x_resized, y) = self._apply_resizing(x, y)
        (output_tensor, _, _) = self._get_losses(x=x_resized, y=y)
        output = {}
        for (key, value) in output_tensor.items():
            if key in self.attack_losses:
                output[key] = value.detach().cpu().numpy()
        return output

    def compute_loss(self, x: Union[np.ndarray, 'torch.Tensor'], y: List[Dict[str, Union[np.ndarray, 'torch.Tensor']]], **kwargs) -> Union[np.ndarray, 'torch.Tensor']:
        if False:
            print('Hello World!')
        '\n        Compute the loss of the neural network for samples `x`.\n\n        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,\n                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).\n        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The\n                  fields of the Dict are as follows:\n\n                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values                     between 0 and H and 0 and W\n                  - labels (Int64Tensor[N]): the predicted labels for each image\n                  - scores (Tensor[N]): the scores or each prediction.\n        :return: Loss.\n        '
        import torch
        (x, y) = self._apply_resizing(x, y)
        (output, _, _) = self._get_losses(x=x, y=y)
        loss = None
        for loss_name in self.attack_losses:
            if loss is None:
                loss = output[loss_name]
            else:
                loss = loss + output[loss_name]
        assert loss is not None
        if isinstance(x, torch.Tensor):
            return loss
        return loss.detach().cpu().numpy()

    def _apply_resizing(self, x: Union[np.ndarray, 'torch.Tensor'], y: Any=None, height: int=800, width: int=800) -> Tuple[Union[np.ndarray, 'torch.Tensor'], List[Any]]:
        if False:
            return 10
        '\n        Resize the input and targets to dimensions expected by DETR.\n\n        :param x: Array or Tensor representing images of any size\n        :param y: List of targets to be transformed\n        :param height: Int representing desired height, the default is compatible with DETR\n        :param width: Int representing desired width, the default is compatible with DETR\n        '
        import cv2
        import torchvision.transforms as T
        import torch
        from art.estimators.object_detection.detr import revert_rescale_bboxes
        if self._input_shape[1] < self.MIN_IMAGE_SIZE or self._input_shape[1] > self.MAX_IMAGE_SIZE or self._input_shape[2] < self.MIN_IMAGE_SIZE or (self.input_shape[2] > self.MAX_IMAGE_SIZE):
            resized_imgs = []
            if isinstance(x, torch.Tensor):
                x = T.Resize(size=(height, width))(x).to(self.device)
            else:
                for i in x:
                    resized = cv2.resize(i.transpose(1, 2, 0), dsize=(height, width), interpolation=cv2.INTER_CUBIC)
                    resized = resized.transpose(2, 0, 1)
                    resized_imgs.append(resized)
                x = np.array(resized_imgs)
        elif self._input_shape[1] != self._input_shape[2]:
            rescale_dim = max(self._input_shape[1], self._input_shape[2])
            resized_imgs = []
            if isinstance(x, torch.Tensor):
                x = T.Resize(size=(rescale_dim, rescale_dim))(x).to(self.device)
            else:
                for i in x:
                    resized = cv2.resize(i.transpose(1, 2, 0), dsize=(rescale_dim, rescale_dim), interpolation=cv2.INTER_CUBIC)
                    resized = resized.transpose(2, 0, 1)
                    resized_imgs.append(resized)
                x = np.array(resized_imgs)
        targets: List[Any] = []
        if y is not None:
            if isinstance(y[0]['boxes'], torch.Tensor):
                for target in y:
                    assert isinstance(target['boxes'], torch.Tensor)
                    assert isinstance(target['labels'], torch.Tensor)
                    assert isinstance(target['scores'], torch.Tensor)
                    cxcy_norm = revert_rescale_bboxes(target['boxes'], (self.input_shape[2], self.input_shape[1]))
                    targets.append({'labels': target['labels'].type(torch.int64).to(self.device), 'boxes': cxcy_norm.to(self.device), 'scores': target['scores'].type(torch.float).to(self.device)})
            else:
                for target in y:
                    tensor_box = torch.from_numpy(target['boxes'])
                    cxcy_norm = revert_rescale_bboxes(tensor_box, (self.input_shape[2], self.input_shape[1]))
                    targets.append({'labels': torch.from_numpy(target['labels']).type(torch.int64).to(self.device), 'boxes': cxcy_norm.to(self.device), 'scores': torch.from_numpy(target['scores']).type(torch.float).to(self.device)})
        return (x, targets)