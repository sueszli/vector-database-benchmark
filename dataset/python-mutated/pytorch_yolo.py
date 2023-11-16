"""
This module implements the task specific estimator for PyTorch YOLO v3 and v5 object detectors.

| Paper link: https://arxiv.org/abs/1804.02767
"""
import logging
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.object_detection.utils import cast_inputs_to_pt
from art.estimators.pytorch import PyTorchEstimator
if TYPE_CHECKING:
    import torch
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor
logger = logging.getLogger(__name__)

def translate_predictions_xcycwh_to_x1y1x2y2(y_pred_xcycwh: 'torch.Tensor', height: int, width: int) -> List[Dict[str, 'torch.Tensor']]:
    if False:
        while True:
            i = 10
    '\n    Convert object detection predictions from xcycwh (YOLO) to x1y1x2y2 (torchvision).\n\n    :param y_pred_xcycwh: Object detection labels in format xcycwh (YOLO).\n    :param height: Height of images in pixels.\n    :param width: Width if images in pixels.\n    :return: Object detection labels in format x1y1x2y2 (torchvision).\n    '
    import torch
    y_pred_x1y1x2y2 = []
    device = y_pred_xcycwh.device
    for y_pred in y_pred_xcycwh:
        boxes = torch.vstack([torch.maximum(y_pred[:, 0] - y_pred[:, 2] / 2, torch.tensor(0, device=device)), torch.maximum(y_pred[:, 1] - y_pred[:, 3] / 2, torch.tensor(0, device=device)), torch.minimum(y_pred[:, 0] + y_pred[:, 2] / 2, torch.tensor(height, device=device)), torch.minimum(y_pred[:, 1] + y_pred[:, 3] / 2, torch.tensor(width, device=device))]).permute((1, 0))
        labels = torch.argmax(y_pred[:, 5:], dim=1, keepdim=False)
        scores = y_pred[:, 4]
        y_i = {'boxes': boxes, 'labels': labels, 'scores': scores}
        y_pred_x1y1x2y2.append(y_i)
    return y_pred_x1y1x2y2

def translate_labels_x1y1x2y2_to_xcycwh(labels_x1y1x2y2: List[Dict[str, 'torch.Tensor']], height: int, width: int) -> 'torch.Tensor':
    if False:
        return 10
    '\n    Translate object detection labels from x1y1x2y2 (torchvision) to xcycwh (YOLO).\n\n    :param labels_x1y1x2y2: Object detection labels in format x1y1x2y2 (torchvision).\n    :param height: Height of images in pixels.\n    :param width: Width if images in pixels.\n    :return: Object detection labels in format xcycwh (YOLO).\n    '
    import torch
    labels_xcycwh_list = []
    device = labels_x1y1x2y2[0]['boxes'].device
    for (i, label_dict) in enumerate(labels_x1y1x2y2):
        labels = torch.zeros(len(label_dict['boxes']), 6, device=device)
        labels[:, 0] = i
        labels[:, 1] = label_dict['labels']
        labels[:, 2:6] = label_dict['boxes']
        labels[:, 2:6:2] = labels[:, 2:6:2] / width
        labels[:, 3:6:2] = labels[:, 3:6:2] / height
        labels[:, 4] -= labels[:, 2]
        labels[:, 5] -= labels[:, 3]
        labels[:, 2] += labels[:, 4] / 2
        labels[:, 3] += labels[:, 5] / 2
        labels_xcycwh_list.append(labels)
    labels_xcycwh = torch.vstack(labels_xcycwh_list)
    return labels_xcycwh

class PyTorchYolo(ObjectDetectorMixin, PyTorchEstimator):
    """
    This module implements the model- and task specific estimator for YOLO v3, v5 object detector models in PyTorch.

    | Paper link: https://arxiv.org/abs/1804.02767
    """
    estimator_params = PyTorchEstimator.estimator_params + ['input_shape', 'optimizer', 'attack_losses']

    def __init__(self, model: 'torch.nn.Module', input_shape: Tuple[int, ...]=(3, 416, 416), optimizer: Optional['torch.optim.Optimizer']=None, clip_values: Optional['CLIP_VALUES_TYPE']=None, channels_first: Optional[bool]=True, preprocessing_defences: Union['Preprocessor', List['Preprocessor'], None]=None, postprocessing_defences: Union['Postprocessor', List['Postprocessor'], None]=None, preprocessing: 'PREPROCESSING_TYPE'=None, attack_losses: Tuple[str, ...]=('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'), device_type: str='gpu'):
        if False:
            return 10
        "\n        Initialization.\n\n        :param model: YOLO v3 or v5 model wrapped as demonstrated in examples/get_started_yolo.py.\n                      The output of the model is `List[Dict[str, torch.Tensor]]`, one for each input image.\n                      The fields of the Dict are as follows:\n\n                      - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and\n                        0 <= y1 < y2 <= H.\n                      - labels [N]: the labels for each image.\n                      - scores [N]: the scores of each prediction.\n        :param input_shape: The shape of one input sample.\n        :param optimizer: The optimizer for training the classifier.\n        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and\n               maximum values allowed for features. If floats are provided, these will be used as the range of all\n               features. If arrays are provided, each value will be considered the bound for a feature, thus\n               the shape of clip values needs to match the total number of features.\n        :param channels_first: Set channels first or last.\n        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.\n        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.\n        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be\n               used for data preprocessing. The first value will be subtracted from the input. The input will then\n               be divided by the second one.\n        :param attack_losses: Tuple of any combination of strings of loss components: 'loss_classifier', 'loss_box_reg',\n                              'loss_objectness', and 'loss_rpn_box_reg'.\n        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU\n                            if available otherwise run on CPU.\n        "
        import torch
        super().__init__(model=model, clip_values=clip_values, channels_first=channels_first, preprocessing_defences=preprocessing_defences, postprocessing_defences=postprocessing_defences, preprocessing=preprocessing, device_type=device_type)
        self._input_shape = input_shape
        self._optimizer = optimizer
        self._attack_losses = attack_losses
        if self.clip_values is not None:
            if self.clip_values[0] != 0:
                raise ValueError('This estimator requires un-normalized input images with clip_vales=(0, max_value).')
            if self.clip_values[1] <= 0:
                raise ValueError('This estimator requires un-normalized input images with clip_vales=(0, max_value).')
        if self.postprocessing_defences is not None:
            raise ValueError('This estimator does not support `postprocessing_defences`.')
        self._model: torch.nn.Module
        self._model.to(self._device)
        self._model.eval()

    @property
    def native_label_is_pytorch_format(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Return are the native labels in PyTorch format [x1, y1, x2, y2]?\n\n        :return: Are the native labels in PyTorch format [x1, y1, x2, y2]?\n        '
        return True

    @property
    def model(self) -> 'torch.nn.Module':
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the model.\n\n        :return: The model.\n        '
        return self._model

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        '\n        Return the shape of one input sample.\n\n        :return: Shape of one input sample.\n        '
        return self._input_shape

    @property
    def optimizer(self) -> Optional['torch.optim.Optimizer']:
        if False:
            print('Hello World!')
        '\n        Return the optimizer.\n\n        :return: The optimizer.\n        '
        return self._optimizer

    @property
    def attack_losses(self) -> Tuple[str, ...]:
        if False:
            print('Hello World!')
        '\n        Return the combination of strings of the loss components.\n\n        :return: The combination of strings of the loss components.\n        '
        return self._attack_losses

    @property
    def device(self) -> 'torch.device':
        if False:
            print('Hello World!')
        '\n        Get current used device.\n\n        :return: Current used device.\n        '
        return self._device

    def _preprocess_and_convert_inputs(self, x: Union[np.ndarray, 'torch.Tensor'], y: Optional[List[Dict[str, Union[np.ndarray, 'torch.Tensor']]]]=None, fit: bool=False, no_grad: bool=True) -> Tuple['torch.Tensor', List[Dict[str, 'torch.Tensor']]]:
        if False:
            while True:
                i = 10
        '\n        Apply preprocessing on inputs `(x, y)` and convert to tensors, if needed.\n\n        :param x: Samples of shape NCHW or NHWC.\n        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.\n                  The fields of the Dict are as follows:\n\n                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                  - labels [N]: the labels for each image.\n        :param fit: `True` if the function is call before fit/training and `False` if the function is called before a\n                    predict operation.\n        :param no_grad: `True` if no gradients required.\n        :return: Preprocessed inputs `(x, y)` as tensors.\n        '
        import torch
        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0
        if self.all_framework_preprocessing:
            (x_tensor, y_tensor) = cast_inputs_to_pt(x, y)
            if not self.channels_first:
                x_tensor = torch.permute(x_tensor, (0, 3, 1, 2))
            x_tensor = x_tensor / norm_factor
            if not no_grad:
                if x_tensor.is_leaf:
                    x_tensor.requires_grad = True
                else:
                    x_tensor.retain_grad()
            (x_preprocessed, y_preprocessed) = self._apply_preprocessing(x=x_tensor, y=y_tensor, fit=fit, no_grad=no_grad)
        elif isinstance(x, np.ndarray):
            (x_preprocessed, y_preprocessed) = self._apply_preprocessing(x=x, y=y, fit=fit, no_grad=no_grad)
            (x_preprocessed, y_preprocessed) = cast_inputs_to_pt(x_preprocessed, y_preprocessed)
            if not self.channels_first:
                x_preprocessed = torch.permute(x_preprocessed, (0, 3, 1, 2))
            x_preprocessed = x_preprocessed / norm_factor
            if not no_grad:
                x_preprocessed.requires_grad = True
        else:
            raise NotImplementedError('Combination of inputs and preprocessing not supported.')
        return (x_preprocessed, y_preprocessed)

    def _get_losses(self, x: Union[np.ndarray, 'torch.Tensor'], y: List[Dict[str, Union[np.ndarray, 'torch.Tensor']]]) -> Tuple[Dict[str, 'torch.Tensor'], 'torch.Tensor']:
        if False:
            while True:
                i = 10
        '\n        Get the loss tensor output of the model including all preprocessing.\n\n        :param x: Samples of shape NCHW or NHWC.\n        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.\n                  The fields of the Dict are as follows:\n\n                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                  - labels [N]: the labels for each image.\n        :return: Loss components and gradients of the input `x`.\n        '
        self._model.train()
        (x_preprocessed, y_preprocessed) = self._preprocess_and_convert_inputs(x=x, y=y, fit=False, no_grad=False)
        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]
        y_preprocessed_yolo = translate_labels_x1y1x2y2_to_xcycwh(labels_x1y1x2y2=y_preprocessed, height=height, width=width)
        x_preprocessed = x_preprocessed.to(self.device)
        y_preprocessed_yolo = y_preprocessed_yolo.to(self.device)
        if x_preprocessed.is_leaf:
            x_preprocessed.requires_grad = True
        else:
            x_preprocessed.retain_grad()
        loss_components = self._model(x_preprocessed, y_preprocessed_yolo)
        return (loss_components, x_preprocessed)

    def loss_gradient(self, x: Union[np.ndarray, 'torch.Tensor'], y: List[Dict[str, Union[np.ndarray, 'torch.Tensor']]], **kwargs) -> Union[np.ndarray, 'torch.Tensor']:
        if False:
            i = 10
            return i + 15
        '\n        Compute the gradient of the loss function w.r.t. `x`.\n\n        :param x: Samples of shape NCHW or NHWC.\n        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.\n                  The fields of the Dict are as follows:\n\n                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                  - labels [N]: the labels for each image.\n        :return: Loss gradients of the same shape as `x`.\n        '
        import torch
        (loss_components, x_grad) = self._get_losses(x=x, y=y)
        loss = None
        for loss_name in self.attack_losses:
            if loss is None:
                loss = loss_components[loss_name]
            else:
                loss = loss + loss_components[loss_name]
        self._model.zero_grad()
        loss.backward(retain_graph=True)
        if x_grad.grad is not None:
            if isinstance(x, np.ndarray):
                grads = x_grad.grad.cpu().numpy()
            else:
                grads = x_grad.grad.clone()
        else:
            raise ValueError('Gradient term in PyTorch model is `None`.')
        if self.clip_values is not None:
            grads = grads / self.clip_values[1]
        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)
        if not self.channels_first:
            if isinstance(x, np.ndarray):
                grads = np.transpose(grads, (0, 2, 3, 1))
            else:
                grads = torch.permute(grads, (0, 2, 3, 1))
        assert grads.shape == x.shape
        return grads

    def predict(self, x: np.ndarray, batch_size: int=128, **kwargs) -> List[Dict[str, np.ndarray]]:
        if False:
            print('Hello World!')
        '\n        Perform prediction for a batch of inputs.\n\n        :param x: Samples of shape NCHW or NHWC.\n        :param batch_size: Batch size.\n        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict\n                 are as follows:\n\n                 - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                 - labels [N]: the labels for each image.\n                 - scores [N]: the scores of each prediction.\n        '
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        self._model.eval()
        (x_preprocessed, _) = self._preprocess_and_convert_inputs(x=x, y=None, fit=False, no_grad=True)
        dataset = TensorDataset(x_preprocessed)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]
        predictions: List[Dict[str, np.ndarray]] = []
        for (x_batch,) in dataloader:
            x_batch = x_batch.to(self._device)
            with torch.no_grad():
                predictions_xcycwh = self._model(x_batch)
            predictions_x1y1x2y2 = translate_predictions_xcycwh_to_x1y1x2y2(y_pred_xcycwh=predictions_xcycwh, height=height, width=width)
            for prediction_x1y1x2y2 in predictions_x1y1x2y2:
                prediction = {}
                prediction['boxes'] = prediction_x1y1x2y2['boxes'].detach().cpu().numpy()
                prediction['labels'] = prediction_x1y1x2y2['labels'].detach().cpu().numpy()
                prediction['scores'] = prediction_x1y1x2y2['scores'].detach().cpu().numpy()
                if 'masks' in prediction_x1y1x2y2:
                    prediction['masks'] = prediction_x1y1x2y2['masks'].detach().cpu().numpy().squeeze()
                predictions.append(prediction)
        return predictions

    def fit(self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, 'torch.Tensor']]], batch_size: int=128, nb_epochs: int=10, drop_last: bool=False, scheduler: Optional['torch.optim.lr_scheduler._LRScheduler']=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Fit the classifier on the training set `(x, y)`.\n\n        :param x: Samples of shape NCHW or NHWC.\n        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.\n                  The fields of the Dict are as follows:\n\n                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                  - labels [N]: the labels for each image.\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for training.\n        :param drop_last: Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by\n                          the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then\n                          the last batch will be smaller. (default: ``False``)\n        :param scheduler: Learning rate scheduler to run at the start of every epoch.\n        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch\n                       and providing it takes no effect.\n        '
        import torch
        from torch.utils.data import Dataset, DataLoader
        self._model.train()
        if self._optimizer is None:
            raise ValueError('An optimizer is needed to train the model, but none for provided.')
        (x_preprocessed, y_preprocessed) = self._preprocess_and_convert_inputs(x=x, y=y, fit=True, no_grad=True)

        class ObjectDetectionDataset(Dataset):
            """
            Object detection dataset in PyTorch.
            """

            def __init__(self, x, y):
                if False:
                    print('Hello World!')
                self.x = x
                self.y = y

            def __len__(self):
                if False:
                    while True:
                        i = 10
                return len(self.x)

            def __getitem__(self, idx):
                if False:
                    return 10
                return (self.x[idx], self.y[idx])
        dataset = ObjectDetectionDataset(x_preprocessed, y_preprocessed)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, collate_fn=lambda batch: list(zip(*batch)))
        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]
        for _ in range(nb_epochs):
            for (x_batch, y_batch) in dataloader:
                x_batch = torch.stack(x_batch)
                y_batch = translate_labels_x1y1x2y2_to_xcycwh(labels_x1y1x2y2=y_batch, height=height, width=width)
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self._optimizer.zero_grad()
                loss_components = self._model(x_batch, y_batch)
                if isinstance(loss_components, dict):
                    loss = sum(loss_components.values())
                else:
                    loss = loss_components
                loss.backward()
                self._optimizer.step()
            if scheduler is not None:
                scheduler.step()

    def get_activations(self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool=False) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def compute_losses(self, x: Union[np.ndarray, 'torch.Tensor'], y: List[Dict[str, Union[np.ndarray, 'torch.Tensor']]]) -> Dict[str, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute all loss components.\n\n        :param x: Samples of shape NCHW or NHWC.\n        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.\n                  The fields of the Dict are as follows:\n\n                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                  - labels [N]: the labels for each image.\n        :return: Dictionary of loss components.\n        '
        (loss_components, _) = self._get_losses(x=x, y=y)
        output = {}
        for (key, value) in loss_components.items():
            output[key] = value.detach().cpu().numpy()
        return output

    def compute_loss(self, x: Union[np.ndarray, 'torch.Tensor'], y: List[Dict[str, Union[np.ndarray, 'torch.Tensor']]], **kwargs) -> Union[np.ndarray, 'torch.Tensor']:
        if False:
            i = 10
            return i + 15
        '\n        Compute the loss of the neural network for samples `x`.\n\n        :param x: Samples of shape NCHW or NHWC.\n        :param y: Target values of format `List[Dict[str, Union[np.ndarray, torch.Tensor]]]`, one for each input image.\n                  The fields of the Dict are as follows:\n\n                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                  - labels [N]: the labels for each image.\n        :return: Loss.\n        '
        import torch
        (loss_components, _) = self._get_losses(x=x, y=y)
        loss = None
        for loss_name in self.attack_losses:
            if loss is None:
                loss = loss_components[loss_name]
            else:
                loss = loss + loss_components[loss_name]
        assert loss is not None
        if isinstance(x, torch.Tensor):
            return loss
        return loss.detach().cpu().numpy()