"""
This module implements the task specific estimator for PyTorch GOTURN object tracker.
"""
import logging
import time
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from art.estimators.object_tracking.object_tracker import ObjectTrackerMixin
from art.estimators.pytorch import PyTorchEstimator
if TYPE_CHECKING:
    import PIL
    import torch
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor
logger = logging.getLogger(__name__)

class PyTorchGoturn(ObjectTrackerMixin, PyTorchEstimator):
    """
    This module implements the task- and model-specific estimator for PyTorch GOTURN (object tracking).
    """
    estimator_params = PyTorchEstimator.estimator_params + ['attack_losses']

    def __init__(self, model, input_shape: Tuple[int, ...], clip_values: Optional['CLIP_VALUES_TYPE']=None, channels_first: Optional[bool]=None, preprocessing_defences: Union['Preprocessor', List['Preprocessor'], None]=None, postprocessing_defences: Union['Postprocessor', List['Postprocessor'], None]=None, preprocessing: 'PREPROCESSING_TYPE'=None, device_type: str='gpu'):
        if False:
            print('Hello World!')
        '\n        Initialization.\n\n        :param model: GOTURN model.\n        :param input_shape: Shape of one input sample as expected by the model, e.g. input_shape=(3, 227, 227).\n        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and\n               maximum values allowed for features. If floats are provided, these will be used as the range of all\n               features. If arrays are provided, each value will be considered the bound for a feature, thus\n               the shape of clip values needs to match the total number of features.\n        :param channels_first: Set channels first or last.\n        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.\n        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.\n        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be\n               used for data preprocessing. The first value will be subtracted from the input. The input will then\n               be divided by the second one.\n        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU\n                            if available otherwise run on CPU.\n        '
        import torch
        self._device: torch.device
        if device_type == 'cpu' or not torch.cuda.is_available():
            self._device = torch.device('cpu')
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device(f'cuda:{cuda_idx}')
        model.to(self._device)
        super().__init__(model=model, clip_values=clip_values, channels_first=channels_first, preprocessing_defences=preprocessing_defences, postprocessing_defences=postprocessing_defences, preprocessing=preprocessing, device_type=device_type)
        self.name = 'PyTorchGoturn'
        self.is_deterministic = True
        self._input_shape = input_shape
        if self.clip_values is not None:
            if self.clip_values[0] != 0:
                raise ValueError('This classifier requires un-normalized input images with clip_vales=(0, 255).')
            if self.clip_values[1] not in [1, 255]:
                raise ValueError('This classifier requires un-normalized input images with clip_vales=(0, 1) or clip_vales=(0, 255).')
        if self.postprocessing_defences is not None:
            raise ValueError('This estimator does not support `postprocessing_defences`.')
        self.attack_losses: Tuple[str, ...] = ('torch.nn.L1Loss',)

    @property
    def native_label_is_pytorch_format(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Are the native labels in PyTorch format [x1, y1, x2, y2]?\n        '
        return True

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        '\n        Return the shape of one input sample.\n\n        :return: Shape of one input sample.\n        '
        return self._input_shape

    @property
    def device(self) -> 'torch.device':
        if False:
            while True:
                i = 10
        '\n        Get current used device.\n\n        :return: Current used device.\n        '
        return self._device

    def _get_losses(self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, 'torch.Tensor']]], reduction: str='sum') -> Tuple[Dict[str, Union['torch.Tensor', int, List['torch.Tensor']]], List['torch.Tensor'], List['torch.Tensor']]:
        if False:
            return 10
        "\n        Get the loss tensor output of the model including all preprocessing.\n\n        :param x: Samples of shape (nb_samples, nb_frames, height, width, nb_channels).\n        :param y: Target values of format `List[Dict[str, np.ndarray]]`, one dictionary for each input image. The keys\n                  of the dictionary are:\n\n                  - boxes [N_FRAMES, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and\n                                         0 <= y1 < y2 <= H.\n        :param reduction: Specifies the reduction to apply to the output: 'none' | 'sum'.\n                          'none': no reduction will be applied.\n                          'sum': the output will be summed.\n        :return: Loss dictionary, list of input tensors, and list of gradient tensors.\n        "
        import torch
        self._model.train()
        if self.all_framework_preprocessing:
            if isinstance(x, torch.Tensor):
                raise NotImplementedError
            if y is not None and isinstance(y[0]['boxes'], np.ndarray):
                y_tensor = []
                for (i, y_i) in enumerate(y):
                    y_t = {}
                    y_t['boxes'] = torch.from_numpy(y_i['boxes']).float().to(self.device)
                    y_tensor.append(y_t)
            else:
                y_tensor = y
            image_tensor_list_grad = []
            y_preprocessed = []
            inputs_t: List['torch.Tensor'] = []
            for i in range(x.shape[0]):
                if self.clip_values is not None:
                    x_grad = torch.from_numpy(x[i]).to(self.device).float()
                else:
                    x_grad = torch.from_numpy(x[i]).to(self.device).float()
                x_grad.requires_grad = True
                image_tensor_list_grad.append(x_grad)
                x_grad_1 = torch.unsqueeze(x_grad, dim=0)
                (x_preprocessed_i, y_preprocessed_i) = self._apply_preprocessing(x_grad_1, y=[y_tensor[i]], fit=False, no_grad=False)
                x_preprocessed_i = torch.squeeze(x_preprocessed_i)
                y_preprocessed.append(y_preprocessed_i[0])
                inputs_t.append(x_preprocessed_i)
        elif isinstance(x, np.ndarray):
            raise NotImplementedError
        else:
            raise NotImplementedError('Combination of inputs and preprocessing not supported.')
        labels_t = y_preprocessed
        if isinstance(y[0]['boxes'], np.ndarray):
            y_init = torch.from_numpy(y[0]['boxes']).to(self.device)
        else:
            y_init = y[0]['boxes']
        loss_list = []
        for i in range(x.shape[0]):
            x_i = inputs_t[i]
            y_pred = self._track(x=x_i, y_init=y_init[i])
            gt_bb = labels_t[i]['boxes']
            loss = torch.nn.L1Loss(size_average=False)(y_pred.float(), gt_bb.float())
            loss_list.append(loss)
        loss_dict: Dict[str, Union['torch.Tensor', int, List['torch.Tensor']]] = {}
        if reduction == 'sum':
            loss_dict['torch.nn.L1Loss'] = sum(loss_list)
        elif reduction == 'none':
            loss_dict['torch.nn.L1Loss'] = loss_list
        else:
            raise ValueError('Reduction not recognised.')
        return (loss_dict, inputs_t, image_tensor_list_grad)

    def loss_gradient(self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, 'torch.Tensor']]], **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Compute the gradient of the loss function w.r.t. `x`.\n\n        :param x: Samples of shape (nb_samples, height, width, nb_channels).\n        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The\n                  fields of the Dict are as follows:\n\n                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values                     between 0 and H and 0 and W.\n                  - labels (Int64Tensor[N]): the predicted labels for each image.\n                  - scores (Tensor[N]): the scores or each prediction.\n        :return: Loss gradients of the same shape as `x`.\n        '
        grad_list = []
        for i in range(x.shape[0]):
            x_i = x[[i]]
            y_i = [y[i]]
            (output, _, image_tensor_list_grad) = self._get_losses(x=x_i, y=y_i)
            loss = None
            for loss_name in self.attack_losses:
                if loss is None:
                    loss = output[loss_name]
                else:
                    loss = loss + output[loss_name]
            self._model.zero_grad()
            loss.backward(retain_graph=True)
            for img in image_tensor_list_grad:
                if img.grad is not None:
                    gradients = img.grad.cpu().numpy().copy()
                else:
                    gradients = None
                grad_list.append(gradients)
        grads = np.array(grad_list)
        if grads.shape[0] == 1:
            grads_ = np.empty(len(grads), dtype=object)
            grads_[:] = list(grads)
            grads = grads_
        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)
        if x.dtype != object:
            grads = np.array([i for i in grads], dtype=x.dtype)
            assert grads.shape == x.shape and grads.dtype == x.dtype
        return grads

    def _preprocess(self, img: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            return 10
        '\n        Preprocess image before forward pass, this is the same preprocessing used during training, please refer to\n        collate function in train.py for reference\n\n        :param img: Single frame od shape (nb_samples, height, width, nb_channels).\n        :return: Preprocessed frame.\n        '
        import torch
        from torch.nn.functional import interpolate
        from art.preprocessing.standardisation_mean_std.pytorch import StandardisationMeanStdPyTorch
        if self.preprocessing is not None and isinstance(self.preprocessing, StandardisationMeanStdPyTorch):
            mean_np = self.preprocessing.mean
            std_np = self.preprocessing.std
        else:
            mean_np = np.ones((3, 1, 1))
            std_np = np.ones((3, 1, 1))
        mean = torch.from_numpy(mean_np).reshape((3, 1, 1))
        std = torch.from_numpy(std_np).reshape((3, 1, 1))
        img = img.permute(2, 0, 1)
        img = img * std + mean
        img = torch.unsqueeze(img, dim=0)
        img = interpolate(img, size=(self.input_shape[1], self.input_shape[2]), mode='bicubic')
        if self.clip_values is not None:
            img = torch.clamp(img, float(self.clip_values[0]), float(self.clip_values[1]))
        img = torch.squeeze(img)
        img = (img - mean) / std
        return img

    def _track_step(self, curr_frame: 'torch.Tensor', prev_frame: 'torch.Tensor', rect: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            print('Hello World!')
        '\n        Track current frame.\n\n        :param curr_frame: Current frame.\n        :param prev_frame: Previous frame.\n        :return: bounding box of previous frame\n        '
        import torch
        prev_bbox = rect
        k_context_factor = 2

        def compute_output_height_f(bbox_tight: 'torch.Tensor') -> 'torch.Tensor':
            if False:
                print('Hello World!')
            '\n            Compute height of search/target region.\n\n            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].\n            :return: Output height.\n            '
            bbox_height = bbox_tight[3] - bbox_tight[1]
            output_height = k_context_factor * bbox_height
            return torch.maximum(torch.tensor(1.0).to(self.device), output_height)

        def compute_output_width_f(bbox_tight: 'torch.Tensor') -> 'torch.Tensor':
            if False:
                return 10
            '\n            Compute width of search/target region.\n\n            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].\n            :return: Output width.\n            '
            bbox_width = bbox_tight[2] - bbox_tight[0]
            output_width = k_context_factor * bbox_width
            return torch.maximum(torch.tensor(1.0).to(self.device), output_width)

        def get_center_x_f(bbox_tight: 'torch.Tensor') -> 'torch.Tensor':
            if False:
                print('Hello World!')
            '\n            Compute x-coordinate of the bounding box center.\n\n            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].\n            :return: x-coordinate of the bounding box center.\n            '
            return (bbox_tight[0] + bbox_tight[2]) / 2.0

        def get_center_y_f(bbox_tight: 'torch.Tensor') -> 'torch.Tensor':
            if False:
                print('Hello World!')
            '\n            Compute y-coordinate of the bounding box center\n\n            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].\n            :return: y-coordinate of the bounding box center.\n            '
            return (bbox_tight[1] + bbox_tight[3]) / 2.0

        def compute_crop_pad_image_location(bbox_tight: 'torch.Tensor', image: 'torch.Tensor') -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
            if False:
                i = 10
                return i + 15
            '\n            Get the valid image coordinates for the context region in target or search region in full image\n\n            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].\n            :param image: Frame to be cropped and padded.\n            :return: x-coordinate of the bounding box center.\n            '
            bbox_center_x = get_center_x_f(bbox_tight)
            bbox_center_y = get_center_y_f(bbox_tight)
            image_height = image.shape[0]
            image_width = image.shape[1]
            output_width = compute_output_width_f(bbox_tight)
            output_height = compute_output_height_f(bbox_tight)
            roi_left = torch.maximum(torch.tensor(0.0).to(self.device), bbox_center_x - output_width / 2.0)
            roi_bottom = torch.maximum(torch.tensor(0.0).to(self.device), bbox_center_y - output_height / 2.0)
            left_half = torch.minimum(output_width / 2.0, bbox_center_x)
            right_half = torch.minimum(output_width / 2.0, image_width - bbox_center_x)
            roi_width = torch.maximum(torch.tensor(1.0).to(self.device), left_half + right_half)
            top_half = torch.minimum(output_height / 2.0, bbox_center_y)
            bottom_half = torch.minimum(output_height / 2.0, image_height - bbox_center_y)
            roi_height = torch.maximum(torch.tensor(1.0).to(self.device), top_half + bottom_half)
            return (roi_left, roi_bottom, roi_left + roi_width, roi_bottom + roi_height)

        def edge_spacing_x_f(bbox_tight: 'torch.Tensor') -> 'torch.Tensor':
            if False:
                for i in range(10):
                    print('nop')
            '\n            Edge spacing X to take care of if search/target pad region goes out of bound.\n\n            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].\n            :return: Edge spacing X.\n            '
            output_width = compute_output_width_f(bbox_tight)
            bbox_center_x = get_center_x_f(bbox_tight)
            return torch.maximum(torch.tensor(0.0).to(self.device), output_width / 2 - bbox_center_x)

        def edge_spacing_y_f(bbox_tight: 'torch.Tensor') -> 'torch.Tensor':
            if False:
                i = 10
                return i + 15
            '\n            Edge spacing X to take care of if search/target pad region goes out of bound.\n\n            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].\n            :return: Edge spacing X.\n            '
            output_height = compute_output_height_f(bbox_tight)
            bbox_center_y = get_center_y_f(bbox_tight)
            return torch.maximum(torch.tensor(0.0).to(self.device), output_height / 2 - bbox_center_y)

        def crop_pad_image(bbox_tight: 'torch.Tensor', image: 'torch.Tensor') -> Tuple['torch.Tensor', Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor', 'torch.Tensor'], 'torch.Tensor', 'torch.Tensor']:
            if False:
                return 10
            '\n            Around the bounding box, we define a extra context factor of 2, which we will crop from the original image.\n\n            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].\n            :param image: Frame to be cropped and padded.\n            :return: Cropped and Padded image.\n            '
            import math
            import torch
            pad_image_location = compute_crop_pad_image_location(bbox_tight, image)
            roi_left = torch.minimum(pad_image_location[0], torch.tensor(image.shape[1] - 1).to(self.device))
            roi_bottom = torch.minimum(pad_image_location[1], torch.tensor(image.shape[0] - 1).to(self.device))
            roi_width = min(image.shape[1], max(1, math.ceil(pad_image_location[2] - pad_image_location[0])))
            roi_height = min(image.shape[0], max(1, math.ceil(pad_image_location[3] - pad_image_location[1])))
            roi_bottom_int = int(roi_bottom)
            roi_bottom_height_int = roi_bottom_int + roi_height
            roi_left_int = int(roi_left)
            roi_left_width_int = roi_left_int + roi_width
            cropped_image = image[roi_bottom_int:roi_bottom_height_int, roi_left_int:roi_left_width_int]
            output_width = max(math.ceil(compute_output_width_f(bbox_tight)), roi_width)
            output_height = max(math.ceil(compute_output_height_f(bbox_tight)), roi_height)
            if image.ndim > 2:
                output_image = torch.zeros((int(output_height), int(output_width), image.shape[2]), dtype=image.dtype)
            else:
                output_image = torch.zeros((int(output_height), int(output_width)), dtype=image.dtype)
            edge_spacing_x = torch.minimum(edge_spacing_x_f(bbox_tight), torch.tensor(image.shape[1] - 1))
            edge_spacing_y = torch.minimum(edge_spacing_y_f(bbox_tight), torch.tensor(image.shape[0] - 1))
            output_image[int(edge_spacing_y):int(edge_spacing_y) + cropped_image.shape[0], int(edge_spacing_x):int(edge_spacing_x) + cropped_image.shape[1]] = cropped_image
            return (output_image, pad_image_location, edge_spacing_x, edge_spacing_y)
        (target_pad, _, _, _) = crop_pad_image(prev_bbox, prev_frame)
        (cur_search_region, search_location, edge_spacing_x, edge_spacing_y) = crop_pad_image(prev_bbox, curr_frame)
        target_pad_in = self._preprocess(target_pad).unsqueeze(0).to(self.device)
        cur_search_region_in = self._preprocess(cur_search_region).unsqueeze(0).to(self.device)
        pred_bb = self._model.forward(target_pad_in.float(), cur_search_region_in.float())
        pred_bb = torch.squeeze(pred_bb)
        k_scale_factor = 10
        height = cur_search_region.shape[0]
        width = cur_search_region.shape[1]
        pred_bb[0] = pred_bb[0] / k_scale_factor * width
        pred_bb[2] = pred_bb[2] / k_scale_factor * width
        pred_bb[1] = pred_bb[1] / k_scale_factor * height
        pred_bb[3] = pred_bb[3] / k_scale_factor * height
        raw_image = curr_frame
        pred_bb[0] = max(0.0, pred_bb[0] + search_location[0] - edge_spacing_x)
        pred_bb[1] = max(0.0, pred_bb[1] + search_location[1] - edge_spacing_y)
        pred_bb[2] = min(raw_image.shape[1], pred_bb[2] + search_location[0] - edge_spacing_x)
        pred_bb[3] = min(raw_image.shape[0], pred_bb[3] + search_location[1] - edge_spacing_y)
        return pred_bb

    def _track(self, x: 'torch.Tensor', y_init: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            for i in range(10):
                print('nop')
        '\n        Track object across frames.\n\n        :param x: A single video of shape (nb_frames, nb_height, nb_width, nb_channels)\n        :param y_init: Initial bounding box around object on the first frame of `x`.\n        :return: Predicted bounding box coordinates for all frames of shape (nb_frames, 4) in format [x1, y1, x2, y2].\n        '
        import torch
        num_frames = x.shape[0]
        prev = x[0]
        bbox_0 = y_init
        y_pred_list = [y_init]
        for i in range(1, num_frames):
            curr = x[i]
            bbox_0 = self._track_step(curr, prev, bbox_0)
            bbox = bbox_0
            prev = curr
            y_pred_list.append(bbox)
        y_pred = torch.stack(y_pred_list)
        return y_pred

    def predict(self, x: np.ndarray, batch_size: int=128, **kwargs) -> List[Dict[str, np.ndarray]]:
        if False:
            i = 10
            return i + 15
        '\n        Perform prediction for a batch of inputs.\n\n        :param x: Samples of shape (nb_samples, nb_frames, height, width, nb_channels).\n        :param batch_size: Batch size.\n\n        :Keyword Arguments:\n            * *y_init* (``np.ndarray``) --\n              Initial box around object to be tracked as [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and\n              0 <= y1 < y2 <= H.\n\n        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one dictionary for each input image. The keys of\n                 the dictionary are:\n\n                  - boxes [N_FRAMES, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and\n                                         0 <= y1 < y2 <= H.\n                  - labels [N_FRAMES]: the labels for each image, default 0.\n                  - scores [N_FRAMES]: the scores or each prediction, default 1.\n        '
        import torch
        self._model.eval()
        if hasattr(self._model, 'freeze'):
            self._model.freeze()
        y_init = kwargs.get('y_init')
        if y_init is None:
            raise ValueError('y_init is a required argument for method `predict`.')
        if isinstance(y_init, np.ndarray):
            y_init = torch.from_numpy(y_init).to(self.device).float()
        else:
            y_init = y_init.to(self.device).float()
        predictions = []
        for i in range(x.shape[0]):
            if isinstance(x, np.ndarray):
                x_i = torch.from_numpy(x[i]).to(self.device)
            else:
                x_i = x[i].to(self.device)
            x_i = torch.unsqueeze(x_i, dim=0)
            (x_i, _) = self._apply_preprocessing(x_i, y=None, fit=False, no_grad=False)
            x_i = torch.squeeze(x_i)
            y_pred = self._track(x=x_i, y_init=y_init[i])
            prediction_dict = {}
            if isinstance(x, np.ndarray):
                prediction_dict['boxes'] = y_pred.detach().cpu().numpy()
            else:
                prediction_dict['boxes'] = y_pred
            predictions.append(prediction_dict)
        return predictions

    def fit(self, x: np.ndarray, y, batch_size: int=128, nb_epochs: int=20, **kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        Not implemented.\n        '
        raise NotImplementedError

    def get_activations(self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool=False) -> np.ndarray:
        if False:
            return 10
        '\n        Not implemented.\n        '
        raise NotImplementedError

    def compute_losses(self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, 'torch.Tensor']]]) -> Dict[str, np.ndarray]:
        if False:
            while True:
                i = 10
        '\n        Compute losses.\n\n        :param x: Samples of shape (nb_samples, nb_frames, height, width, nb_channels).\n        :param y: Target values of format `List[Dict[str, np.ndarray]]`, one dictionary for each input image. The keys\n                  of the dictionary are:\n\n                  - boxes [N_FRAMES, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and\n                                         0 <= y1 < y2 <= H.\n        :return: Dictionary of loss components.\n        '
        output = self.compute_loss(x=x, y=y)
        output_dict = {}
        output_dict['torch.nn.L1Loss'] = output
        return output_dict

    def compute_loss(self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, 'torch.Tensor']]], **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute loss.\n\n        :param x: Samples of shape (nb_samples, nb_frames, height, width, nb_channels).\n        :param y: Target values of format `List[Dict[str, np.ndarray]]`, one dictionary for each input image. The keys\n                  of the dictionary are:\n\n                  - boxes [N_FRAMES, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and\n                                         0 <= y1 < y2 <= H.\n        :return: Total loss.\n        '
        import torch
        (output_dict, _, _) = self._get_losses(x=x, y=y)
        if isinstance(output_dict['torch.nn.L1Loss'], list):
            output_list = []
            for out in output_dict['torch.nn.L1Loss']:
                output_list.append(out.detach().cpu().numpy())
            output = np.array(output_list)
        elif isinstance(output_dict['torch.nn.L1Loss'], torch.Tensor):
            output = output_dict['torch.nn.L1Loss'].detach().cpu().numpy()
        else:
            output = np.array(output_dict['torch.nn.L1Loss'])
        return output

    def init(self, image: 'PIL.JpegImagePlugin.JpegImageFile', box: np.ndarray):
        if False:
            while True:
                i = 10
        '\n        Method `init` for GOT-10k trackers.\n\n        :param image: Current image.\n        :return: Predicted box.\n        '
        import torch
        self.prev = np.array(image) / 255.0
        if self.clip_values is not None:
            self.prev = self.prev * self.clip_values[1]
        self.box = torch.from_numpy(np.array([box[0], box[1], box[2] + box[0], box[3] + box[1]])).to(self.device)

    def update(self, image: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Method `update` for GOT-10k trackers.\n\n        :param image: Current image.\n        :return: Predicted box.\n        '
        import torch
        curr = torch.from_numpy(np.array(image) / 255.0)
        if self.clip_values is not None:
            curr = curr * self.clip_values[1]
        curr = curr.to(self.device)
        prev = torch.from_numpy(self.prev).to(self.device)
        (curr, _) = self._apply_preprocessing(curr, y=None, fit=False)
        self.box = self._track_step(curr, prev, self.box)
        self.prev = curr.cpu().detach().numpy()
        box_return = self.box.cpu().detach().numpy()
        box_return = np.array([box_return[0], box_return[1], box_return[2] - box_return[0], box_return[3] - box_return[1]])
        return box_return

    def track(self, img_files: List[str], box: np.ndarray, visualize: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            return 10
        '\n        Method `track` for GOT-10k toolkit trackers (MIT licence).\n\n        :param img_files: Image files.\n        :param box: Initial boxes.\n        :param visualize: Visualise tracking.\n        '
        from got10k.utils.viz import show_frame
        from PIL import Image
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)
        for (i_f, img_file) in enumerate(img_files):
            image = Image.open(img_file)
            if not image.mode == 'RGB':
                image = image.convert('RGB')
            start_time = time.time()
            if i_f == 0:
                self.init(image, box)
            else:
                boxes[i_f, :] = self.update(image)
            times[i_f] = time.time() - start_time
            if visualize:
                show_frame(image, boxes[i_f, :])
        return (boxes, times)