"""
This module implements the task specific estimator for Faster R-CNN in TensorFlowV2.
"""
import logging
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.tensorflow import TensorFlowV2Estimator
from art.utils import get_file
from art import config
if TYPE_CHECKING:
    import tensorflow as tf
    from object_detection.meta_architectures.faster_rcnn_meta_arch import FasterRCNNMetaArch
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor
logger = logging.getLogger(__name__)

class TensorFlowV2FasterRCNN(ObjectDetectorMixin, TensorFlowV2Estimator):
    """
    This class implements a model-specific object detector using Faster-RCNN and TensorFlowV2.
    """
    estimator_params = TensorFlowV2Estimator.estimator_params + ['images', 'is_training', 'attack_losses']

    def __init__(self, input_shape: Tuple[int, ...], model: Optional['FasterRCNNMetaArch']=None, filename: Optional[str]=None, url: Optional[str]=None, is_training: bool=False, clip_values: Optional['CLIP_VALUES_TYPE']=None, channels_first: bool=False, preprocessing_defences: Union['Preprocessor', List['Preprocessor'], None]=None, postprocessing_defences: Union['Postprocessor', List['Postprocessor'], None]=None, preprocessing: 'PREPROCESSING_TYPE'=(0.0, 1.0), attack_losses: Tuple[str, ...]=('Loss/RPNLoss/localization_loss', 'Loss/RPNLoss/objectness_loss', 'Loss/BoxClassifierLoss/localization_loss', 'Loss/BoxClassifierLoss/classification_loss')):
        if False:
            print('Hello World!')
        '\n        Initialization of an instance TensorFlowV2FasterRCNN.\n\n        :param input_shape: A Tuple indicating input shape in form (height, width, channels)\n        :param model: A TensorFlowV2 Faster-RCNN model. The output that can be computed from the model includes a tuple\n                      of (predictions, losses, detections):\n                        - predictions: a dictionary holding "raw" prediction tensors.\n                        - losses: a dictionary mapping loss keys (`Loss/RPNLoss/localization_loss`,\n                                  `Loss/RPNLoss/objectness_loss`, `Loss/BoxClassifierLoss/localization_loss`,\n                                  `Loss/BoxClassifierLoss/classification_loss`) to scalar tensors representing\n                                  corresponding loss values.\n                        - detections: a dictionary containing final detection results.\n        :param filename: Filename of the detection model without filename extension.\n        :param url: URL to download archive of detection model including filename extension.\n        :param is_training: A boolean indicating whether the training version of the computation graph should be\n                            constructed.\n        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and\n                            maximum values allowed for input image features. If floats are provided, these will be\n                            used as the range of all features. If arrays are provided, each value will be considered\n                            the bound for a feature, thus the shape of clip values needs to match the total number\n                            of features.\n        :param channels_first: Set channels first or last.\n        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.\n        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.\n        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be\n                              used for data preprocessing. The first value will be subtracted from the input. The\n                              input will then be divided by the second one.\n        :param attack_losses: Tuple of any combination of strings of the following loss components:\n                              `first_stage_localization_loss`, `first_stage_objectness_loss`,\n                              `second_stage_localization_loss`, `second_stage_classification_loss`.\n        '
        if model is None:
            if filename is None or url is None:
                (filename, url) = ('faster_rcnn_resnet50_v1_640x640_coco17_tpu-8', 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz')
            model = self._load_model(filename=filename, url=url, is_training=is_training)
        super().__init__(model=model, clip_values=clip_values, channels_first=channels_first, preprocessing_defences=preprocessing_defences, postprocessing_defences=postprocessing_defences, preprocessing=preprocessing)
        if self.clip_values is not None:
            if not np.all(self.clip_values[0] == 0):
                raise ValueError('This estimator supports input images with clip_vales=(0, 255).')
            if not np.all(self.clip_values[1] == 255):
                raise ValueError('This estimator supports input images with clip_vales=(0, 255).')
        if self.preprocessing_defences is not None:
            raise ValueError('This estimator does not support `preprocessing_defences`.')
        if self.postprocessing_defences is not None:
            raise ValueError('This estimator does not support `postprocessing_defences`.')
        self._input_shape: Tuple[int, ...] = input_shape
        self._detections: List[Dict[str, np.ndarray]] = []
        self._predictions: List[np.ndarray] = []
        self._losses: Dict[str, np.ndarray] = {}
        self.is_training: bool = is_training
        self.attack_losses: Tuple[str, ...] = attack_losses

    @property
    def native_label_is_pytorch_format(self) -> bool:
        if False:
            return 10
        '\n        Are the native labels in PyTorch format [x1, y1, x2, y2]?\n        '
        return False

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if False:
            i = 10
            return i + 15
        '\n        Return the shape of one input sample.\n\n        :return: Shape of one input sample.\n        '
        return self._input_shape

    @staticmethod
    def _load_model(filename: Optional[str]=None, url: Optional[str]=None, is_training: bool=False) -> Tuple[Dict[str, 'tf.Tensor'], ...]:
        if False:
            return 10
        '\n        Download, extract and load a model from a URL if it is not already in the cache. The file indicated by `url`\n        is downloaded to the path ~/.art/data and given the name `filename`. Files in tar, tar.gz, tar.bz, and zip\n        formats will also be extracted. Then the model is loaded, pipelined and its outputs are returned as a tuple\n        of (predictions, losses, detections).\n\n        :param filename: Name of the file.\n        :param url: Download URL.\n        :param is_training: A boolean indicating whether the training version of the computation graph should be\n                            constructed.\n        :return: the object detection model restored from checkpoint\n        '
        import tensorflow as tf
        from object_detection.utils import config_util
        from object_detection.builders import model_builder
        if filename is None or url is None:
            raise ValueError('Need input parameters `filename` and `url` to download, extract and load the object detection model.')
        path = get_file(filename=filename, path=config.ART_DATA_PATH, url=url, extract=True)
        pipeline_config = path + '/pipeline.config'
        configs = config_util.get_configs_from_pipeline_file(pipeline_config)
        configs['model'].faster_rcnn.second_stage_batch_size = configs['model'].faster_rcnn.first_stage_max_proposals
        obj_detection_model = model_builder.build(model_config=configs['model'], is_training=is_training, add_summaries=False)
        ckpt = tf.compat.v2.train.Checkpoint(model=obj_detection_model)
        ckpt.restore(path + '/checkpoint/ckpt-0').expect_partial()
        return obj_detection_model

    def loss_gradient(self, x: np.ndarray, y: List[Dict[str, np.ndarray]], standardise_output: bool=False, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Compute the gradient of the loss function w.r.t. `x`.\n\n        :param x: Samples of shape (nb_samples, height, width, nb_channels).\n        :param y: Targets of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict are\n                  as follows:\n\n                 - boxes [N, 4]: the boxes in [y1, x1, y2, x2] in scale [0, 1] (`standardise_output=False`) or\n                                 [x1, y1, x2, y2] in image scale (`standardise_output=True`) format,\n                                 with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                 - labels [N]: the labels for each image in TensorFlow (`standardise_output=False`) or PyTorch\n                               (`standardise_output=True`) format\n\n        :param standardise_output: True if `y` is provided in standardised PyTorch format. Box coordinates will be\n                                   scaled back to [0, 1], label index will be decreased by 1 and the boxes will be\n                                   changed from [x1, y1, x2, y2] to [y1, x1, y2, x2] format, with\n                                   0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n        :return: Loss gradients of the same shape as `x`.\n        '
        import tensorflow as tf
        if self.is_training:
            raise NotImplementedError('This object detector was loaded in training mode and therefore not support loss_gradient.')
        if standardise_output:
            from art.estimators.object_detection.utils import convert_pt_to_tf
            y = convert_pt_to_tf(y=y, height=x.shape[1], width=x.shape[2])
        (x_preprocessed, _) = self._apply_preprocessing(x, y=None, fit=False)
        groundtruth_boxes_list = [tf.convert_to_tensor(y[i]['boxes']) for i in range(x.shape[0])]
        groundtruth_classes_list = [tf.one_hot(groundtruth_class, self._model.num_classes, on_value=1.0, off_value=0.0) for groundtruth_class in [tf.convert_to_tensor(y[i]['labels']) for i in range(x.shape[0])]]
        groundtruth_weights_list = [[1] * len(y[i]['labels']) for i in range(x.shape[0])]
        self._model.provide_groundtruth(groundtruth_boxes_list=groundtruth_boxes_list, groundtruth_classes_list=groundtruth_classes_list, groundtruth_weights_list=groundtruth_weights_list)
        with tf.GradientTape() as tape:
            x_preprocessed = tf.convert_to_tensor(x_preprocessed)
            tape.watch(x_preprocessed)
            (preprocessed_images, true_image_shapes) = self._model.preprocess(x_preprocessed)
            predictions = self._model.predict(preprocessed_images, true_image_shapes)
            losses = self._model.loss(predictions, true_image_shapes)
            loss = None
            for loss_name in self.attack_losses:
                if loss is None:
                    loss = losses[loss_name]
                else:
                    loss = loss + losses[loss_name]
        grads = tape.gradient(loss, x_preprocessed)
        grads = self._apply_preprocessing_gradient(x, grads)
        assert grads.shape == x_preprocessed.shape
        return grads

    def predict(self, x: np.ndarray, batch_size: int=128, standardise_output: bool=False, **kwargs) -> List[Dict[str, np.ndarray]]:
        if False:
            while True:
                i = 10
        '\n        Perform prediction for a batch of inputs.\n\n        :param x: Samples of shape (nb_samples, height, width, nb_channels).\n        :param batch_size: Batch size.\n        :param standardise_output: True if output should be standardised to PyTorch format. Box coordinates will be\n                                   scaled from [0, 1] to image dimensions, label index will be increased by 1 to adhere\n                                   to COCO categories and the boxes will be changed to [x1, y1, x2, y2] format, with\n                                   0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n\n\n        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The\n                 fields of the Dict are as follows:\n\n                 - boxes [N, 4]: the boxes in [y1, x1, y2, x2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                                 Can be changed to PyTorch format with `standardise_output=True`.\n                 - labels [N]: the labels for each image in TensorFlow format. Can be changed to PyTorch format with\n                               `standardise_output=True`.\n                 - scores [N]: the scores or each prediction.\n        '
        import tensorflow as tf
        if self.is_training:
            raise NotImplementedError('This object detector was loaded in training mode and therefore does not support prediction.')
        (x, _) = self._apply_preprocessing(x, y=None, fit=False)
        num_samples = x.shape[0]
        num_batch = int(np.ceil(num_samples / float(batch_size)))
        results = []
        for m in range(num_batch):
            (begin, end) = (m * batch_size, min((m + 1) * batch_size, num_samples))
            (preprocessed_images, true_image_shapes) = self._model.preprocess(tf.convert_to_tensor(x[begin:end]))
            predictions = self._model.predict(preprocessed_images, true_image_shapes)
            batch_results = self._model.postprocess(predictions, true_image_shapes)
            for i in range(end - begin):
                d_sample = {}
                d_sample['boxes'] = batch_results['detection_boxes'][i].numpy()
                d_sample['labels'] = batch_results['detection_classes'][i].numpy().astype(np.int32)
                if standardise_output:
                    from art.estimators.object_detection.utils import convert_tf_to_pt
                    d_sample = convert_tf_to_pt(y=[d_sample], height=x.shape[1], width=x.shape[2])[0]
                d_sample['scores'] = batch_results['detection_scores'][i].numpy()
                results.append(d_sample)
        self._detections = results
        self._predictions = [i['scores'] for i in results]
        return results

    @property
    def predictions(self) -> List[np.ndarray]:
        if False:
            i = 10
            return i + 15
        '\n        Get the `_predictions` attribute.\n\n        :return: A dictionary holding "raw" prediction tensors.\n        '
        return self._predictions

    @property
    def losses(self) -> Dict[str, np.ndarray]:
        if False:
            return 10
        '\n        Get the `_losses` attribute.\n\n        :return: A dictionary mapping loss keys (`Loss/RPNLoss/localization_loss`, `Loss/RPNLoss/objectness_loss`,\n                 `Loss/BoxClassifierLoss/localization_loss`, `Loss/BoxClassifierLoss/classification_loss`) to scalar\n                 tensors representing corresponding loss values.\n        '
        return self._losses

    @property
    def detections(self) -> List[Dict[str, np.ndarray]]:
        if False:
            return 10
        '\n        Get the `_detections` attribute.\n\n        :return: A dictionary containing final detection results.\n        '
        return self._detections

    def fit(self, x: np.ndarray, y, batch_size: int=128, nb_epochs: int=20, **kwargs) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def get_activations(self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool=False) -> np.ndarray:
        if False:
            return 10
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Compute the loss.\n\n        :param x: Sample input with shape as expected by the model.\n        :param y: Targets of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict are\n                  as follows:\n                    - boxes [N, 4]: the boxes in [y1, x1, y2, x2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                    - labels [N]: the labels for each image in TensorFlow format.\n                    - scores [N]: the scores or each prediction.\n        :return: np.float32 representing total loss.\n        '
        import tensorflow as tf
        (x_preprocessed, _) = self._apply_preprocessing(x, y=None, fit=False)
        x_preprocessed = tf.convert_to_tensor(x_preprocessed)
        groundtruth_boxes_list = [tf.convert_to_tensor(y[i]['boxes']) for i in range(x.shape[0])]
        groundtruth_classes_list = [tf.one_hot(groundtruth_class, self._model.num_classes, on_value=1.0, off_value=0.0) for groundtruth_class in [tf.convert_to_tensor(y[i]['labels']) for i in range(x.shape[0])]]
        groundtruth_weights_list = [[1] * len(y[i]['labels']) for i in range(x.shape[0])]
        self._model.provide_groundtruth(groundtruth_boxes_list=groundtruth_boxes_list, groundtruth_classes_list=groundtruth_classes_list, groundtruth_weights_list=groundtruth_weights_list)
        (preprocessed_images, true_image_shapes) = self._model.preprocess(x_preprocessed)
        predictions = self._model.predict(preprocessed_images, true_image_shapes)
        losses = self._model.loss(predictions, true_image_shapes)
        loss = None
        for loss_name in self.attack_losses:
            if loss is None:
                loss = losses[loss_name].numpy()
            else:
                loss = loss + losses[loss_name].numpy()
        total_loss = np.array([loss])
        return total_loss

    def compute_losses(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        if False:
            return 10
        '\n        Compute all loss components.\n\n        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,\n                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).\n        :param y: Targets of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict are\n                  as follows:\n                    - boxes [N, 4]: the boxes in [y1, x1, y2, x2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                    - labels [N]: the labels for each image in TensorFlow format.\n                    - scores [N]: the scores or each prediction.\n        :return: Dictionary of loss components.\n        '
        import tensorflow as tf
        (x_preprocessed, _) = self._apply_preprocessing(x, y=None, fit=False)
        x_preprocessed = tf.convert_to_tensor(x_preprocessed)
        groundtruth_boxes_list = [tf.convert_to_tensor(y[i]['boxes']) for i in range(x.shape[0])]
        groundtruth_classes_list = [tf.one_hot(groundtruth_class, self._model.num_classes, on_value=1.0, off_value=0.0) for groundtruth_class in [tf.convert_to_tensor(y[i]['labels']) for i in range(x.shape[0])]]
        groundtruth_weights_list = [[1] * len(y[i]['labels']) for i in range(x.shape[0])]
        self._model.provide_groundtruth(groundtruth_boxes_list=groundtruth_boxes_list, groundtruth_classes_list=groundtruth_classes_list, groundtruth_weights_list=groundtruth_weights_list)
        (preprocessed_images, true_image_shapes) = self._model.preprocess(x_preprocessed)
        predictions = self._model.predict(preprocessed_images, true_image_shapes)
        losses = self._model.loss(predictions, true_image_shapes)
        for loss_name in self.attack_losses:
            self._losses[loss_name] = losses[loss_name].numpy()
        return self._losses