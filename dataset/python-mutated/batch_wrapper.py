"""Contains code for BatchWrapper."""
from typing import Dict, List, Optional, Union
import numpy as np
from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.vision.utils.image_functions import crop_image
from deepchecks.vision.utils.image_properties import calc_default_image_properties, default_image_properties
from deepchecks.vision.utils.vision_properties import PropertiesInputType, calc_vision_properties, validate_properties
from deepchecks.vision.vision_data.utils import BatchOutputFormat, TaskType, sequence_to_numpy
__all__ = ['BatchWrapper']

class BatchWrapper:
    """Represents dataset batch returned by the dataloader during iteration."""

    def __init__(self, batch: BatchOutputFormat, task_type: TaskType, images_seen_num: int):
        if False:
            while True:
                i = 10
        self._task_type = task_type
        self._batch = batch
        (self._labels, self._predictions, self._images) = (None, None, None)
        (self._embeddings, self._additional_data) = (None, None)
        self._image_identifiers = batch.get('image_identifiers')
        if self._image_identifiers is None:
            self._image_identifiers = np.asarray(range(images_seen_num, images_seen_num + len(self)), dtype='str')
        self._vision_properties_cache = dict.fromkeys(PropertiesInputType)

    def _get_relevant_data_for_properties(self, input_type: PropertiesInputType):
        if False:
            i = 10
            return i + 15
        result = []
        if input_type == PropertiesInputType.PARTIAL_IMAGES:
            for (img, bboxes_in_img) in zip(self.numpy_images, self.numpy_labels):
                if bboxes_in_img is None:
                    continue
                result = result + [crop_image(img, *bbox[1:]) for bbox in bboxes_in_img]
        elif input_type == PropertiesInputType.IMAGES:
            result = self.numpy_images
        elif input_type == PropertiesInputType.LABELS:
            result = self.numpy_labels
        elif input_type == PropertiesInputType.PREDICTIONS:
            result = self.numpy_predictions
        return result

    def vision_properties(self, properties_list: Optional[List[Dict]], input_type: PropertiesInputType):
        if False:
            i = 10
            return i + 15
        'Calculate and cache the properties for the batch according to the property input type.\n\n        Parameters\n        ----------\n        properties_list: Optional[List[Dict]]\n            List of properties to calculate. If None, default properties will be calculated.\n        input_type: PropertiesInputType\n            The input type of the properties.\n\n        Returns\n        -------\n        Dict[str, Any]\n            Dictionary of the properties name to list of property values per data element.\n        '
        if self._vision_properties_cache[input_type] is None:
            self._vision_properties_cache[input_type] = {}
        keys_in_cache = self._vision_properties_cache[input_type].keys()
        if properties_list is not None:
            properties_list = validate_properties(properties_list)
            requested_properties_names = [prop['name'] for prop in properties_list]
            properties_to_calc = [p for p in properties_list if p['name'] not in keys_in_cache]
            if len(properties_to_calc) > 0:
                data = self._get_relevant_data_for_properties(input_type)
                self._vision_properties_cache[input_type].update(calc_vision_properties(data, properties_to_calc))
        else:
            if input_type not in [PropertiesInputType.PARTIAL_IMAGES, PropertiesInputType.IMAGES]:
                raise DeepchecksProcessError(f'None was passed to properties calculation for input type {input_type}.')
            requested_properties_names = [prop['name'] for prop in default_image_properties]
            if any((x not in keys_in_cache for x in requested_properties_names)):
                data = self._get_relevant_data_for_properties(input_type)
                self._vision_properties_cache[input_type].update(calc_default_image_properties(data))
        return {key: value for (key, value) in self._vision_properties_cache[input_type].items() if key in requested_properties_names}

    @property
    def original_labels(self):
        if False:
            while True:
                i = 10
        'Return labels for the batch, formatted in deepchecks format.'
        if self._labels is None:
            self._labels = self._batch.get('labels')
        return self._labels

    @property
    def numpy_labels(self) -> List[Union[np.ndarray, int]]:
        if False:
            for i in range(10):
                print('nop')
        'Return labels for the batch in numpy format.'
        required_dim = 0 if self._task_type == TaskType.CLASSIFICATION else 2
        return sequence_to_numpy(self.original_labels, expected_ndim_per_object=required_dim)

    @property
    def original_predictions(self):
        if False:
            return 10
        'Return predictions for the batch, formatted in deepchecks format.'
        if self._predictions is None:
            self._predictions = self._batch.get('predictions')
        return self._predictions

    @property
    def numpy_predictions(self) -> List[np.ndarray]:
        if False:
            i = 10
            return i + 15
        'Return predictions for the batch in numpy format.'
        if self._task_type == TaskType.CLASSIFICATION:
            required_dim = 1
        elif self._task_type == TaskType.OBJECT_DETECTION:
            required_dim = 2
        elif self._task_type == TaskType.SEMANTIC_SEGMENTATION:
            required_dim = 3
        else:
            required_dim = None
        return sequence_to_numpy(self.original_predictions, expected_ndim_per_object=required_dim)

    @property
    def original_images(self):
        if False:
            print('Hello World!')
        'Return images for the batch, formatted in deepchecks format.'
        if self._images is None:
            self._images = self._batch.get('images')
        return self._images

    @property
    def numpy_images(self) -> List[Union[np.ndarray]]:
        if False:
            for i in range(10):
                print('nop')
        'Return images for the batch in numpy format.'
        return sequence_to_numpy(self.original_images, 'uint8', 3)

    @property
    def original_embeddings(self):
        if False:
            while True:
                i = 10
        'Return embedding for the batch, formatted in deepchecks format.'
        if self._embeddings is None:
            self._embeddings = self._batch.get('embeddings')
        return self._embeddings

    @property
    def numpy_embeddings(self) -> List[Union[np.ndarray]]:
        if False:
            for i in range(10):
                print('nop')
        'Return embedding for the batch in numpy format.'
        return sequence_to_numpy(self.original_embeddings, 'float32')

    @property
    def original_additional_data(self):
        if False:
            return 10
        'Return additional data for the batch, formatted in deepchecks format.'
        if self._additional_data is None:
            self._additional_data = self._batch.get('additional_data')
        return self._additional_data

    @property
    def numpy_additional_data(self):
        if False:
            while True:
                i = 10
        'Return additional data for the batch in numpy format.'
        return sequence_to_numpy(self.original_additional_data)

    @property
    def original_image_identifiers(self):
        if False:
            for i in range(10):
                print('nop')
        'Return image identifiers for the batch, formatted in deepchecks format.'
        return self._image_identifiers

    @property
    def numpy_image_identifiers(self) -> List[Union[str, int]]:
        if False:
            i = 10
            return i + 15
        'Return image identifiers for the batch in numpy format.'
        return sequence_to_numpy(self.original_image_identifiers, 'str', 0)

    def __len__(self):
        if False:
            while True:
                i = 10
        'Return length of batch.'
        data = self.numpy_images if self.numpy_images is not None else self.numpy_predictions if self.numpy_predictions is not None else self.numpy_labels if self.numpy_labels is not None else self.numpy_embeddings if self.numpy_embeddings is not None else self.numpy_additional_data
        return len(data)