"""
This module implements the BadDet Object Generation Attack (OGA) on object detectors.

| Paper link: https://arxiv.org/abs/2205.14497
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Dict, List, Tuple, Union
import numpy as np
from tqdm.auto import tqdm
from art.attacks.attack import PoisoningAttackObjectDetector
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
logger = logging.getLogger(__name__)

class BadDetObjectGenerationAttack(PoisoningAttackObjectDetector):
    """
    Implementation of the BadDet Object Generation Attack.

    | Paper link: https://arxiv.org/abs/2205.14497
    """
    attack_params = PoisoningAttackObjectDetector.attack_params + ['backdoor', 'bbox_height', 'bbox_width', 'class_target', 'percent_poison', 'channels_first', 'verbose']
    _estimator_requirements = ()

    def __init__(self, backdoor: PoisoningAttackBackdoor, bbox_height: int, bbox_width: int, class_target: int=1, percent_poison: float=0.3, channels_first: bool=False, verbose: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new BadDet Object Generation Attack\n\n        :param backdoor: the backdoor chosen for this attack.\n        :param bbox_height: The height of the false-positive bounding box.\n        :param bbox_width: The width of the false-positive bounding box.\n        :param class_target: The target label to which the poisoned model needs to misclassify.\n        :param percent_poison: The ratio of samples to poison in the source class, with range [0, 1].\n        :param channels_first: Set channels first or last.\n        :param verbose: Show progress bars.\n        '
        super().__init__()
        self.backdoor = backdoor
        self.bbox_height = bbox_height
        self.bbox_width = bbox_width
        self.class_target = class_target
        self.percent_poison = percent_poison
        self.channels_first = channels_first
        self.verbose = verbose
        self._check_params()

    def poison(self, x: Union[np.ndarray, List[np.ndarray]], y: List[Dict[str, np.ndarray]], **kwargs) -> Tuple[Union[np.ndarray, List[np.ndarray]], List[Dict[str, np.ndarray]]]:
        if False:
            return 10
        '\n        Generate poisoning examples by inserting the backdoor onto the input `x` and changing the classification\n        for labels `y`.\n\n        :param x: Sample images of shape `NCHW` or `NHWC` or a list of sample images of any size.\n        :param y: True labels of type `List[Dict[np.ndarray]]`, one dictionary per input image. The keys and values\n                  of the dictionary are:\n\n                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                  - labels [N]: the labels for each image.\n        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.\n        '
        if isinstance(x, np.ndarray):
            x_ndim = len(x.shape)
        else:
            x_ndim = len(x[0].shape) + 1
        if x_ndim != 4:
            raise ValueError('Unrecognized input dimension. BadDet OGA can only be applied to image data.')
        x_poison: Union[np.ndarray, List[np.ndarray]]
        if isinstance(x, np.ndarray):
            x_poison = x.copy()
        else:
            x_poison = [x_i.copy() for x_i in x]
        y_poison: List[Dict[str, np.ndarray]] = []
        for y_i in y:
            target_dict = {k: v.copy() for (k, v) in y_i.items()}
            y_poison.append(target_dict)
        all_indices = np.arange(len(x))
        num_poison = int(self.percent_poison * len(all_indices))
        selected_indices = np.random.choice(all_indices, num_poison, replace=False)
        for i in tqdm(selected_indices, desc='BadDet OGA iteration', disable=not self.verbose):
            image = x_poison[i]
            boxes = y_poison[i]['boxes']
            labels = y_poison[i]['labels']
            if self.channels_first:
                image = np.transpose(image, (1, 2, 0))
            (height, width, _) = image.shape
            y_1 = np.random.randint(0, height - self.bbox_height)
            x_1 = np.random.randint(0, width - self.bbox_width)
            y_2 = y_1 + self.bbox_height
            x_2 = x_1 + self.bbox_width
            bounding_box = image[y_1:y_2, x_1:x_2, :]
            (poisoned_input, _) = self.backdoor.poison(bounding_box[np.newaxis], labels)
            image[y_1:y_2, x_1:x_2, :] = poisoned_input[0]
            if self.channels_first:
                image = np.transpose(image, (2, 0, 1))
            x_poison[i] = image
            y_poison[i]['boxes'] = np.concatenate((boxes, [[x_1, y_1, x_2, y_2]]))
            y_poison[i]['labels'] = np.concatenate((labels, [self.class_target]))
            if 'scores' in y_poison[i]:
                y_poison[i]['scores'] = np.concatenate((y_poison[i]['scores'], [1.0]))
            if 'masks' in y_poison[i]:
                mask = np.zeros_like(image)
                mask[y_1:y_2, x_1:x_2, :] = 1
                y_poison[i]['masks'] = np.concatenate((y_poison[i]['masks'], [mask]))
        return (x_poison, y_poison)

    def _check_params(self) -> None:
        if False:
            while True:
                i = 10
        if not isinstance(self.backdoor, PoisoningAttackBackdoor):
            raise ValueError('Backdoor must be of type PoisoningAttackBackdoor')
        if self.bbox_height <= 0:
            raise ValueError('bbox_height must be positive')
        if self.bbox_width <= 0:
            raise ValueError('bbox_width must be positive')
        if not 0 < self.percent_poison <= 1:
            raise ValueError('percent_poison must be between 0 and 1')