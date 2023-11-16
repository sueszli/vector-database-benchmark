"""
This module implements the BadDet Regional Misclassification Attack (RMA) on object detectors.

| Paper link: https://arxiv.org/abs/2205.14497
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from tqdm.auto import tqdm
from art.attacks.attack import PoisoningAttackObjectDetector
from art.attacks.poisoning.backdoor_attack import PoisoningAttackBackdoor
logger = logging.getLogger(__name__)

class BadDetRegionalMisclassificationAttack(PoisoningAttackObjectDetector):
    """
    Implementation of the BadDet Regional Misclassification Attack.

    | Paper link: https://arxiv.org/abs/2205.14497
    """
    attack_params = PoisoningAttackObjectDetector.attack_params + ['backdoor', 'class_source', 'class_target', 'percent_poison', 'channels_first', 'verbose']
    _estimator_requirements = ()

    def __init__(self, backdoor: PoisoningAttackBackdoor, class_source: Optional[int]=None, class_target: int=1, percent_poison: float=0.3, channels_first: bool=False, verbose: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new BadDet Regional Misclassification Attack\n\n        :param backdoor: the backdoor chosen for this attack.\n        :param class_source: The source class (optionally) from which triggers were selected. If no source is\n                             provided, then all classes will be poisoned.\n        :param class_target: The target label to which the poisoned model needs to misclassify.\n        :param percent_poison: The ratio of samples to poison in the source class, with range [0, 1].\n        :param channels_first: Set channels first or last.\n        :param verbose: Show progress bars.\n        '
        super().__init__()
        self.backdoor = backdoor
        self.class_source = class_source
        self.class_target = class_target
        self.percent_poison = percent_poison
        self.channels_first = channels_first
        self.verbose = verbose
        self._check_params()

    def poison(self, x: Union[np.ndarray, List[np.ndarray]], y: List[Dict[str, np.ndarray]], **kwargs) -> Tuple[Union[np.ndarray, List[np.ndarray]], List[Dict[str, np.ndarray]]]:
        if False:
            while True:
                i = 10
        '\n        Generate poisoning examples by inserting the backdoor onto the input `x` and changing the classification\n        for labels `y`.\n\n        :param x: Sample images of shape `NCHW` or `NHWC` or a list of sample images of any size.\n        :param y: True labels of type `List[Dict[np.ndarray]]`, one dictionary per input image. The keys and values\n                  of the dictionary are:\n\n                  - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.\n                  - labels [N]: the labels for each image.\n        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.\n        '
        if isinstance(x, np.ndarray):
            x_ndim = len(x.shape)
        else:
            x_ndim = len(x[0].shape) + 1
        if x_ndim != 4:
            raise ValueError('Unrecognized input dimension. BadDet RMA can only be applied to image data.')
        x_poison: Union[np.ndarray, List[np.ndarray]]
        if isinstance(x, np.ndarray):
            x_poison = x.copy()
        else:
            x_poison = [x_i.copy() for x_i in x]
        y_poison: List[Dict[str, np.ndarray]] = []
        source_indices = []
        for (i, y_i) in enumerate(y):
            target_dict = {k: v.copy() for (k, v) in y_i.items()}
            y_poison.append(target_dict)
            if self.class_source is None or self.class_source in y_i['labels']:
                source_indices.append(i)
        num_poison = int(self.percent_poison * len(source_indices))
        selected_indices = np.random.choice(source_indices, num_poison, replace=False)
        for i in tqdm(selected_indices, desc='BadDet RMA iteration', disable=not self.verbose):
            image = x_poison[i]
            boxes = y_poison[i]['boxes']
            labels = y_poison[i]['labels']
            if self.channels_first:
                image = np.transpose(image, (1, 2, 0))
            for (j, (box, label)) in enumerate(zip(boxes, labels)):
                if self.class_source is None or label == self.class_source:
                    (x_1, y_1, x_2, y_2) = box.astype(int)
                    bounding_box = image[y_1:y_2, x_1:x_2, :]
                    (poisoned_input, _) = self.backdoor.poison(bounding_box[np.newaxis], label)
                    image[y_1:y_2, x_1:x_2, :] = poisoned_input[0]
                    labels[j] = self.class_target
            if self.channels_first:
                image = np.transpose(image, (2, 0, 1))
            x_poison[i] = image
        return (x_poison, y_poison)

    def _check_params(self) -> None:
        if False:
            print('Hello World!')
        if not isinstance(self.backdoor, PoisoningAttackBackdoor):
            raise ValueError('Backdoor must be of type PoisoningAttackBackdoor')
        if not 0 < self.percent_poison <= 1:
            raise ValueError('percent_poison must be between 0 and 1')