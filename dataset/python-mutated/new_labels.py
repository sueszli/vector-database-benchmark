"""Module contains the New Labels check."""
import string
from collections import defaultdict
from secrets import choice
from typing import Dict, Optional
import numpy as np
from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.reduce_classes import ReduceLabelMixin
from deepchecks.utils.strings import format_number, format_percent
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.base_checks import TrainTestCheck
from deepchecks.vision.context import Context
from deepchecks.vision.utils.image_functions import draw_image
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper
__all__ = ['NewLabels']

@docstrings
class NewLabels(TrainTestCheck, ReduceLabelMixin):
    """Detects labels that appear only in the test set.

    Parameters
    ----------
    max_images_to_display_per_label : int , default: 3
        maximum number of images to show from each newly found label in the test set.
    max_new_labels_to_display : int , default: 3
        Maximum number of new labels to display in output.
    {additional_check_init_params:2*indent}
    """

    def __init__(self, max_images_to_display_per_label: int=3, max_new_labels_to_display: int=3, n_samples: Optional[int]=10000, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        if not isinstance(max_images_to_display_per_label, int):
            raise DeepchecksValueError('max_num_images_to_display_per_label must be an integer')
        if not isinstance(max_new_labels_to_display, int):
            raise DeepchecksValueError('max_num_new_labels_to_display must be an integer')
        self.max_images_to_display_per_label = max_images_to_display_per_label
        self.max_new_labels_to_display = max_new_labels_to_display
        self.n_samples = n_samples
        self._display_images = defaultdict()

    def update(self, context: Context, batch: BatchWrapper, dataset_kind):
        if False:
            return 10
        'No additional caching required for this check.'
        if dataset_kind == DatasetKind.TRAIN:
            pass
        data = context.get_data_by_kind(dataset_kind)
        for (image, label, identifier) in zip(batch.numpy_images, batch.numpy_labels, batch.numpy_image_identifiers):
            if data.task_type == TaskType.CLASSIFICATION:
                self._update_images_dict(label, label, image, image_identifier=identifier)
            elif data.task_type == TaskType.OBJECT_DETECTION and len(label) > 0:
                for class_id in np.unique(label[:, 0]):
                    bboxes_of_label = label[label[:, 0] == class_id]
                    self._update_images_dict(bboxes_of_label, class_id, image, image_identifier=identifier)
            elif len(label) > 0:
                raise DeepchecksValueError(f'Unsupported task type {data.task_type.value} for NewLabels check')

    def _update_images_dict(self, label, class_id, image, image_identifier):
        if False:
            while True:
                i = 10
        if class_id not in self._display_images:
            self._display_images[class_id] = {'images': [image], 'labels': [label], 'image_identifiers': [image_identifier]}
        elif len(self._display_images[class_id]['images']) < self.max_images_to_display_per_label:
            self._display_images[class_id]['images'].append(image)
            self._display_images[class_id]['labels'].append(label)
            self._display_images[class_id]['image_identifiers'].append(image_identifier)

    def compute(self, context: Context) -> CheckResult:
        if False:
            return 10
        'Calculate which class_id are only available in the test data set and display them.\n\n        Returns\n        -------\n        CheckResult\n            value: A dictionary showing new class_ids introduced in the test set and number of times they were spotted.\n            display: Images containing said class_ids from the test set.\n        '
        test_data = context.get_data_by_kind(DatasetKind.TEST)
        labels_only_in_test = {key: value for (key, value) in test_data.get_cache()['labels'].items() if key not in context.get_data_by_kind(DatasetKind.TRAIN).get_cache()['labels']}
        labels_only_in_test = dict(sorted(labels_only_in_test.items(), key=lambda item: -item[1]))
        result_value = {'new_labels': labels_only_in_test, 'all_labels_count': sum(test_data.get_cache()['labels'].values())}
        if context.with_display:
            displays = []
            images_per_class = {test_data.label_map[key]: value for (key, value) in self._display_images.items()}
            for (class_name, num_occurrences) in labels_only_in_test.items():
                sid = ''.join([choice(string.ascii_uppercase) for _ in range(3)])
                thumbnail_images = [draw_image(img, labels, test_data.task_type, test_data.label_map) for (img, labels) in zip(images_per_class[class_name]['images'], images_per_class[class_name]['labels'])]
                images_combine = ''.join([f'<div class="{sid}-item">{img}</div>' for img in thumbnail_images])
                html = HTML_TEMPLATE.format(label_name=class_name, images=images_combine, count=format_number(num_occurrences), dataset_name=context.test.name, id=sid)
                displays.append(html)
                if len(displays) == self.max_new_labels_to_display:
                    break
        else:
            displays = None
        return CheckResult(result_value, display=displays)

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        if False:
            for i in range(10):
                print('nop')
        'Reduce check result value.\n\n        Returns\n        -------\n        Dict[str, float]\n            number of samples per each new label\n        '
        return check_result.value['new_labels']

    def greater_is_better(self):
        if False:
            i = 10
            return i + 15
        'Return True if the check reduce_output is better when it is greater.'
        return False

    def add_condition_new_label_ratio_less_or_equal(self, max_allowed_new_labels_ratio: float=0.005):
        if False:
            print('Hello World!')
        '\n        Add condition - Ratio of labels that appear only in the test set required to be less or equal to the threshold.\n\n        Parameters\n        ----------\n        max_allowed_new_labels_ratio: float , default: 0.005\n            the max threshold for percentage of labels that only apper in the test set.\n        '

        def condition(result: Dict) -> ConditionResult:
            if False:
                while True:
                    i = 10
            total_labels_in_test_set = result['all_labels_count']
            new_labels_in_test_set = sum(result['new_labels'].values())
            percent_new_labels = new_labels_in_test_set / total_labels_in_test_set
            if new_labels_in_test_set > 0:
                top_new_class = list(result['new_labels'].keys())[:3]
                message = f'{format_percent(percent_new_labels)} of labels found in test set were not in train set. '
                message += f'New labels most common in test set: {top_new_class}'
            else:
                message = 'No new labels were found in test set.'
            category = ConditionCategory.PASS if percent_new_labels <= max_allowed_new_labels_ratio else ConditionCategory.FAIL
            return ConditionResult(category, message)
        name = f'Percentage of new labels in the test set is less or equal to {format_percent(max_allowed_new_labels_ratio)}'
        return self.add_condition(name, condition)
HTML_TEMPLATE = '\n<style>\n    .{id}-container {{\n        overflow-x: auto;\n        display: flex;\n        flex-direction: column;\n        gap: 10px;\n    }}\n    .{id}-row {{\n      display: flex;\n      flex-direction: row;\n      align-items: center;\n      gap: 10px;\n    }}\n    .{id}-item {{\n      display: flex;\n      min-width: 200px;\n      position: relative;\n      word-wrap: break-word;\n      align-items: center;\n      justify-content: center;\n    }}\n    .{id}-title {{\n        font-family: "Open Sans", verdana, arial, sans-serif;\n        color: #2a3f5f\n    }}\n    /* A fix for jupyter widget which doesn\'t have width defined on HTML widget */\n    .widget-html-content {{\n        width: -moz-available;          /* WebKit-based browsers will ignore this. */\n        width: -webkit-fill-available;  /* Mozilla-based browsers will ignore this. */\n        width: fill-available;\n    }}\n</style>\n<h3><b>Label  "{label_name}"</b></h3>\n<div>\nAppears {count} times in {dataset_name} set.\n</div>\n<div class="{id}-container">\n    <div class="{id}-row">\n        {images}\n    </div>\n</div>\n'