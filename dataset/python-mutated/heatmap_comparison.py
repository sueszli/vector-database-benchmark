"""Module contains label Drift check."""
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple
import cv2
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from deepchecks.core import CheckResult, DatasetKind
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.base_checks import TrainTestCheck
from deepchecks.vision.context import Context
from deepchecks.vision.utils.image_functions import apply_heatmap_image_properties, numpy_grayscale_to_heatmap_figure
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper
__all__ = ['HeatmapComparison']

@docstrings
class HeatmapComparison(TrainTestCheck):
    """Check if the average image brightness (or bbox location if applicable) is similar between train and test set.

    The check computes the average grayscale image per dataset (train and test) and compares the resulting images.
    Additionally, in case of an object detection task, the check will compare the average locations of the bounding
    boxes between the datasets.

    Parameters
    ----------
    {additional_check_init_params:2*indent}
    """

    def __init__(self, n_samples: Optional[int]=10000, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.n_samples = n_samples

    def initialize_run(self, context: Context):
        if False:
            print('Hello World!')
        'Initialize run.'
        self._task_type = context.train.task_type
        self._grayscale_heatmap = defaultdict(lambda : 0)
        self._shape = None
        self._counter = {}
        self._counter[DatasetKind.TRAIN] = 0
        self._counter[DatasetKind.TEST] = 0
        if self._task_type == TaskType.OBJECT_DETECTION:
            self._bbox_heatmap = defaultdict(lambda : 0)

    def update(self, context: Context, batch: BatchWrapper, dataset_kind):
        if False:
            i = 10
            return i + 15
        'Perform update on batch for train or test counters and histograms.'
        (valid_labels, valid_images) = (batch.numpy_labels, batch.numpy_images)
        if valid_images is not None and len(valid_images) != 0:
            self._counter[dataset_kind] += len(valid_images)
            summed_image = self._grayscale_sum_image(valid_images)
            self._grayscale_heatmap[dataset_kind] += summed_image
            if self._task_type == TaskType.OBJECT_DETECTION:
                label_image_batch = self._label_to_image_batch(valid_labels, valid_images)
                summed_bbox_image = self._grayscale_sum_image(label_image_batch)
                self._bbox_heatmap[dataset_kind] += summed_bbox_image

    def compute(self, context: Context) -> CheckResult:
        if False:
            print('Hello World!')
        'Create the average images and display them.\n\n        Returns\n        -------\n        CheckResult\n            value: The difference images. One for average image brightness, and one for bbox locations if applicable.\n            display: Heatmaps for image brightness (train, test, diff) and heatmap for bbox locations if applicable.\n        '
        train_grayscale = (np.expand_dims(self._grayscale_heatmap[DatasetKind.TRAIN], axis=2) / self._counter[DatasetKind.TRAIN]).astype(np.uint8)
        test_grayscale = (np.expand_dims(self._grayscale_heatmap[DatasetKind.TEST], axis=2) / self._counter[DatasetKind.TEST]).astype(np.uint8)
        value = {'diff': self._image_diff(test_grayscale, train_grayscale)}
        if context.with_display:
            display = [self.plot_row_of_heatmaps(train_grayscale, test_grayscale, 'Compare average image brightness', [context.train.name, context.test.name])]
            display[0].update_layout(coloraxis={'colorscale': 'Inferno', 'cmin': 0, 'cmax': 255}, coloraxis_colorbar={'title': 'Pixel Value'})
        if self._task_type == TaskType.OBJECT_DETECTION:
            train_bbox = (100 * np.expand_dims(self._bbox_heatmap[DatasetKind.TRAIN], axis=2) / self._counter[DatasetKind.TRAIN] / 255).astype(np.uint8)
            test_bbox = (100 * np.expand_dims(self._bbox_heatmap[DatasetKind.TEST], axis=2) / self._counter[DatasetKind.TEST] / 255).astype(np.uint8)
            value['diff_bbox'] = self._image_diff(test_bbox, train_bbox)
            if context.with_display:
                display.append(self.plot_row_of_heatmaps(train_bbox, test_bbox, 'Compare average label bbox locations', [context.train.name, context.test.name]))
                display[1].update_layout(coloraxis={'colorscale': 'Inferno', 'cmin': 0, 'cmax': 100}, coloraxis_colorbar={'title': '% Coverage'})
        return CheckResult(value=value, display=display if context.with_display else None, header='Heatmap Comparison')

    @staticmethod
    def plot_row_of_heatmaps(train_img: np.ndarray, test_img: np.ndarray, title: str, column_titles: List[str]) -> go.Figure:
        if False:
            for i in range(10):
                print('nop')
        'Plot a row of heatmaps for train and test images.'
        fig = make_subplots(rows=1, cols=2, column_titles=column_titles)
        fig.add_trace(numpy_grayscale_to_heatmap_figure(train_img), row=1, col=1)
        fig.add_trace(numpy_grayscale_to_heatmap_figure(test_img), row=1, col=2)
        fig.update_yaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
        fig.update_xaxes(showticklabels=False, visible=True, fixedrange=True, automargin=True)
        fig.update_layout(title=title)
        apply_heatmap_image_properties(fig)
        return fig

    @staticmethod
    def _image_diff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'Return the difference between two grayscale images as a grayscale image.'
        diff = img1.astype(np.int32) - img2.astype(np.int32)
        return np.abs(diff).astype(np.uint8)

    def _label_to_image(self, label: np.ndarray, original_shape: Tuple[int]) -> np.ndarray:
        if False:
            print('Hello World!')
        'Convert label array to an image where pixels inside the bboxes are white and the rest are black.'
        image = np.zeros(original_shape, dtype=np.uint8)
        label = label.reshape((-1, 5))
        x_min = label[:, 1].astype(np.int32)
        y_min = label[:, 2].astype(np.int32)
        x_max = (label[:, 1] + label[:, 3]).astype(np.int32)
        y_max = (label[:, 2] + label[:, 4]).astype(np.int32)
        for i in range(len(label)):
            image[y_min[i]:y_max[i], x_min[i]:x_max[i]] = 255
        return np.expand_dims(image, axis=2)

    def _label_to_image_batch(self, label_batch: List[np.ndarray], image_batch: List[np.ndarray]) -> List[np.ndarray]:
        if False:
            i = 10
            return i + 15
        'Convert label batch to batch of images where pixels inside the bboxes are white and the rest are black.'
        return_bbox_image_batch = []
        for (image, label) in zip(image_batch, label_batch):
            return_bbox_image_batch.append(self._label_to_image(label, image.shape[:2]))
        return return_bbox_image_batch

    def _grayscale_sum_image(self, batch: Iterable[np.ndarray]) -> np.ndarray:
        if False:
            return 10
        'Sum all images in batch to one grayscale image of shape target_shape.\n\n        Parameters\n        ----------\n        batch: np.ndarray\n            batch of images.\n\n        Returns\n        -------\n        np.ndarray\n            summed image.\n        '
        summed_image = None
        for img in batch:
            if img.shape[2] == 1:
                resized_img = img
            elif img.shape[2] == 3:
                resized_img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2GRAY)
            else:
                raise NotImplementedError('Images must be RGB or grayscale')
            if self._shape is None:
                self._shape = resized_img.shape[:2][::-1]
            resized_img = cv2.resize(resized_img.astype('uint8'), self._shape, interpolation=cv2.INTER_AREA)
            if summed_image is None:
                summed_image = resized_img.squeeze().astype(np.int64)
            else:
                summed_image += resized_img.squeeze().astype(np.int64)
        return summed_image