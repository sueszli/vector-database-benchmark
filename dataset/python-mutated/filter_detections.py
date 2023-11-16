"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import tensorflow
from tensorflow import keras
from .. import backend

def filter_detections(boxes, classification, other=[], class_specific_filter=True, nms=True, score_threshold=0.05, max_detections=300, nms_threshold=0.5):
    if False:
        print('Hello World!')
    " Filter detections using the boxes and classification values.\n\n    Args\n        boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.\n        classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.\n        other                 : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.\n        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.\n        nms                   : Flag to enable/disable non maximum suppression.\n        score_threshold       : Threshold used to prefilter the boxes with.\n        max_detections        : Maximum number of detections to keep.\n        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.\n\n    Returns\n        A list of [boxes, scores, labels, other[0], other[1], ...].\n        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.\n        scores is shaped (max_detections,) and contains the scores of the predicted class.\n        labels is shaped (max_detections,) and contains the predicted label.\n        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.\n        In case there are less than max_detections detections, the tensors are padded with -1's.\n    "

    def _filter_detections(scores, labels):
        if False:
            for i in range(10):
                print('nop')
        indices = tensorflow.where(keras.backend.greater(scores, score_threshold))
        if nms:
            filtered_boxes = tensorflow.gather_nd(boxes, indices)
            filtered_scores = keras.backend.gather(scores, indices)[:, 0]
            nms_indices = tensorflow.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)
            indices = keras.backend.gather(indices, nms_indices)
        labels = tensorflow.gather_nd(labels, indices)
        indices = keras.backend.stack([indices[:, 0], labels], axis=1)
        return indices
    if class_specific_filter:
        all_indices = []
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tensorflow.ones((keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))
        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores = keras.backend.max(classification, axis=1)
        labels = keras.backend.argmax(classification, axis=1)
        indices = _filter_detections(scores, labels)
    scores = tensorflow.gather_nd(classification, indices)
    labels = indices[:, 1]
    (scores, top_indices) = tensorflow.nn.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))
    indices = keras.backend.gather(indices[:, 0], top_indices)
    boxes = keras.backend.gather(boxes, indices)
    labels = keras.backend.gather(labels, top_indices)
    other_ = [keras.backend.gather(o, indices) for o in other]
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes = tensorflow.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tensorflow.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tensorflow.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = keras.backend.cast(labels, 'int32')
    other_ = [tensorflow.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for (o, s) in zip(other_, [list(keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])
    return [boxes, scores, labels] + other_

class FilterDetections(keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(self, nms=True, class_specific_filter=True, nms_threshold=0.5, score_threshold=0.05, max_detections=300, parallel_iterations=32, **kwargs):
        if False:
            i = 10
            return i + 15
        ' Filters detections using score threshold, NMS and selecting the top-k detections.\n\n        Args\n            nms                   : Flag to enable/disable NMS.\n            class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.\n            nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.\n            score_threshold       : Threshold used to prefilter the boxes with.\n            max_detections        : Maximum number of detections to keep.\n            parallel_iterations   : Number of batch items to process in parallel.\n        '
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        if False:
            return 10
        ' Constructs the NMS graph.\n\n        Args\n            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.\n        '
        boxes = inputs[0]
        classification = inputs[1]
        other = inputs[2:]

        def _filter_detections(args):
            if False:
                while True:
                    i = 10
            boxes = args[0]
            classification = args[1]
            other = args[2]
            return filter_detections(boxes, classification, other, nms=self.nms, class_specific_filter=self.class_specific_filter, score_threshold=self.score_threshold, max_detections=self.max_detections, nms_threshold=self.nms_threshold)
        dtypes = [keras.backend.floatx(), keras.backend.floatx(), 'int32'] + [o.dtype for o in other]
        shapes = [(self.max_detections, 4), (self.max_detections,), (self.max_detections,)]
        shapes.extend([(self.max_detections,) + o.shape[2:] for o in other])
        outputs = backend.map_fn(_filter_detections, elems=[boxes, classification, other], dtype=dtypes, shapes=shapes, parallel_iterations=self.parallel_iterations)
        return outputs

    def compute_output_shape(self, input_shape):
        if False:
            for i in range(10):
                print('nop')
        ' Computes the output shapes given the input shapes.\n\n        Args\n            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].\n\n        Returns\n            List of tuples representing the output shapes:\n            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]\n        '
        return [(input_shape[0][0], self.max_detections, 4), (input_shape[1][0], self.max_detections), (input_shape[1][0], self.max_detections)] + [tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))]

    def compute_mask(self, inputs, mask=None):
        if False:
            while True:
                i = 10
        ' This is required in Keras when there is more than 1 output.\n        '
        return (len(inputs) + 1) * [None]

    def get_config(self):
        if False:
            print('Hello World!')
        ' Gets the configuration of this layer.\n\n        Returns\n            Dictionary containing the parameters of this layer.\n        '
        config = super(FilterDetections, self).get_config()
        config.update({'nms': self.nms, 'class_specific_filter': self.class_specific_filter, 'nms_threshold': self.nms_threshold, 'score_threshold': self.score_threshold, 'max_detections': self.max_detections, 'parallel_iterations': self.parallel_iterations})
        return config