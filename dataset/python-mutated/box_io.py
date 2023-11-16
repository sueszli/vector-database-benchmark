"""Python interface for Boxes proto.

Support read and write of Boxes from/to numpy arrays and file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from delf import box_pb2

def ArraysToBoxes(boxes, scores, class_indices):
    if False:
        i = 10
        return i + 15
    'Converts `boxes` to Boxes proto.\n\n  Args:\n    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,\n      left, bottom, right].\n    scores: [N] float array with detection scores.\n    class_indices: [N] int array with class indices.\n\n  Returns:\n    boxes_proto: Boxes object.\n  '
    num_boxes = len(scores)
    assert num_boxes == boxes.shape[0]
    assert num_boxes == len(class_indices)
    boxes_proto = box_pb2.Boxes()
    for i in range(num_boxes):
        boxes_proto.box.add(ymin=boxes[i, 0], xmin=boxes[i, 1], ymax=boxes[i, 2], xmax=boxes[i, 3], score=scores[i], class_index=class_indices[i])
    return boxes_proto

def BoxesToArrays(boxes_proto):
    if False:
        print('Hello World!')
    'Converts data saved in Boxes proto to numpy arrays.\n\n  If there are no boxes, the function returns three empty arrays.\n\n  Args:\n    boxes_proto: Boxes proto object.\n\n  Returns:\n    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,\n      left, bottom, right].\n    scores: [N] float array with detection scores.\n    class_indices: [N] int array with class indices.\n  '
    num_boxes = len(boxes_proto.box)
    if num_boxes == 0:
        return (np.array([]), np.array([]), np.array([]))
    boxes = np.zeros([num_boxes, 4])
    scores = np.zeros([num_boxes])
    class_indices = np.zeros([num_boxes])
    for i in range(num_boxes):
        box_proto = boxes_proto.box[i]
        boxes[i] = [box_proto.ymin, box_proto.xmin, box_proto.ymax, box_proto.xmax]
        scores[i] = box_proto.score
        class_indices[i] = box_proto.class_index
    return (boxes, scores, class_indices)

def SerializeToString(boxes, scores, class_indices):
    if False:
        while True:
            i = 10
    'Converts numpy arrays to serialized Boxes.\n\n  Args:\n    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,\n      left, bottom, right].\n    scores: [N] float array with detection scores.\n    class_indices: [N] int array with class indices.\n\n  Returns:\n    Serialized Boxes string.\n  '
    boxes_proto = ArraysToBoxes(boxes, scores, class_indices)
    return boxes_proto.SerializeToString()

def ParseFromString(string):
    if False:
        while True:
            i = 10
    'Converts serialized Boxes proto string to numpy arrays.\n\n  Args:\n    string: Serialized Boxes string.\n\n  Returns:\n    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,\n      left, bottom, right].\n    scores: [N] float array with detection scores.\n    class_indices: [N] int array with class indices.\n  '
    boxes_proto = box_pb2.Boxes()
    boxes_proto.ParseFromString(string)
    return BoxesToArrays(boxes_proto)

def ReadFromFile(file_path):
    if False:
        i = 10
        return i + 15
    'Helper function to load data from a Boxes proto format in a file.\n\n  Args:\n    file_path: Path to file containing data.\n\n  Returns:\n    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,\n      left, bottom, right].\n    scores: [N] float array with detection scores.\n    class_indices: [N] int array with class indices.\n  '
    with tf.gfile.GFile(file_path, 'rb') as f:
        return ParseFromString(f.read())

def WriteToFile(file_path, boxes, scores, class_indices):
    if False:
        for i in range(10):
            print('nop')
    'Helper function to write data to a file in Boxes proto format.\n\n  Args:\n    file_path: Path to file that will be written.\n    boxes: [N, 4] float array denoting bounding box coordinates, in format [top,\n      left, bottom, right].\n    scores: [N] float array with detection scores.\n    class_indices: [N] int array with class indices.\n  '
    serialized_data = SerializeToString(boxes, scores, class_indices)
    with tf.gfile.GFile(file_path, 'w') as f:
        f.write(serialized_data)