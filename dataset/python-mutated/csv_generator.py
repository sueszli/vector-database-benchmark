"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
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
from .generator import Generator
from ..utils.image import read_image_bgr
import numpy as np
from PIL import Image
from six import raise_from
import csv
import sys
import os.path
from collections import OrderedDict

def _parse(value, function, fmt):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse a string into a value, and format a nice ValueError if it fails.\n\n    Returns `function(value)`.\n    Any `ValueError` raised is catched and a new `ValueError` is raised\n    with message `fmt.format(e)`, where `e` is the caught `ValueError`.\n    '
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)

def _read_classes(csv_reader):
    if False:
        print('Hello World!')
    ' Parse the classes file given by csv_reader.\n    '
    result = OrderedDict()
    for (line, row) in enumerate(csv_reader):
        line += 1
        try:
            (class_name, class_id) = row
        except ValueError:
            raise_from(ValueError("line {}: format should be 'class_name,class_id'".format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))
        if class_name in result:
            raise ValueError("line {}: duplicate class name: '{}'".format(line, class_name))
        result[class_name] = class_id
    return result

def _read_annotations(csv_reader, classes):
    if False:
        i = 10
        return i + 15
    ' Read annotations from the csv_reader.\n    '
    result = OrderedDict()
    for (line, row) in enumerate(csv_reader):
        line += 1
        try:
            (img_file, x1, y1, x2, y2, class_name) = row[:6]
        except ValueError:
            raise_from(ValueError("line {}: format should be 'img_file,x1,y1,x2,y2,class_name' or 'img_file,,,,,'".format(line)), None)
        if img_file not in result:
            result[img_file] = []
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue
        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))
        if class_name not in classes:
            raise ValueError("line {}: unknown class name: '{}' (classes: {})".format(line, class_name, classes))
        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result

def _open_for_csv(path):
    if False:
        i = 10
        return i + 15
    ' Open a file with flags suitable for csv.reader.\n\n    This is different for python2 it means with mode \'rb\',\n    for python3 this means \'r\' with "universal newlines".\n    '
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')

class CSVGenerator(Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(self, csv_data_file, csv_class_file, base_dir=None, **kwargs):
        if False:
            return 10
        ' Initialize a CSV data generator.\n\n        Args\n            csv_data_file: Path to the CSV annotations file.\n            csv_class_file: Path to the CSV classes file.\n            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).\n        '
        self.image_names = []
        self.image_data = {}
        self.base_dir = base_dir
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_data_file)
        try:
            with _open_for_csv(csv_class_file) as file:
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)
        self.labels = {}
        for (key, value) in self.classes.items():
            self.labels[value] = key
        try:
            with _open_for_csv(csv_data_file) as file:
                self.image_data = _read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        self.image_names = list(self.image_data.keys())
        super(CSVGenerator, self).__init__(**kwargs)

    def size(self):
        if False:
            print('Hello World!')
        ' Size of the dataset.\n        '
        return len(self.image_names)

    def num_classes(self):
        if False:
            print('Hello World!')
        ' Number of classes in the dataset.\n        '
        return max(self.classes.values()) + 1

    def has_label(self, label):
        if False:
            print('Hello World!')
        ' Return True if label is a known label.\n        '
        return label in self.labels

    def has_name(self, name):
        if False:
            i = 10
            return i + 15
        ' Returns True if name is a known class.\n        '
        return name in self.classes

    def name_to_label(self, name):
        if False:
            return 10
        ' Map name to label.\n        '
        return self.classes[name]

    def label_to_name(self, label):
        if False:
            return 10
        ' Map label to name.\n        '
        return self.labels[label]

    def image_path(self, image_index):
        if False:
            print('Hello World!')
        ' Returns the image path for image_index.\n        '
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        if False:
            return 10
        ' Compute the aspect ratio for an image with image_index.\n        '
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        if False:
            print('Hello World!')
        ' Load an image at the image_index.\n        '
        return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        if False:
            i = 10
            return i + 15
        ' Load annotations for an image_index.\n        '
        path = self.image_names[image_index]
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}
        for (idx, annot) in enumerate(self.image_data[path]):
            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
            annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[float(annot['x1']), float(annot['y1']), float(annot['x2']), float(annot['y2'])]]))
        return annotations