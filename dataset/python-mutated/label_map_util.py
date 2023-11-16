"""Label map utility functions."""
import logging
import tensorflow as tf
from google.protobuf import text_format
import string_int_label_map_pb2

def _validate_label_map(label_map):
    if False:
        return 10
    'Checks if a label map is valid.\n\n  Args:\n    label_map: StringIntLabelMap to validate.\n\n  Raises:\n    ValueError: if label map is invalid.\n  '
    for item in label_map.item:
        if item.id < 0:
            raise ValueError('Label map ids should be >= 0.')
        if item.id == 0 and item.name != 'background' and (item.display_name != 'background'):
            raise ValueError('Label map id 0 is reserved for the background label')

def create_category_index(categories):
    if False:
        return 10
    "Creates dictionary of COCO compatible categories keyed by category id.\n\n  Args:\n    categories: a list of dicts, each of which has the following keys:\n      'id': (required) an integer id uniquely identifying this category.\n      'name': (required) string representing category name\n        e.g., 'cat', 'dog', 'pizza'.\n\n  Returns:\n    category_index: a dict containing the same entries as categories, but keyed\n      by the 'id' field of each category.\n  "
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index

def get_max_label_map_index(label_map):
    if False:
        for i in range(10):
            print('nop')
    'Get maximum index in label map.\n\n  Args:\n    label_map: a StringIntLabelMapProto\n\n  Returns:\n    an integer\n  '
    return max([item.id for item in label_map.item])

def convert_label_map_to_categories(label_map, max_num_classes, use_display_name=True):
    if False:
        while True:
            i = 10
    "Loads label map proto and returns categories list compatible with eval.\n\n  This function loads a label map and returns a list of dicts, each of which\n  has the following keys:\n    'id': (required) an integer id uniquely identifying this category.\n    'name': (required) string representing category name\n      e.g., 'cat', 'dog', 'pizza'.\n  We only allow class into the list if its id-label_id_offset is\n  between 0 (inclusive) and max_num_classes (exclusive).\n  If there are several items mapping to the same id in the label map,\n  we will only keep the first one in the categories list.\n\n  Args:\n    label_map: a StringIntLabelMapProto or None.  If None, a default categories\n      list is created with max_num_classes categories.\n    max_num_classes: maximum number of (consecutive) label indices to include.\n    use_display_name: (boolean) choose whether to load 'display_name' field\n      as category name.  If False or if the display_name field does not exist,\n      uses 'name' field as category names instead.\n  Returns:\n    categories: a list of dictionaries representing all possible categories.\n  "
    categories = []
    list_of_ids_already_added = []
    if not label_map:
        label_id_offset = 1
        for class_id in range(max_num_classes):
            categories.append({'id': class_id + label_id_offset, 'name': 'category_{}'.format(class_id + label_id_offset)})
        return categories
    for item in label_map.item:
        if not 0 < item.id <= max_num_classes:
            logging.info('Ignore item %d since it falls outside of requested label range.', item.id)
            continue
        if use_display_name and item.HasField('display_name'):
            name = item.display_name
        else:
            name = item.name
        if item.id not in list_of_ids_already_added:
            list_of_ids_already_added.append(item.id)
            categories.append({'id': item.id, 'name': name})
    return categories

def load_labelmap(path):
    if False:
        print('Hello World!')
    'Loads label map proto.\n\n  Args:\n    path: path to StringIntLabelMap proto text file.\n  Returns:\n    a StringIntLabelMapProto\n  '
    with tf.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    _validate_label_map(label_map)
    return label_map

def get_label_map_dict(label_map_path, use_display_name=False):
    if False:
        i = 10
        return i + 15
    "Reads a label map and returns a dictionary of label names to id.\n\n  Args:\n    label_map_path: path to label_map.\n    use_display_name: whether to use the label map items' display names as keys.\n\n  Returns:\n    A dictionary mapping label names to id.\n  "
    label_map = load_labelmap(label_map_path)
    label_map_dict = {}
    for item in label_map.item:
        if use_display_name:
            label_map_dict[item.display_name] = item.id
        else:
            label_map_dict[item.name] = item.id
    return label_map_dict

def create_category_index_from_labelmap(label_map_path):
    if False:
        for i in range(10):
            print('nop')
    "Reads a label map and returns a category index.\n\n  Args:\n    label_map_path: Path to `StringIntLabelMap` proto text file.\n\n  Returns:\n    A category index, which is a dictionary that maps integer ids to dicts\n    containing categories, e.g.\n    {1: {'id': 1, 'name': 'dog'}, 2: {'id': 2, 'name': 'cat'}, ...}\n  "
    label_map = load_labelmap(label_map_path)
    max_num_classes = max((item.id for item in label_map.item))
    categories = convert_label_map_to_categories(label_map, max_num_classes)
    return create_category_index(categories)

def create_class_agnostic_category_index():
    if False:
        for i in range(10):
            print('nop')
    'Creates a category index with a single `object` class.'
    return {1: {'id': 1, 'name': 'object'}}