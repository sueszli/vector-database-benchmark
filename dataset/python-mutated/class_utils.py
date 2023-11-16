"""Utility functions for handling dataset object categories."""

def coco_split_class_ids(split_name):
    if False:
        i = 10
        return i + 15
    'Return the COCO class split ids based on split name and training mode.\n\n  Args:\n    split_name: The name of dataset split.\n\n  Returns:\n    class_ids: a python list of integer.\n  '
    if split_name == 'all':
        return []
    elif split_name == 'voc':
        return [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
    elif split_name == 'nonvoc':
        return [8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    else:
        raise ValueError('Invalid split name {}!!!'.format(split_name))