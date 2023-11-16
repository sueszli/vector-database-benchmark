from collections import OrderedDict
import random

def get_random_rgb():
    if False:
        print('Hello World!')
    'Generate a random RGB value\n\n    Returns\n    -------\n    list of float\n        Random RGB array\n    '
    return [round(random.random() * 255) for _ in range(0, 3)]

def assign_random_colors(data_vector):
    if False:
        for i in range(10):
            print('nop')
    'Produces lookup table keyed by each class of data, with value as an RGB array\n\n    Parameters\n    ---------\n    data_vector : list\n        Vector of data classes to be categorized, passed from the data itself\n\n    Returns\n    -------\n    collections.OrderedDict\n        Dictionary of random RGBA value per class, keyed on class\n    '
    deduped_classes = list(set(data_vector))
    classes = sorted([str(x) for x in deduped_classes])
    colors = []
    for _ in classes:
        colors.append(get_random_rgb())
    return OrderedDict([item for item in zip(classes, colors)])