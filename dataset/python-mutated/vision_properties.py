"""Module for calculating the properties used in Vision checks."""
from enum import Enum
__all__ = ['PropertiesInputType', 'validate_properties', 'calc_vision_properties']
from collections import defaultdict
from typing import Any, Dict, List
from deepchecks.core.errors import DeepchecksValueError

class PropertiesInputType(Enum):
    """Enum containing supported task types."""
    IMAGES = 'images'
    PARTIAL_IMAGES = 'partial_images'
    LABELS = 'labels'
    PREDICTIONS = 'predictions'

def calc_vision_properties(raw_data: List, properties_list: List) -> Dict[str, list]:
    if False:
        print('Hello World!')
    '\n    Calculate the image properties for a batch of images.\n\n    Parameters\n    ----------\n    raw_data : torch.Tensor\n        Batch of images to transform to image properties.\n\n    properties_list: List[Dict] , default: None\n        A list of properties to calculate.\n\n    Returns\n    ------\n    batch_properties: dict[str, List]\n        A dict of property name, property value per sample.\n    '
    batch_properties = defaultdict(list)
    for single_property in properties_list:
        property_list = single_property['method'](raw_data)
        batch_properties[single_property['name']] = property_list
    return batch_properties

def validate_properties(properties: List[Dict[str, Any]]):
    if False:
        print('Hello World!')
    'Validate structure of measurements.'
    if not isinstance(properties, list):
        raise DeepchecksValueError(f'Expected properties to be a list, instead got {type(properties).__name__}')
    if len(properties) == 0:
        raise DeepchecksValueError("Properties list can't be empty")
    expected_keys = ('name', 'method', 'output_type')
    output_types = ('categorical', 'numerical', 'class_id')
    errors = []
    for (index, image_property) in enumerate(properties):
        if not isinstance(image_property, dict) or not all((key in image_property for key in expected_keys)):
            errors.append(f'Item #{index}: property must be of type dict, and include keys {expected_keys}.')
            continue
        image_property['name'] = property_name = image_property.get('name') or f'#{index}'
        difference = sorted(set(expected_keys).difference(set(image_property.keys())))
        if len(difference) > 0:
            errors.append(f'Property {property_name}: dictionary must include keys {expected_keys}. Next keys are missed {difference}')
            continue
        property_output_type = image_property['output_type']
        if property_output_type not in output_types:
            errors.append(f'Property {property_name}: field "output_type" must be one of {output_types}, instead got {property_output_type}')
    if len(errors) > 0:
        errors = '\n+ '.join(errors)
        raise DeepchecksValueError(f'List of properties contains next problems:\n+ {errors}')
    return properties
STATIC_PROPERTIES_FORMAT = Dict[int, Dict[PropertiesInputType, Dict[str, Any]]]
PROPERTIES_CACHE_FORMAT = Dict[PropertiesInputType, Dict[str, List]]