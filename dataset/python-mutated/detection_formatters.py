"""Module for defining detection encoders."""
from collections import Counter
from typing import Iterable, List, Sequence, Tuple, Union
import numpy as np
from PIL.Image import Image
from deepchecks.vision.vision_data.utils import is_torch_object
__all__ = ['verify_bbox_format_notation', 'convert_batch_of_bboxes', 'convert_bbox', 'DEFAULT_PREDICTION_FORMAT']
DEFAULT_PREDICTION_FORMAT = 'xywhsl'

def verify_bbox_format_notation(notation: str) -> Tuple[bool, List[str]]:
    if False:
        return 10
    "Verify and tokenize bbox format notation.\n\n    Parameters\n    ----------\n    notation : str\n        format notation to verify and to tokenize\n\n    Returns\n    -------\n    Tuple[\n        bool,\n        List[Literal['label', 'score', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'xcenter', 'ycenter']]\n    ]\n        first item indicates whether coordinates are normalized or not,\n        second represents format of the bbox\n    "
    tokens = []
    are_coordinates_normalized = False
    current = notation = notation.strip().lower()
    current_pos = 0
    while current:
        if current.startswith('l'):
            tokens.append('l')
            current = current[1:]
            current_pos = current_pos + 1
        elif current.startswith('s'):
            tokens.append('s')
            current = current[1:]
            current_pos = current_pos + 1
        elif current.startswith('wh'):
            tokens.append('wh')
            current = current[2:]
            current_pos = current_pos + 2
        elif current.startswith('xy'):
            tokens.append('xy')
            current = current[2:]
            current_pos = current_pos + 2
        elif current.startswith('cxcy'):
            tokens.append('cxcy')
            current = current[4:]
            current_pos = current_pos + 4
        elif current.startswith('n') and current_pos == 0:
            are_coordinates_normalized = True
            current = current[1:]
            current_pos = current_pos + 1
        elif current.startswith('n') and current_pos + 1 == len(notation):
            are_coordinates_normalized = True
            current_pos = current_pos + 1
            break
        else:
            raise ValueError(f'Wrong bbox format notation - {notation}. Incorrect or unknown sequence of charecters starting from position {current_pos} (sequence: ...{notation[current_pos:]}')
    received_combination = Counter(tokens)
    allowed_combinations = [{'l': 1, 'xy': 2}, {'l': 1, 'xy': 1, 'wh': 1}, {'l': 1, 'cxcy': 1, 'wh': 1}]
    allowed_combinations += [{**c, 's': 1} for c in allowed_combinations]
    if sum((c == received_combination for c in allowed_combinations)) != 1:
        raise ValueError(f'Incorrect bbox format notation - {notation}.\nOnly next combinations of elements are allowed:\n+ lxyxy (label, upper-left corner, bottom-right corner)\n+ lxywh (label, upper-left corner, bbox width and height)\n+ lcxcywh (label, bbox center, bbox width and height)\n+ lcxcywhn (label, normalized bbox center, bbox width and height)\n\nNote:\n- notation elements (l, xy, cxcy, wh) can be placed in any order but only above combinations of elements are allowed\n- "n" at the begining or at the ned of the notation indicates normalized coordinates\n')
    normalized_tokens = []
    for t in tokens:
        if t == 'l':
            normalized_tokens.append('label')
        elif t == 's':
            normalized_tokens.append('score')
        elif t == 'wh':
            normalized_tokens.extend(('width', 'height'))
        elif t == 'cxcy':
            normalized_tokens.extend(('xcenter', 'ycenter'))
        elif t == 'xy':
            if 'xmin' not in normalized_tokens and 'ymin' not in normalized_tokens:
                normalized_tokens.extend(('xmin', 'ymin'))
            else:
                normalized_tokens.extend(('xmax', 'ymax'))
        else:
            raise RuntimeError('Internal Error! Unreachable part of code reached')
    return (are_coordinates_normalized, normalized_tokens)
_BatchOfSamples = Iterable[Tuple[Union[Image, np.ndarray], Sequence[Sequence[Union[int, float]]]]]

def convert_batch_of_bboxes(batch: _BatchOfSamples, notation: str) -> List[np.ndarray]:
    if False:
        while True:
            i = 10
    'Convert batch of bboxes to the required format.\n\n    Parameters\n    ----------\n    batch : iterable of tuple like object with two items - image, list of bboxes\n        batch of images and bboxes corresponding to them\n    notation : str\n        bboxes format notation\n\n    Returns\n    -------\n    List[np.ndarray]\n        list of transformed bboxes\n    '
    (are_coordinates_normalized, notation_tokens) = verify_bbox_format_notation(notation)
    output = []
    for (image, bboxes) in batch:
        if len(bboxes) == 0:
            output.append(np.asarray([]))
            continue
        if are_coordinates_normalized is False:
            image_height = None
            image_width = None
        elif isinstance(image, Image):
            (image_height, image_width) = (image.height, image.width)
        elif is_torch_object(image) or isinstance(image, np.ndarray):
            (image_height, image_width, *_) = image.shape
        else:
            raise TypeError(f'Do not know how to take dimension sizes of object of type - {type(image)}')
        r = []
        for bbox in bboxes:
            if len(bbox) < 5:
                raise ValueError('incorrect bbox')
            else:
                r.append(_convert_bbox(bbox, notation_tokens, image_width=image_width, image_height=image_height))
        output.append(np.stack(r, axis=0))
    return output

def convert_bbox(bbox: Sequence[Union[int, float]], notation: str, image_width: Union[int, float, None]=None, image_height: Union[int, float, None]=None, _strict: bool=True) -> np.ndarray:
    if False:
        return 10
    'Convert bbox to the required format.\n\n    Parameters\n    ----------\n    bbox : Sequence[Sequence[Union[int, float]]]\n        bbox to transform\n    notation : str\n        bboxes format notation\n    image_width : Union[int, float, None], default: None\n        width of the image to denormalize bbox coordinates\n    image_height : Union[int, float, None], default: None\n        height of the image to denormalize bbox coordinates\n\n    Returns\n    -------\n    np.ndarray\n        bbox transformed to the required by deepchecks format\n    '
    if len(bbox) < 5:
        raise ValueError('incorrect bbox')
    (are_coordinates_normalized, notation_tokens) = verify_bbox_format_notation(notation)
    if are_coordinates_normalized is True and (image_height is None or image_width is None):
        raise ValueError("bbox format notation indicates that coordinates of the bbox are normalized but 'image_height' and 'image_width' parameters were not provided. Please pass image height and width parameters or remove 'n' element from the format notation.")
    if are_coordinates_normalized is False and (image_height is not None or image_width is not None):
        if _strict is True:
            raise ValueError("bbox format notation indicates that coordinates of the bbox are not normalized but 'image_height' and 'image_width' were provided. Those parameters are redundant in the case when bbox coordinates are not normalized. Please remove those parameters or add 'n' element to the format notation to indicate that coordinates are indeed normalized.")
        else:
            image_height = None
            image_width = None
    return _convert_bbox(bbox, notation_tokens, image_width, image_height)

def _convert_bbox(bbox: Sequence[Union[int, float]], notation_tokens: List[str], image_width: Union[int, float, None]=None, image_height: Union[int, float, None]=None) -> np.ndarray:
    if False:
        print('Hello World!')
    assert image_width is not None and image_height is not None or (image_width is None and image_height is None)
    data = dict(zip(notation_tokens, bbox))
    if 'xcenter' in data and 'ycenter' in data:
        if image_width is not None and image_height is not None:
            (xcenter, ycenter) = (data['xcenter'] * image_width, data['ycenter'] * image_height)
        else:
            (xcenter, ycenter) = (data['xcenter'], data['ycenter'])
        return np.asarray([data['label'], xcenter - data['width'] / 2, ycenter - data['height'] / 2, data['width'], data['height']])
    elif 'height' in data and 'width' in data:
        if image_width is not None and image_height is not None:
            (xmin, ymin) = (data['xmin'] * image_width, data['ymin'] * image_height)
        else:
            (xmin, ymin) = (data['xmin'], data['ymin'])
        return np.asarray([data['label'], xmin, ymin, data['width'], data['height']])
    else:
        if image_width is not None and image_height is not None:
            (xmin, ymin) = (data['xmin'] * image_width, data['ymin'] * image_height)
            (xmax, ymax) = (data['xmax'] * image_width, data['ymax'] * image_height)
        else:
            (xmin, ymin) = (data['xmin'], data['ymin'])
            (xmax, ymax) = (data['xmax'], data['ymax'])
        return np.asarray([data['label'], xmin, ymin, xmax - xmin, ymax - ymin])