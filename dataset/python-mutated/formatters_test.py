from itertools import permutations
from typing import Union
import numpy as np
import torch
from hamcrest import assert_that, calling, close_to, equal_to, raises
from torch.utils.data import DataLoader, Dataset
from deepchecks.vision.datasets.detection import coco_torch
from deepchecks.vision.utils import image_properties
from deepchecks.vision.utils.detection_formatters import convert_batch_of_bboxes, convert_bbox, verify_bbox_format_notation
from deepchecks.vision.vision_data import VisionData

class SimpleImageData(VisionData):

    def batch_to_images(self, batch):
        if False:
            print('Hello World!')
        return batch

def numpy_shape_dataloader(shape: tuple=None, value: Union[float, np.ndarray]=255, collate_fn=None):
    if False:
        while True:
            i = 10
    if collate_fn is None:
        collate_fn = np.stack

    class TwoTupleDataset(Dataset):

        def __getitem__(self, index):
            if False:
                while True:
                    i = 10
            if isinstance(value, (float, int)):
                return np.ones(shape) * value
            else:
                return value

        def __len__(self) -> int:
            if False:
                return 10
            return 8
    return DataLoader(TwoTupleDataset(), batch_size=4, collate_fn=collate_fn)

def test_brightness_grayscale():
    if False:
        i = 10
        return i + 15
    value = np.concatenate([np.zeros((3, 10, 1)), np.ones((7, 10, 1))], axis=0)
    batch = next(iter(numpy_shape_dataloader(value=value)))
    res = image_properties.brightness(batch)
    assert_that(res, equal_to([0.7] * 4))

def test_brightness_rgb():
    if False:
        return 10
    value = np.concatenate([np.ones((10, 10, 1)) * 1, np.ones((10, 10, 1)) * 2, np.ones((10, 10, 1)) * 3], axis=2)
    batch = next(iter(numpy_shape_dataloader(value=value)))
    res = image_properties.brightness(batch)
    assert_that(res[0], close_to(1.86, 0.01))

def test_rms_contrast_grayscale():
    if False:
        return 10
    value = np.concatenate([np.zeros((3, 10, 1)), np.ones((7, 10, 1))], axis=0)
    expected_value = np.sqrt((70 * 0.3 ** 2 + 30 * 0.7 ** 2) / 100)
    batch = next(iter(numpy_shape_dataloader(value=value)))
    res = image_properties.rms_contrast(batch)
    assert_that(res[0], equal_to(expected_value))

def test_rms_contrast_rgb():
    if False:
        while True:
            i = 10
    value = np.concatenate([np.zeros((3, 10, 3)), np.concatenate([np.ones((7, 10, 1)) * 1 / 0.2125, np.ones((7, 10, 1)) * 1 / 0.7154, np.ones((7, 10, 1)) * 1 / 0.0721], axis=2)], axis=0)
    expected_value = np.sqrt((210 * 0.9 ** 2 + 90 * 2.1 ** 2) / 300)
    batch = next(iter(numpy_shape_dataloader(value=value)))
    res = image_properties.rms_contrast(batch)
    assert_that(res[0], close_to(expected_value, 1e-05))

def test_aspect_ratio():
    if False:
        for i in range(10):
            print('nop')
    batch = next(iter(numpy_shape_dataloader((10, 20, 3))))
    res = image_properties.aspect_ratio(batch)
    assert_that(res, equal_to([0.5] * 4))

def test_area():
    if False:
        return 10
    batch = next(iter(numpy_shape_dataloader((10, 20, 3))))
    res = image_properties.area(batch)
    assert_that(res, equal_to([200] * 4))

def test_normalized_mean_red():
    if False:
        for i in range(10):
            print('nop')
    value = np.concatenate([np.ones((10, 10, 1)) * 1, np.ones((10, 10, 1)) * 2, np.ones((10, 10, 1)) * 3], axis=2)
    expected_result = 1 / 6
    batch = next(iter(numpy_shape_dataloader(value=value)))
    res = image_properties.mean_red_relative_intensity(batch)
    assert_that(res[0], close_to(expected_result, 1e-07))

def test_normalized_mean_green():
    if False:
        print('Hello World!')
    value = np.concatenate([np.ones((10, 10, 1)) * 1, np.ones((10, 10, 1)) * 2, np.ones((10, 10, 1)) * 3], axis=2)
    expected_result = 2 / 6
    batch = next(iter(numpy_shape_dataloader(value=value)))
    res = image_properties.mean_green_relative_intensity(batch)
    assert_that(res[0], close_to(expected_result, 1e-07))

def test_normalized_mean_blue():
    if False:
        i = 10
        return i + 15
    value = np.concatenate([np.ones((10, 10, 1)) * 1, np.ones((10, 10, 1)) * 2, np.ones((10, 10, 1)) * 3], axis=2)
    expected_result = 3 / 6
    batch = next(iter(numpy_shape_dataloader(value=value)))
    res = image_properties.mean_blue_relative_intensity(batch)
    assert_that(res[0], close_to(expected_result, 1e-07))

def test_allowed_bbox_format_notations():
    if False:
        for i in range(10):
            print('nop')
    notations = ('  lxyxy       ', 'LxywH', *[''.join(it) for it in set(permutations(['l', 'xy', 'xy'], 3))], *['n' + ''.join(it) for it in set(permutations(['l', 'xy', 'xy'], 3))], *[''.join(it) for it in permutations(['l', 'xy', 'wh'], 3)], *[''.join(it) for it in permutations(['l', 'xy', 'wh'], 3)], *[''.join(it) for it in permutations(['l', 'cxcy', 'wh'], 3)], *[''.join(it) + 'n' for it in permutations(['l', 'cxcy', 'wh'], 3)])
    tokens = {'label', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'xcenter', 'ycenter'}
    for notation in notations:
        (are_coordinates_normalized, tokens) = verify_bbox_format_notation(notation)
        assert_that(len(set(tokens).difference(tokens)), equal_to(0))
        if 'n' in notation:
            assert_that(are_coordinates_normalized is True)

def test_bbox_format_notations_with_forbidden_combination_of_elements():
    if False:
        i = 10
        return i + 15
    notations = ['l', 'xy', 'wh', 'cxcy', *[''.join(it) for it in permutations(['l', 'xy', 'wh'], 2)], *[''.join(it) for it in permutations(['l', 'xy', 'cxcy'], 3)], *[''.join(it) for it in permutations(['l', 'xy', 'cxcy'], 2)], *[''.join(it) for it in permutations(['wh', 'cxcy', 'xy'], 2)], *[''.join(it) for it in permutations(['wh', 'cxcy', 'xy'], 3)]]
    for n in notations:
        assert_that(calling(verify_bbox_format_notation).with_args(n), raises(ValueError, f'Incorrect bbox format notation - {n}\\.\\nOnly next combinations of elements are allowed:.*\\n'))

def test_bbox_format_notations_with_unknown_elements():
    if False:
        while True:
            i = 10
    notations = ['lxyah', 'xyxyhw', 'cxywhl', 'l xy xy', 'lxxyy']
    for n in notations:
        assert_that(calling(verify_bbox_format_notation).with_args(n), raises(ValueError, f'Wrong bbox format notation - {n}\\. Incorrect or unknown sequence of charecters starting from position.*'))

def test_bbox_format_notation_with_coord_normalization_element_at_wrong_position():
    if False:
        i = 10
        return i + 15
    notations = ['lnxyxy', 'lxynxy', 'whnxyl', 'lcxcynwh']
    for notation in notations:
        assert_that(calling(verify_bbox_format_notation).with_args(notation), raises(ValueError, f'Wrong bbox format notation - {notation}\\. Incorrect or unknown sequence of charecters starting from position.*'))

def test_batch_of_bboxes_convertion():
    if False:
        print('Hello World!')
    loader = coco_torch.load_dataset()
    (_, input_bboxes) = batch = loader.dataset[9]
    output_bboxes = convert_batch_of_bboxes([batch], 'xywhl')[0]
    assert_that(len(output_bboxes) == len(input_bboxes))
    for (in_bbox, out_bbox) in zip(input_bboxes, output_bboxes):
        assert_that((out_bbox == np.asarray([in_bbox[-1], *in_bbox[:-1]])).all(), f'Input bbox: {in_bbox}; Output bbox: {out_bbox}')

def test_batch_of_bboxes_convertion_with_normalized_coordinates():
    if False:
        i = 10
        return i + 15
    loader = coco_torch.load_dataset()
    (image, input_bboxes) = loader.dataset[9]
    normilized_input_bboxes = torch.stack([torch.tensor([bbox[0] / image.width, bbox[1] / image.height, bbox[2], bbox[3], bbox[4]]) for bbox in input_bboxes], dim=0)
    output_bboxes = convert_batch_of_bboxes([(np.array(image), normilized_input_bboxes)], 'nxywhl')[0]
    assert_that(len(normilized_input_bboxes) == len(input_bboxes))
    for (index, output_bbox) in enumerate(output_bboxes):
        assert_that((output_bbox == np.asarray([input_bboxes[index][-1], *input_bboxes[index][:-1]])).all(), f'Original bbox: {input_bboxes[index]}; Normalized bbox: {normilized_input_bboxes[index]}; Output bbox: {output_bbox}')

def test_bbox_convertion_to_the_required_format():
    if False:
        i = 10
        return i + 15
    data = ((dict(notation='xylxy', bbox=np.asarray([20, 15, 2, 41, 23])), np.asarray([2, 20, 15, 21, 8])), (dict(notation='cxcywhl', bbox=np.asarray([50, 55, 100, 100, 0])), np.asarray([0, 50 - 100 / 2, 55 - 100 / 2, 100, 100])), (dict(notation='whxyl', bbox=np.asarray([35, 70, 10, 15, 1])), np.asarray([1, 10, 15, 35, 70])), (dict(notation='nxywhl', bbox=np.asarray([0.2, 0.2, 20, 40, 1]), image_width=100, image_height=100), np.asarray([1, 20, 20, 20, 40])), (dict(notation='cxcylwhn', bbox=np.asarray([0.12, 0.17, 0, 50, 100]), image_width=600, image_height=1200), np.asarray([0, 47, 154.00000000000003, 50, 100])))
    for (args, expected_result) in data:
        result = convert_bbox(**args)
        assert_that((result == expected_result).all(), f'Arguments: {args}, Result: {result}')

def test_convert_bbox_function_with_ambiguous_combination_of_parameters():
    if False:
        return 10
    (image_width, image_height) = (100, 100)
    bbox = torch.tensor([35, 70, 10, 15, 1])
    notation = 'whxyl'
    assert_that(calling(convert_bbox).with_args(bbox=bbox, notation=notation, image_width=image_width, image_height=image_height), raises(ValueError, "bbox format notation indicates that coordinates of the bbox are not normalized but \\'image_height\\' and \\'image_width\\' were provided. Those parameters are redundant in the case when bbox coordinates are not normalized\\. Please remove those parameters or add \\'n\\' element to the format notation to indicate that coordinates are indeed normalized\\."))
    normalized_bbox = torch.tensor([35, 70, 0.1, 0.15, 1])
    normalized_notation = 'whxyln'
    assert_that(calling(convert_bbox).with_args(bbox=normalized_bbox, notation=normalized_notation), raises(ValueError, "bbox format notation indicates that coordinates of the bbox are normalized but \\'image_height\\' and \\'image_width\\' parameters were not provided\\. Please pass image height and width parameters or remove \\'n\\' element from the format notation\\."))
    convert_bbox(bbox=bbox, notation=notation)
    convert_bbox(bbox=normalized_bbox, notation=normalized_notation, image_width=image_width, image_height=image_height)