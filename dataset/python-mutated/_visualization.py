from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
import numpy as _np
from turicreate.toolkits._internal_utils import _numeric_param_check_range
from turicreate.toolkits._main import ToolkitError as _ToolkitError

def _string_hash(s):
    if False:
        for i in range(10):
            print('nop')
    'String hash (djb2) with consistency between py2/py3 and persistency between runs (unlike `hash`).'
    h = 5381
    for c in s:
        h = h * 33 + ord(c)
    return h
COLOR_NAMES = ['AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange', 'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey', 'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen', 'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin', 'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed', 'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple', 'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown', 'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue', 'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White', 'WhiteSmoke', 'Yellow', 'YellowGreen']

def _annotate_image(pil_image, anns, confidence_threshold):
    if False:
        return 10
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    BUF = 2
    for ann in reversed(anns):
        if 'confidence' in ann and ann['confidence'] < confidence_threshold:
            continue
        if 'label' in ann:
            color = COLOR_NAMES[_string_hash(ann['label']) % len(COLOR_NAMES)]
        else:
            color = 'White'
        left = ann['coordinates']['x'] - ann['coordinates']['width'] / 2
        top = ann['coordinates']['y'] - ann['coordinates']['height'] / 2
        right = ann['coordinates']['x'] + ann['coordinates']['width'] / 2
        bottom = ann['coordinates']['y'] + ann['coordinates']['height'] / 2
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=4, fill=color)
        if 'confidence' in ann:
            text = '{} {:.0%}'.format(ann['label'], ann['confidence'])
        else:
            text = ann['label']
        (width, height) = font.getsize(text)
        if top < height + 2 * BUF:
            label_top = bottom + height + 2 * BUF
        else:
            label_top = top
        draw.rectangle([(left - 1, label_top - height - 2 * BUF), (left + width + 2 * BUF, label_top)], fill=color)
        draw.text((left + BUF, label_top - height - BUF), text, fill='black', font=font)

def draw_bounding_boxes(images, annotations, confidence_threshold=0):
    if False:
        return 10
    '\n    Visualizes bounding boxes (ground truth or predictions) by\n    returning annotated copies of the images.\n\n    Parameters\n    ----------\n    images: SArray or Image\n        An `SArray` of type `Image`. A single `Image` instance may also be\n        given.\n\n    annotations: SArray or list\n        An `SArray` of annotations (either output from the\n        `ObjectDetector.predict` function or ground truth). A single list of\n        annotations may also be given, provided that it is coupled with a\n        single image.\n\n    confidence_threshold: float\n        Confidence threshold can limit the number of boxes to draw. By\n        default, this is set to 0, since the prediction may have already pruned\n        with an appropriate confidence threshold.\n\n    Returns\n    -------\n    annotated_images: SArray or Image\n        Similar to the input `images`, except the images are decorated with\n        boxes to visualize the object instances.\n\n    See also\n    --------\n    unstack_annotations\n    '
    _numeric_param_check_range('confidence_threshold', confidence_threshold, 0.0, 1.0)
    from PIL import Image

    def draw_single_image(row):
        if False:
            while True:
                i = 10
        image = row['image']
        anns = row['annotations']
        row_number = row['id']
        if anns == None:
            anns = []
        elif type(anns) == dict:
            anns = [anns]
        try:
            pil_img = Image.fromarray(image.pixel_data)
            _annotate_image(pil_img, anns, confidence_threshold=confidence_threshold)
            image = _np.array(pil_img)
            if len(image.shape) == 2:
                image = image.reshape(image.shape[0], image.shape[1], 1)
            FORMAT_RAW = 2
            annotated_image = _tc.Image(_image_data=image.tobytes(), _width=image.shape[1], _height=image.shape[0], _channels=image.shape[2], _format_enum=FORMAT_RAW, _image_data_size=image.size)
        except Exception as e:
            if row_number == -1:
                raise _ToolkitError(e)
            raise _ToolkitError('Received exception at row ' + str(row_number) + ': ' + str(e))
        return annotated_image
    if isinstance(images, _tc.Image) and isinstance(annotations, list):
        return draw_single_image({'image': images, 'annotations': annotations, 'id': -1})
    else:
        sf = _tc.SFrame({'image': images, 'annotations': annotations})
        sf = sf.add_row_number()
        annotated_images = sf.apply(draw_single_image)
        return annotated_images