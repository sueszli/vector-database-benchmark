import numpy as np

def _convert_input_type_range(img):
    if False:
        print('Hello World!')
    'Convert the type and range of the input image.\n    It converts the input image to np.float32 type and range of [0, 1].\n    It is mainly used for pre-processing the input image in colorspace\n    conversion functions such as rgb2ycbcr and ycbcr2rgb.\n    Args:\n        img (ndarray): The input image. It accepts:\n            1. np.uint8 type with range [0, 255];\n            2. np.float32 type with range [0, 1].\n    Returns:\n        (ndarray): The converted image with type of np.float32 and range of\n            [0, 1].\n    '
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.0
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img

def _convert_output_type_range(img, dst_type):
    if False:
        i = 10
        return i + 15
    'Convert the type and range of the image according to dst_type.\n    It converts the image to desired type and range. If `dst_type` is np.uint8,\n    images will be converted to np.uint8 type with range [0, 255]. If\n    `dst_type` is np.float32, it converts the image to np.float32 type with\n    range [0, 1].\n    It is mainly used for post-processing images in colorspace conversion\n    functions such as rgb2ycbcr and ycbcr2rgb.\n    Args:\n        img (ndarray): The image to be converted with np.float32 type and\n            range [0, 255].\n        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it\n            converts the image to np.uint8 type with range [0, 255]. If\n            dst_type is np.float32, it converts the image to np.float32 type\n            with range [0, 1].\n    Returns:\n        (ndarray): The converted image with desired type and range.\n    '
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.0
    return img.astype(dst_type)

def bgr2ycbcr(img, y_only=False):
    if False:
        return 10
    'Convert a BGR image to YCbCr image.\n    The bgr version of rgb2ycbcr.\n    It implements the ITU-R BT.601 conversion for standard-definition\n    television. See more details in\n    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.\n    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.\n    In OpenCV, it implements a JPEG conversion. See more details in\n    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.\n    Args:\n        img (ndarray): The input image. It accepts:\n            1. np.uint8 type with range [0, 255];\n            2. np.float32 type with range [0, 1].\n        y_only (bool): Whether to only return Y channel. Default: False.\n    Returns:\n        ndarray: The converted YCbCr image. The output image has the same type\n            and range as input image.\n    '
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def reorder_image(img, input_order='HWC'):
    if False:
        while True:
            i = 10
    "Reorder images to 'HWC' order.\n    If the input_order is (h, w), return (h, w, 1);\n    If the input_order is (c, h, w), return (h, w, c);\n    If the input_order is (h, w, c), return as it is.\n    Args:\n        img (ndarray): Input image.\n        input_order (str): Whether the input order is 'HWC' or 'CHW'.\n            If the input image shape is (h, w), input_order will not have\n            effects. Default: 'HWC'.\n    Returns:\n        ndarray: reordered image.\n    "
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def to_y_channel(img):
    if False:
        print('Hello World!')
    'Change to Y channel of YCbCr.\n    Args:\n        img (ndarray): Images with range [0, 255].\n    Returns:\n        (ndarray): Images with range [0, 255] (float type) without round.\n    '
    img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.0