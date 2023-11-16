from functools import lru_cache
import cv2
import torch
import torch.nn.functional as funct
from PIL import Image
from modelscope.models.cv.video_depth_estimation.utils.misc import same_shape

def load_image(path):
    if False:
        return 10
    '\n    Read an image using PIL\n\n    Parameters\n    ----------\n    path : str\n        Path to the image\n\n    Returns\n    -------\n    image : PIL.Image\n        Loaded image\n    '
    return Image.open(path)

def write_image(filename, image):
    if False:
        return 10
    '\n    Write an image to file.\n\n    Parameters\n    ----------\n    filename : str\n        File where image will be saved\n    image : np.array [H,W,3]\n        RGB image\n    '
    cv2.imwrite(filename, image[:, :, ::-1])

def flip_lr(image):
    if False:
        print('Hello World!')
    '\n    Flip image horizontally\n\n    Parameters\n    ----------\n    image : torch.Tensor [B,3,H,W]\n        Image to be flipped\n\n    Returns\n    -------\n    image_flipped : torch.Tensor [B,3,H,W]\n        Flipped image\n    '
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])

def flip_lr_intr(intr, width):
    if False:
        for i in range(10):
            print('nop')
    '\n    Flip image horizontally\n\n    Parameters\n    ----------\n    image : torch.Tensor [B,3,H,W]\n        Image to be flipped\n\n    Returns\n    -------\n    image_flipped : torch.Tensor [B,3,H,W]\n        Flipped image\n    '
    assert intr.shape[1:] == (3, 3)
    intr[:, 0, 0] = -1 * intr[:, 0, 0]
    intr[:, 0, 2] = width - intr[:, 0, 2]
    return intr

def flip_model(model, image, flip):
    if False:
        print('Hello World!')
    '\n    Flip input image and flip output inverse depth map\n\n    Parameters\n    ----------\n    model : nn.Module\n        Module to be used\n    image : torch.Tensor [B,3,H,W]\n        Input image\n    flip : bool\n        True if the flip is happening\n\n    Returns\n    -------\n    inv_depths : list of torch.Tensor [B,1,H,W]\n        List of predicted inverse depth maps\n    '
    if flip:
        return [flip_lr(inv_depth) for inv_depth in model(flip_lr(image))]
    else:
        return model(image)

def flip_mf_model(model, image, ref_imgs, intrinsics, flip, gt_depth=None, gt_poses=None):
    if False:
        i = 10
        return i + 15
    '\n    Flip input image and flip output inverse depth map\n\n    Parameters\n    ----------\n    model : nn.Module\n        Module to be used\n    image : torch.Tensor [B,3,H,W]\n        Input image\n    flip : bool\n        True if the flip is happening\n\n    Returns\n    -------\n    inv_depths : list of torch.Tensor [B,1,H,W]\n        List of predicted inverse depth maps\n    '
    if flip:
        if ref_imgs is not None:
            return model(flip_lr(image), [flip_lr(img) for img in ref_imgs], intrinsics, None, flip_lr(gt_depth), gt_poses)
        else:
            return model(flip_lr(image), None, intrinsics, None, flip_lr(gt_depth), gt_poses)
    else:
        return model(image, ref_imgs, intrinsics, None, gt_depth, gt_poses)

def gradient_x(image):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates the gradient of an image in the x dimension\n    Parameters\n    ----------\n    image : torch.Tensor [B,3,H,W]\n        Input image\n\n    Returns\n    -------\n    gradient_x : torch.Tensor [B,3,H,W-1]\n        Gradient of image with respect to x\n    '
    return image[:, :, :, :-1] - image[:, :, :, 1:]

def gradient_y(image):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates the gradient of an image in the y dimension\n    Parameters\n    ----------\n    image : torch.Tensor [B,3,H,W]\n        Input image\n\n    Returns\n    -------\n    gradient_y : torch.Tensor [B,3,H-1,W]\n        Gradient of image with respect to y\n    '
    return image[:, :, :-1, :] - image[:, :, 1:, :]

def interpolate_image(image, shape, mode='bilinear', align_corners=True):
    if False:
        print('Hello World!')
    '\n    Interpolate an image to a different resolution\n\n    Parameters\n    ----------\n    image : torch.Tensor [B,?,h,w]\n        Image to be interpolated\n    shape : tuple (H, W)\n        Output shape\n    mode : str\n        Interpolation mode\n    align_corners : bool\n        True if corners will be aligned after interpolation\n\n    Returns\n    -------\n    image : torch.Tensor [B,?,H,W]\n        Interpolated image\n    '
    if len(shape) > 2:
        shape = shape[-2:]
    if same_shape(image.shape[-2:], shape):
        return image
    else:
        return funct.interpolate(image, size=shape, mode=mode, align_corners=align_corners)

def interpolate_scales(images, shape=None, mode='bilinear', align_corners=False):
    if False:
        while True:
            i = 10
    '\n    Interpolate list of images to the same shape\n\n    Parameters\n    ----------\n    images : list of torch.Tensor [B,?,?,?]\n        Images to be interpolated, with different resolutions\n    shape : tuple (H, W)\n        Output shape\n    mode : str\n        Interpolation mode\n    align_corners : bool\n        True if corners will be aligned after interpolation\n\n    Returns\n    -------\n    images : list of torch.Tensor [B,?,H,W]\n        Interpolated images, with the same resolution\n    '
    if shape is None:
        shape = images[0].shape
    if len(shape) > 2:
        shape = shape[-2:]
    return [funct.interpolate(image, shape, mode=mode, align_corners=align_corners) for image in images]

def match_scales(image, targets, num_scales, mode='bilinear', align_corners=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Interpolate one image to produce a list of images with the same shape as targets\n\n    Parameters\n    ----------\n    image : torch.Tensor [B,?,h,w]\n        Input image\n    targets : list of torch.Tensor [B,?,?,?]\n        Tensors with the target resolutions\n    num_scales : int\n        Number of considered scales\n    mode : str\n        Interpolation mode\n    align_corners : bool\n        True if corners will be aligned after interpolation\n\n    Returns\n    -------\n    images : list of torch.Tensor [B,?,?,?]\n        List of images with the same resolutions as targets\n    '
    images = []
    image_shape = image.shape[-2:]
    for i in range(num_scales):
        target_shape = targets[i].shape
        if same_shape(image_shape, target_shape):
            images.append(image)
        else:
            images.append(interpolate_image(image, target_shape, mode=mode, align_corners=align_corners))
    return images

@lru_cache(maxsize=None)
def meshgrid(B, H, W, dtype, device, normalized=False):
    if False:
        return 10
    '\n    Create meshgrid with a specific resolution\n\n    Parameters\n    ----------\n    B : int\n        Batch size\n    H : int\n        Height size\n    W : int\n        Width size\n    dtype : torch.dtype\n        Meshgrid type\n    device : torch.device\n        Meshgrid device\n    normalized : bool\n        True if grid is normalized between -1 and 1\n\n    Returns\n    -------\n    xs : torch.Tensor [B,1,W]\n        Meshgrid in dimension x\n    ys : torch.Tensor [B,H,1]\n        Meshgrid in dimension y\n    '
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W - 1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H - 1, H, device=device, dtype=dtype)
    (ys, xs) = torch.meshgrid([ys, xs])
    return (xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1]))

@lru_cache(maxsize=None)
def image_grid(B, H, W, dtype, device, normalized=False):
    if False:
        return 10
    '\n    Create an image grid with a specific resolution\n\n    Parameters\n    ----------\n    B : int\n        Batch size\n    H : int\n        Height size\n    W : int\n        Width size\n    dtype : torch.dtype\n        Meshgrid type\n    device : torch.device\n        Meshgrid device\n    normalized : bool\n        True if grid is normalized between -1 and 1\n\n    Returns\n    -------\n    grid : torch.Tensor [B,3,H,W]\n        Image grid containing a meshgrid in x, y and 1\n    '
    (xs, ys) = meshgrid(B, H, W, dtype, device, normalized=normalized)
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=1)
    return grid