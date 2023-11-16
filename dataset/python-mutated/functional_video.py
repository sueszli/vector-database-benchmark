import torch

def _is_tensor_video_clip(clip):
    if False:
        i = 10
        return i + 15
    if not torch.is_tensor(clip):
        raise TypeError('clip should be Tesnor. Got %s' % type(clip))
    if not clip.ndimension() == 4:
        raise ValueError('clip should be 4D. Got %dD' % clip.dim())
    return True

def crop(clip, i, j, h, w):
    if False:
        return 10
    '\n    Args:\n        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)\n    '
    assert len(clip.size()) == 4, 'clip should be a 4D tensor'
    return clip[..., i:i + h, j:j + w]

def resize(clip, target_size, interpolation_mode):
    if False:
        return 10
    assert len(target_size) == 2, 'target size should be tuple (height, width)'
    return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode)

def resized_crop(clip, i, j, h, w, size, interpolation_mode='bilinear'):
    if False:
        print('Hello World!')
    '\n    Do spatial cropping and resizing to the video clip\n    Args:\n        clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)\n        i (int): i in (i,j) i.e coordinates of the upper left corner.\n        j (int): j in (i,j) i.e coordinates of the upper left corner.\n        h (int): Height of the cropped region.\n        w (int): Width of the cropped region.\n        size (tuple(int, int)): height and width of resized clip\n    Returns:\n        clip (torch.tensor): Resized and cropped clip. Size is (C, T, H, W)\n    '
    assert _is_tensor_video_clip(clip), 'clip should be a 4D torch.tensor'
    clip = crop(clip, i, j, h, w)
    clip = resize(clip, size, interpolation_mode)
    return clip

def center_crop(clip, crop_size):
    if False:
        i = 10
        return i + 15
    assert _is_tensor_video_clip(clip), 'clip should be a 4D torch.tensor'
    (h, w) = (clip.size(-2), clip.size(-1))
    (th, tw) = crop_size
    assert h >= th and w >= tw, 'height and width must be no smaller than crop_size'
    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)

def to_tensor(clip):
    if False:
        return 10
    '\n    Convert tensor data type from uint8 to float, divide value by 255.0 and\n    permute the dimenions of clip tensor\n    Args:\n        clip (torch.tensor, dtype=torch.uint8): Size is (T, H, W, C)\n    Return:\n        clip (torch.tensor, dtype=torch.float): Size is (C, T, H, W)\n    '
    assert _is_tensor_video_clip(clip), 'clip should be a 4D torch.tensor'
    if not clip.dtype == torch.uint8:
        raise TypeError('clip tensor should have data type uint8. Got %s' % str(clip.dtype))
    return clip.float().permute(3, 0, 1, 2) / 255.0

def normalize(clip, mean, std, inplace=False):
    if False:
        print('Hello World!')
    '\n    Args:\n        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)\n        mean (tuple): pixel RGB mean. Size is (3)\n        std (tuple): pixel standard deviation. Size is (3)\n    Returns:\n        normalized clip (torch.tensor): Size is (C, T, H, W)\n    '
    assert _is_tensor_video_clip(clip), 'clip should be a 4D torch.tensor'
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip

def hflip(clip):
    if False:
        i = 10
        return i + 15
    '\n    Args:\n        clip (torch.tensor): Video clip to be normalized. Size is (C, T, H, W)\n    Returns:\n        flipped clip (torch.tensor): Size is (C, T, H, W)\n    '
    assert _is_tensor_video_clip(clip), 'clip should be a 4D torch.tensor'
    return clip.flip(-1)

def denormalize(clip, mean, std):
    if False:
        while True:
            i = 10
    'Denormalize a sample who was normalized by (x - mean) / std\n    Args:\n        clip (torch.tensor): Video clip to be de-normalized\n        mean (tuple): pixel RGB mean. Size is (3)\n        std (tuple): pixel standard deviation. Size is (3)\n    Returns:\n    '
    result = clip.clone()
    for (t, m, s) in zip(result, mean, std):
        t.mul_(s).add_(m)
    return result