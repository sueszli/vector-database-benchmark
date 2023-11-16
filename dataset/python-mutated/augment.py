from __future__ import annotations
from ..data.all import *
from .core import *
from .data import *
__all__ = ['TensorTypes', 'RandTransform', 'FlipItem', 'DihedralItem', 'CropPad', 'RandomCrop', 'OldRandomCrop', 'Resize', 'RandomResizedCrop', 'RatioResize', 'affine_grid', 'AffineCoordTfm', 'RandomResizedCropGPU', 'mask_tensor', 'affine_mat', 'flip_mat', 'Flip', 'DeterministicDraw', 'DeterministicFlip', 'dihedral_mat', 'Dihedral', 'DeterministicDihedral', 'rotate_mat', 'Rotate', 'zoom_mat', 'Zoom', 'find_coeffs', 'apply_perspective', 'Warp', 'SpaceTfm', 'LightingTfm', 'Brightness', 'Contrast', 'grayscale', 'Saturation', 'rgb2hsv', 'hsv2rgb', 'HSVTfm', 'Hue', 'cutout_gaussian', 'norm_apply_denorm', 'RandomErasing', 'setup_aug_tfms', 'aug_transforms', 'PadMode', 'ResizeMethod']
from torch import stack, zeros_like as t0, ones_like as t1
from torch.distributions.bernoulli import Bernoulli

class RandTransform(DisplayedTransform):
    """A transform that before_call its state at each `__call__`"""
    (do, nm, supports, split_idx) = (True, None, [], 0)

    def __init__(self, p: float=1.0, nm: str=None, before_call: callable=None, **kwargs):
        if False:
            return 10
        store_attr('p')
        super().__init__(**kwargs)
        self.before_call = ifnone(before_call, self.before_call)

    def before_call(self, b, split_idx: int):
        if False:
            i = 10
            return i + 15
        'This function can be overridden. Set `self.do` based on `self.p`'
        self.do = self.p == 1.0 or random.random() < self.p

    def __call__(self, b, split_idx: int=None, **kwargs):
        if False:
            i = 10
            return i + 15
        self.before_call(b, split_idx=split_idx)
        return super().__call__(b, split_idx=split_idx, **kwargs) if self.do else b

def _neg_axis(x, axis):
    if False:
        print('Hello World!')
    x[..., axis] = -x[..., axis]
    return x
TensorTypes = (TensorImage, TensorMask, TensorPoint, TensorBBox)

@patch
def flip_lr(x: Image.Image):
    if False:
        for i in range(10):
            print('nop')
    return x.transpose(Image.FLIP_LEFT_RIGHT)

@patch
def flip_lr(x: TensorImageBase):
    if False:
        return 10
    return x.flip(-1)

@patch
def flip_lr(x: TensorPoint):
    if False:
        return 10
    return TensorPoint(_neg_axis(x.clone(), 0))

@patch
def flip_lr(x: TensorBBox):
    if False:
        while True:
            i = 10
    return TensorBBox(TensorPoint(x.view(-1, 2)).flip_lr().view(-1, 4))

class FlipItem(RandTransform):
    """Randomly flip with probability `p`"""

    def __init__(self, p: float=0.5):
        if False:
            i = 10
            return i + 15
        super().__init__(p=p)

    def encodes(self, x: (Image.Image, *TensorTypes)):
        if False:
            return 10
        return x.flip_lr()

@patch
def dihedral(x: PILImage, k: int):
    if False:
        while True:
            i = 10
    return x if k == 0 else x.transpose(k - 1)

@patch
def dihedral(x: TensorImage, k: int):
    if False:
        i = 10
        return i + 15
    if k in [1, 3, 4, 7]:
        x = x.flip(-1)
    if k in [2, 4, 5, 7]:
        x = x.flip(-2)
    if k in [3, 5, 6, 7]:
        x = x.transpose(-1, -2)
    return x

@patch
def dihedral(x: TensorPoint, k: int):
    if False:
        while True:
            i = 10
    if k in [1, 3, 4, 7]:
        x = _neg_axis(x, 0)
    if k in [2, 4, 5, 7]:
        x = _neg_axis(x, 1)
    if k in [3, 5, 6, 7]:
        x = x.flip(1)
    return x

@patch
def dihedral(x: TensorBBox, k: int):
    if False:
        print('Hello World!')
    pnts = TensorPoint(x.view(-1, 2)).dihedral(k).view(-1, 2, 2)
    (tl, br) = (pnts.min(dim=1)[0], pnts.max(dim=1)[0])
    return TensorBBox(torch.cat([tl, br], dim=1), img_size=x.img_size)

class DihedralItem(RandTransform):
    """Randomly flip with probability `p`"""

    def before_call(self, b, split_idx):
        if False:
            print('Hello World!')
        super().before_call(b, split_idx)
        self.k = random.randint(0, 7)

    def encodes(self, x: (Image.Image, *TensorTypes)):
        if False:
            print('Hello World!')
        return x.dihedral(self.k)
from torchvision.transforms.functional import pad as tvpad
mk_class('PadMode', **{o: o.lower() for o in ['Zeros', 'Border', 'Reflection']}, doc='All possible padding mode as attributes to get tab-completion and typo-proofing')
_all_ = ['PadMode']
_pad_modes = {'zeros': 'constant', 'border': 'edge', 'reflection': 'reflect'}

@patch
def _do_crop_pad(x: Image.Image, sz, tl, orig_sz, pad_mode=PadMode.Zeros, resize_mode=BILINEAR, resize_to=None):
    if False:
        print('Hello World!')
    if any(tl.ge(0)) or any(tl.add(sz).le(orig_sz)):
        c = tl.max(0)
        x = x.crop((*c, *tl.add(sz).min(orig_sz)))
    if any(tl.lt(0)) or any(tl.add(sz).ge(orig_sz)):
        p = (-tl).max(0)
        f = (sz - orig_sz).add(tl).max(0)
        x = tvpad(x, (*p, *f), padding_mode=_pad_modes[pad_mode])
    if resize_to is not None:
        x = x.resize(resize_to, resize_mode)
    return x

@patch
def _do_crop_pad(x: TensorPoint, sz, tl, orig_sz, pad_mode=PadMode.Zeros, resize_to=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    (orig_sz, sz, tl) = map(FloatTensor, (orig_sz, sz, tl))
    return TensorPoint((x + 1) * orig_sz / sz - tl * 2 / sz - 1, sz=sz if resize_to is None else resize_to)

@patch
def _do_crop_pad(x: TensorBBox, sz, tl, orig_sz, pad_mode=PadMode.Zeros, resize_to=None, **kwargs):
    if False:
        i = 10
        return i + 15
    bbox = TensorPoint._do_crop_pad(x.view(-1, 2), sz, tl, orig_sz, pad_mode, resize_to).view(-1, 4)
    return TensorBBox(bbox, img_size=x.img_size)

@patch
def crop_pad(x: TensorBBox | TensorPoint | Image.Image, sz: int | tuple, tl: tuple=None, orig_sz: tuple=None, pad_mode: PadMode=PadMode.Zeros, resize_mode=BILINEAR, resize_to: tuple=None):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(sz, int):
        sz = (sz, sz)
    orig_sz = fastuple(_get_sz(x) if orig_sz is None else orig_sz)
    (sz, tl) = (fastuple(sz), fastuple((_get_sz(x) - sz) // 2 if tl is None else tl))
    return x._do_crop_pad(sz, tl, orig_sz=orig_sz, pad_mode=pad_mode, resize_mode=resize_mode, resize_to=resize_to)

def _process_sz(size):
    if False:
        print('Hello World!')
    if isinstance(size, int):
        size = (size, size)
    return fastuple(size[1], size[0])

def _get_sz(x):
    if False:
        return 10
    if isinstance(x, tuple):
        x = x[0]
    if not isinstance(x, Tensor):
        return fastuple(x.size)
    return fastuple(getattr(x, 'img_size', getattr(x, 'sz', (x.shape[-1], x.shape[-2]))))

@delegates()
class CropPad(DisplayedTransform):
    """Center crop or pad an image to `size`"""
    order = 0

    def __init__(self, size: int | tuple, pad_mode: PadMode=PadMode.Zeros, **kwargs):
        if False:
            while True:
                i = 10
        size = _process_sz(size)
        store_attr()
        super().__init__(**kwargs)

    def encodes(self, x: Image.Image | TensorBBox | TensorPoint):
        if False:
            i = 10
            return i + 15
        orig_sz = _get_sz(x)
        tl = (orig_sz - self.size) // 2
        return x.crop_pad(self.size, tl, orig_sz=orig_sz, pad_mode=self.pad_mode)

@delegates()
class RandomCrop(RandTransform):
    """Randomly crop an image to `size`"""
    (split_idx, order) = (None, 1)

    def __init__(self, size: int | tuple, **kwargs):
        if False:
            i = 10
            return i + 15
        size = _process_sz(size)
        store_attr()
        super().__init__(**kwargs)

    def before_call(self, b, split_idx: int):
        if False:
            while True:
                i = 10
        'Randomly positioning crop if train dataset else center crop'
        self.orig_sz = _get_sz(b)
        if split_idx:
            self.tl = (self.orig_sz - self.size) // 2
        else:
            wd = self.orig_sz[0] - self.size[0]
            hd = self.orig_sz[1] - self.size[1]
            w_rand = (wd, -1) if wd < 0 else (0, wd)
            h_rand = (hd, -1) if hd < 0 else (0, hd)
            self.tl = fastuple(random.randint(*w_rand), random.randint(*h_rand))

    def encodes(self, x: Image.Image | TensorBBox | TensorPoint):
        if False:
            while True:
                i = 10
        return x.crop_pad(self.size, self.tl, orig_sz=self.orig_sz)

class OldRandomCrop(CropPad):
    """Randomly crop an image to `size`"""

    def before_call(self, b, split_idx):
        if False:
            i = 10
            return i + 15
        super().before_call(b, split_idx)
        (w, h) = self.orig_sz
        if not split_idx:
            self.tl = (random.randint(0, w - self.cp_size[0]), random.randint(0, h - self.cp_size[1]))
mk_class('ResizeMethod', **{o: o.lower() for o in ['Squish', 'Crop', 'Pad']}, doc='All possible resize method as attributes to get tab-completion and typo-proofing')
_all_ = ['ResizeMethod']

@delegates()
class Resize(RandTransform):
    (split_idx, mode, mode_mask, order) = (None, BILINEAR, NEAREST, 1)
    'Resize image to `size` using `method`'

    def __init__(self, size: int | tuple, method: ResizeMethod=ResizeMethod.Crop, pad_mode: PadMode=PadMode.Reflection, resamples=(BILINEAR, NEAREST), **kwargs):
        if False:
            while True:
                i = 10
        size = _process_sz(size)
        store_attr()
        super().__init__(**kwargs)
        (self.mode, self.mode_mask) = resamples

    def before_call(self, b, split_idx: int):
        if False:
            for i in range(10):
                print('nop')
        if self.method == ResizeMethod.Squish:
            return
        self.pcts = (0.5, 0.5) if split_idx else (random.random(), random.random())

    def encodes(self, x: Image.Image | TensorBBox | TensorPoint):
        if False:
            i = 10
            return i + 15
        orig_sz = _get_sz(x)
        if self.method == ResizeMethod.Squish:
            return x.crop_pad(orig_sz, fastuple(0, 0), orig_sz=orig_sz, pad_mode=self.pad_mode, resize_mode=self.mode_mask if isinstance(x, PILMask) else self.mode, resize_to=self.size)
        (w, h) = orig_sz
        op = (operator.lt, operator.gt)[self.method == ResizeMethod.Pad]
        m = w / self.size[0] if op(w / self.size[0], h / self.size[1]) else h / self.size[1]
        cp_sz = (int(m * self.size[0]), int(m * self.size[1]))
        tl = fastuple(int(self.pcts[0] * (w - cp_sz[0])), int(self.pcts[1] * (h - cp_sz[1])))
        return x.crop_pad(cp_sz, tl, orig_sz=orig_sz, pad_mode=self.pad_mode, resize_mode=self.mode_mask if isinstance(x, PILMask) else self.mode, resize_to=self.size)

@delegates()
class RandomResizedCrop(RandTransform):
    """Picks a random scaled crop of an image and resize it to `size`"""
    (split_idx, order) = (None, 1)

    def __init__(self, size: int | tuple, min_scale: float=0.08, ratio=(3 / 4, 4 / 3), resamples=(BILINEAR, NEAREST), val_xtra: float=0.14, max_scale: float=1.0, **kwargs):
        if False:
            i = 10
            return i + 15
        size = _process_sz(size)
        store_attr()
        super().__init__(**kwargs)
        (self.mode, self.mode_mask) = resamples

    def before_call(self, b, split_idx):
        if False:
            return 10
        (w, h) = self.orig_sz = _get_sz(b)
        if split_idx:
            xtra = math.ceil(max(*self.size[:2]) * self.val_xtra / 8) * 8
            self.final_size = (self.size[0] + xtra, self.size[1] + xtra)
            (self.tl, self.cp_size) = ((0, 0), self.orig_sz)
            return
        self.final_size = self.size
        for attempt in range(10):
            area = random.uniform(self.min_scale, self.max_scale) * w * h
            ratio = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            nw = int(round(math.sqrt(area * ratio)))
            nh = int(round(math.sqrt(area / ratio)))
            if nw <= w and nh <= h:
                self.cp_size = (nw, nh)
                self.tl = (random.randint(0, w - nw), random.randint(0, h - nh))
                return
        if w / h < self.ratio[0]:
            self.cp_size = (w, int(w / self.ratio[0]))
        elif w / h > self.ratio[1]:
            self.cp_size = (int(h * self.ratio[1]), h)
        else:
            self.cp_size = (w, h)
        self.tl = ((w - self.cp_size[0]) // 2, (h - self.cp_size[1]) // 2)

    def encodes(self, x: Image.Image | TensorBBox | TensorPoint):
        if False:
            i = 10
            return i + 15
        res = x.crop_pad(self.cp_size, self.tl, orig_sz=self.orig_sz, resize_mode=self.mode_mask if isinstance(x, PILMask) else self.mode, resize_to=self.final_size)
        if self.final_size != self.size:
            res = res.crop_pad(self.size)
        return res

class RatioResize(DisplayedTransform):
    """Resizes the biggest dimension of an image to `max_sz` maintaining the aspect ratio"""
    order = 1

    def __init__(self, max_sz: int, resamples=(BILINEAR, NEAREST), **kwargs):
        if False:
            return 10
        store_attr()
        super().__init__(**kwargs)

    def encodes(self, x: Image.Image | TensorBBox | TensorPoint):
        if False:
            i = 10
            return i + 15
        (w, h) = _get_sz(x)
        if w >= h:
            (nw, nh) = (self.max_sz, h * self.max_sz / w)
        else:
            (nw, nh) = (w * self.max_sz / h, self.max_sz)
        return Resize(size=(int(nh), int(nw)), resamples=self.resamples)(x)

def _init_mat(x):
    if False:
        i = 10
        return i + 15
    mat = torch.eye(3, device=x.device).float()
    return mat.unsqueeze(0).expand(x.size(0), 3, 3).contiguous()

def _grid_sample(x, coords, mode='bilinear', padding_mode='reflection', align_corners=None):
    if False:
        print('Hello World!')
    "Resample pixels in `coords` from `x` by `mode`, with `padding_mode` in ('reflection','border','zeros')."
    if mode == 'bilinear':
        (mn, mx) = (coords.min(), coords.max())
        z = 1 / (mx - mn).item() * 2
        d = min(x.shape[-2] / coords.shape[-2], x.shape[-1] / coords.shape[-1]) / 2
        if d > 1 and d > z:
            x = F.interpolate(x, scale_factor=1 / d, mode='area', recompute_scale_factor=True)
    return F.grid_sample(x, coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def affine_grid(theta: Tensor, size: tuple, align_corners: bool=None):
    if False:
        while True:
            i = 10
    ' Generates `TensorFlowField` from a transformation affine matrices `theta`'
    return TensorFlowField(F.affine_grid(theta, size, align_corners=align_corners))

@patch
def affine_coord(x: TensorImage, mat: Tensor=None, coord_tfm: callable=None, sz: int | tuple=None, mode: str='bilinear', pad_mode=PadMode.Reflection, align_corners=True):
    if False:
        while True:
            i = 10
    'Apply affine and coordinate transforms to `TensorImage`'
    if mat is None and coord_tfm is None and (sz is None):
        return x
    size = tuple(x.shape[-2:]) if sz is None else (sz, sz) if isinstance(sz, int) else tuple(sz)
    if mat is None:
        mat = _init_mat(x)[:, :2]
    coords = affine_grid(mat, x.shape[:2] + size, align_corners=align_corners)
    if coord_tfm is not None:
        coords = coord_tfm(coords)
    return TensorImage(_grid_sample(x, coords, mode=mode, padding_mode=pad_mode, align_corners=align_corners))

@patch
def affine_coord(x: TensorMask, mat: Tensor=None, coord_tfm: callable=None, sz: int | tuple=None, mode='nearest', pad_mode=PadMode.Reflection, align_corners=True):
    if False:
        print('Hello World!')
    'Apply affine and coordinate transforms to `TensorMask`'
    add_dim = x.ndim == 3
    if add_dim:
        x = x[:, None]
    res = TensorImage.affine_coord(x.float(), mat, coord_tfm, sz, mode, pad_mode, align_corners).long()
    if add_dim:
        res = res[:, 0]
    return TensorMask(res)

@patch
def affine_coord(x: TensorPoint, mat: Tensor=None, coord_tfm=None, sz=None, mode='nearest', pad_mode=PadMode.Zeros, align_corners=True):
    if False:
        return 10
    'Apply affine and coordinate transforms to `TensorPoint`'
    if sz is None:
        sz = getattr(x, 'img_size', None)
    if coord_tfm is not None:
        x = coord_tfm(x, invert=True)
    if mat is not None:
        mat = TensorPoint(mat)
        x = (x - mat[:, :, 2].unsqueeze(1)) @ torch.inverse(mat[:, :, :2].transpose(1, 2))
    return TensorPoint(x, sz=sz)

@patch
def affine_coord(x: TensorBBox, mat=None, coord_tfm=None, sz=None, mode='nearest', pad_mode=PadMode.Zeros, align_corners=True):
    if False:
        while True:
            i = 10
    'Apply affine and coordinate transforms to `TensorBBox`'
    if mat is None and coord_tfm is None:
        return x
    if sz is None:
        sz = getattr(x, 'img_size', None)
    (bs, n) = x.shape[:2]
    pnts = stack([x[..., :2], stack([x[..., 0], x[..., 3]], dim=2), stack([x[..., 2], x[..., 1]], dim=2), x[..., 2:]], dim=2)
    pnts = TensorPoint(pnts.view(bs, 4 * n, 2), img_size=sz).affine_coord(mat, coord_tfm, sz, mode, pad_mode)
    pnts = pnts.view(bs, n, 4, 2)
    (tl, dr) = (pnts.min(dim=2)[0], pnts.max(dim=2)[0])
    return TensorBBox(torch.cat([tl, dr], dim=2), img_size=sz)

def _prepare_mat(x, mat):
    if False:
        while True:
            i = 10
    (h, w) = getattr(x, 'img_size', x.shape[-2:])
    mat[:, 0, 1] *= h / w
    mat[:, 1, 0] *= w / h
    return mat[:, :2]

class AffineCoordTfm(RandTransform):
    """Combine and apply affine and coord transforms"""
    (order, split_idx) = (30, None)

    def __init__(self, aff_fs: callable | MutableSequence=None, coord_fs: callable | MutableSequence=None, size: int | tuple=None, mode='bilinear', pad_mode=PadMode.Reflection, mode_mask='nearest', align_corners=None, **kwargs):
        if False:
            return 10
        store_attr(but=['aff_fs', 'coord_fs'])
        super().__init__(**kwargs)
        (self.aff_fs, self.coord_fs) = (L(aff_fs), L(coord_fs))
        self.cp_size = None if size is None else (size, size) if isinstance(size, int) else tuple(size)

    def before_call(self, b, split_idx):
        if False:
            return 10
        while isinstance(b, tuple):
            b = b[0]
        self.split_idx = split_idx
        (self.do, self.mat) = (True, self._get_affine_mat(b))
        for t in self.coord_fs:
            t.before_call(b)

    def compose(self, tfm):
        if False:
            return 10
        'Compose `self` with another `AffineCoordTfm` to only do the interpolation step once'
        self.aff_fs += tfm.aff_fs
        self.coord_fs += tfm.coord_fs

    def _get_affine_mat(self, x):
        if False:
            while True:
                i = 10
        aff_m = _init_mat(x)
        if self.split_idx:
            return _prepare_mat(x, aff_m)
        ms = [f(x) for f in self.aff_fs]
        ms = [m for m in ms if m is not None]
        for m in ms:
            aff_m = aff_m @ m
        return _prepare_mat(x, aff_m)

    def _encode(self, x, mode, reverse=False):
        if False:
            return 10
        coord_func = None if len(self.coord_fs) == 0 or self.split_idx else partial(compose_tfms, tfms=self.coord_fs, reverse=reverse)
        return x.affine_coord(self.mat, coord_func, sz=self.size, mode=mode, pad_mode=self.pad_mode, align_corners=self.align_corners)

    def encodes(self, x: TensorImage):
        if False:
            while True:
                i = 10
        return self._encode(x, self.mode)

    def encodes(self, x: TensorMask):
        if False:
            for i in range(10):
                print('nop')
        return self._encode(x, self.mode_mask)

    def encodes(self, x: TensorPoint | TensorBBox):
        if False:
            while True:
                i = 10
        return self._encode(x, self.mode, reverse=True)

class RandomResizedCropGPU(RandTransform):
    """Picks a random scaled crop of an image and resize it to `size`"""
    (split_idx, order) = (None, 30)

    def __init__(self, size, min_scale=0.08, ratio=(3 / 4, 4 / 3), mode='bilinear', valid_scale=1.0, max_scale=1.0, mode_mask='nearest', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(size, int):
            size = (size, size)
        store_attr()
        super().__init__(**kwargs)

    def before_call(self, b, split_idx):
        if False:
            i = 10
            return i + 15
        self.do = True
        (h, w) = fastuple((b[0] if isinstance(b, tuple) else b).shape[-2:])
        for attempt in range(10):
            if split_idx:
                break
            area = random.uniform(self.min_scale, self.max_scale) * w * h
            ratio = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
            nw = int(round(math.sqrt(area * ratio)))
            nh = int(round(math.sqrt(area / ratio)))
            if nw <= w and nh <= h:
                self.cp_size = (nh, nw)
                self.tl = (random.randint(0, h - nh), random.randint(0, w - nw))
                return
        if w / h < self.ratio[0]:
            self.cp_size = (int(w / self.ratio[0]), w)
        elif w / h > self.ratio[1]:
            self.cp_size = (h, int(h * self.ratio[1]))
        else:
            self.cp_size = (h, w)
        if split_idx:
            self.cp_size = (int(self.cp_size[0] * self.valid_scale), int(self.cp_size[1] * self.valid_scale))
        self.tl = ((h - self.cp_size[0]) // 2, (w - self.cp_size[1]) // 2)

    def _encode(self, x, mode):
        if False:
            while True:
                i = 10
        x = x[..., self.tl[0]:self.tl[0] + self.cp_size[0], self.tl[1]:self.tl[1] + self.cp_size[1]]
        return x.affine_coord(sz=self.size, mode=mode)

    def encodes(self, x: TensorImage | TensorPoint | TensorBBox):
        if False:
            return 10
        return self._encode(x, self.mode)

    def encodes(self, x: TensorMask):
        if False:
            i = 10
            return i + 15
        return self._encode(x, self.mode_mask)

def mask_tensor(x: Tensor, p=0.5, neutral=0.0, batch=False):
    if False:
        while True:
            i = 10
    'Mask elements of `x` with `neutral` with probability `1-p`'
    if p == 1.0:
        return x
    if batch:
        return x if random.random() < p else x.new_zeros(*x.size()) + neutral
    if neutral != 0:
        x.add_(-neutral)
    mask = x.new_empty(*x.size()).float().bernoulli_(p).long()
    x.mul_(mask)
    return x.add_(neutral) if neutral != 0 else x

def _draw_mask(x, def_draw, draw=None, p=0.5, neutral=0.0, batch=False):
    if False:
        for i in range(10):
            print('nop')
    'Creates mask_tensor based on `x` with `neutral` with probability `1-p`. '
    if draw is None:
        draw = def_draw
    if callable(draw):
        res = draw(x)
    elif is_listy(draw):
        assert len(draw) >= x.size(0)
        res = tensor(draw[:x.size(0)], dtype=x.dtype, device=x.device)
    else:
        res = x.new_zeros(x.size(0)) + draw
    return TensorBase(mask_tensor(res, p=p, neutral=neutral, batch=batch))

def affine_mat(*ms):
    if False:
        for i in range(10):
            print('nop')
    'Restructure length-6 vector `ms` into an affine matrix with 0,0,1 in the last line'
    return stack([stack([ms[0], ms[1], ms[2]], dim=1), stack([ms[3], ms[4], ms[5]], dim=1), stack([t0(ms[0]), t0(ms[0]), t1(ms[0])], dim=1)], dim=1)

def flip_mat(x: Tensor, p=0.5, draw: int | MutableSequence | callable=None, batch: bool=False):
    if False:
        i = 10
        return i + 15
    'Return a random flip matrix'

    def _def_draw(x):
        if False:
            print('Hello World!')
        return x.new_ones(x.size(0))
    mask = x.new_ones(x.size(0)) - 2 * _draw_mask(x, _def_draw, draw=draw, p=p, batch=batch)
    return affine_mat(mask, t0(mask), t0(mask), t0(mask), t1(mask), t0(mask))

def _get_default(x, mode=None, pad_mode=None):
    if False:
        i = 10
        return i + 15
    if mode is None:
        mode = 'bilinear' if isinstance(x, TensorMask) else 'bilinear'
    if pad_mode is None:
        pad_mode = PadMode.Zeros if isinstance(x, (TensorPoint, TensorBBox)) else PadMode.Reflection
    x0 = x[0] if isinstance(x, tuple) else x
    return (x0, mode, pad_mode)

@patch
def flip_batch(x: TensorImage | TensorMask | TensorPoint | TensorBBox, p=0.5, draw: int | MutableSequence | callable=None, size: int | tuple=None, mode=None, pad_mode=None, align_corners=True, batch=False):
    if False:
        print('Hello World!')
    (x0, mode, pad_mode) = _get_default(x, mode, pad_mode)
    mat = flip_mat(x0, p=p, draw=draw, batch=batch)
    return x.affine_coord(mat=mat[:, :2], sz=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners)

class Flip(AffineCoordTfm):
    """Randomly flip a batch of images with a probability `p`"""

    def __init__(self, p=0.5, draw: int | MutableSequence | callable=None, size: int | tuple=None, mode: str='bilinear', pad_mode=PadMode.Reflection, align_corners=True, batch=False):
        if False:
            for i in range(10):
                print('nop')
        aff_fs = partial(flip_mat, p=p, draw=draw, batch=batch)
        super().__init__(aff_fs, size=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners, p=p)

class DeterministicDraw:

    def __init__(self, vals):
        if False:
            i = 10
            return i + 15
        (self.vals, self.count) = (vals, -1)

    def __call__(self, x):
        if False:
            return 10
        self.count += 1
        return x.new_zeros(x.size(0)) + self.vals[self.count % len(self.vals)]

class DeterministicFlip(Flip):
    """Flip the batch every other call"""

    def __init__(self, size: int | tuple=None, mode: str='bilinear', pad_mode=PadMode.Reflection, align_corners=True, **kwargs):
        if False:
            return 10
        super().__init__(p=1.0, draw=DeterministicDraw([0, 1]), mode=mode, pad_mode=pad_mode, align_corners=align_corners, **kwargs)

def dihedral_mat(x: Tensor, p: float=0.5, draw: int | MutableSequence | callable=None, batch: bool=False):
    if False:
        i = 10
        return i + 15
    'Return a random dihedral matrix'

    def _def_draw(x):
        if False:
            return 10
        return torch.randint(0, 8, (x.size(0),), device=x.device)

    def _def_draw_b(x):
        if False:
            i = 10
            return i + 15
        return random.randint(0, 7) + x.new_zeros((x.size(0),)).long()
    idx = _draw_mask(x, _def_draw_b if batch else _def_draw, draw=draw, p=p, batch=batch).long()
    xs = tensor([1, -1, 1, -1, -1, 1, 1, -1], device=x.device).gather(0, idx)
    ys = tensor([1, 1, -1, 1, -1, -1, 1, -1], device=x.device).gather(0, idx)
    m0 = tensor([1, 1, 1, 0, 1, 0, 0, 0], device=x.device).gather(0, idx)
    m1 = tensor([0, 0, 0, 1, 0, 1, 1, 1], device=x.device).gather(0, idx)
    return affine_mat(xs * m0, xs * m1, t0(xs), ys * m1, ys * m0, t0(xs)).float()

@patch
def dihedral_batch(x: TensorImage | TensorMask | TensorPoint | TensorBBox, p=0.5, draw: int | MutableSequence | callable=None, size: int | tuple=None, mode: str='bilinear', pad_mode=None, batch=False, align_corners=True):
    if False:
        return 10
    (x0, mode, pad_mode) = _get_default(x, mode, pad_mode)
    mat = _prepare_mat(x, dihedral_mat(x0, p=p, draw=draw, batch=batch))
    return x.affine_coord(mat=mat, sz=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners)

class Dihedral(AffineCoordTfm):
    """Apply a random dihedral transformation to a batch of images with a probability `p`"""

    def __init__(self, p=0.5, draw: int | MutableSequence | callable=None, size: int | tuple=None, mode: str='bilinear', pad_mode=PadMode.Reflection, batch=False, align_corners=True):
        if False:
            print('Hello World!')
        f = partial(dihedral_mat, p=p, draw=draw, batch=batch)
        super().__init__(aff_fs=f, size=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners)

class DeterministicDihedral(Dihedral):

    def __init__(self, size: int | tuple=None, mode: str='bilinear', pad_mode=PadMode.Reflection, align_corners=None):
        if False:
            print('Hello World!')
        'Flip the batch every other call'
        super().__init__(p=1.0, draw=DeterministicDraw(list(range(8))), pad_mode=pad_mode, align_corners=align_corners)

def rotate_mat(x: Tensor, max_deg: int=10, p: float=0.5, draw: int | MutableSequence | callable=None, batch: bool=False):
    if False:
        for i in range(10):
            print('nop')
    'Return a random rotation matrix with `max_deg` and `p`'

    def _def_draw(x):
        if False:
            print('Hello World!')
        return x.new_empty(x.size(0)).uniform_(-max_deg, max_deg)

    def _def_draw_b(x):
        if False:
            return 10
        return x.new_zeros(x.size(0)) + random.uniform(-max_deg, max_deg)
    thetas = _draw_mask(x, _def_draw_b if batch else _def_draw, draw=draw, p=p, batch=batch) * math.pi / 180
    return affine_mat(thetas.cos(), thetas.sin(), t0(thetas), -thetas.sin(), thetas.cos(), t0(thetas))

@patch
@delegates(rotate_mat)
def rotate(x: TensorImage | TensorMask | TensorPoint | TensorBBox, size: int | tuple=None, mode: str=None, pad_mode=None, align_corners: bool=True, **kwargs):
    if False:
        i = 10
        return i + 15
    (x0, mode, pad_mode) = _get_default(x, mode, pad_mode)
    mat = _prepare_mat(x, rotate_mat(x0, **kwargs))
    return x.affine_coord(mat=mat, sz=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners)

class Rotate(AffineCoordTfm):
    """Apply a random rotation of at most `max_deg` with probability `p` to a batch of images"""

    def __init__(self, max_deg: int=10, p: float=0.5, draw: int | MutableSequence | callable=None, size: int | tuple=None, mode: str='bilinear', pad_mode=PadMode.Reflection, align_corners: bool=True, batch: bool=False):
        if False:
            while True:
                i = 10
        aff_fs = partial(rotate_mat, max_deg=max_deg, p=p, draw=draw, batch=batch)
        super().__init__(aff_fs=aff_fs, size=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners)

def zoom_mat(x: Tensor, min_zoom: float=1.0, max_zoom: float=1.1, p: float=0.5, draw: float | MutableSequence | callable=None, draw_x: float | MutableSequence | callable=None, draw_y: float | MutableSequence | callable=None, batch: bool=False):
    if False:
        i = 10
        return i + 15
    'Return a random zoom matrix with `max_zoom` and `p`'

    def _def_draw(x):
        if False:
            print('Hello World!')
        return x.new_empty(x.size(0)).uniform_(min_zoom, max_zoom)

    def _def_draw_b(x):
        if False:
            for i in range(10):
                print('nop')
        return x.new_zeros(x.size(0)) + random.uniform(min_zoom, max_zoom)

    def _def_draw_ctr(x):
        if False:
            print('Hello World!')
        return x.new_empty(x.size(0)).uniform_(0, 1)

    def _def_draw_ctr_b(x):
        if False:
            i = 10
            return i + 15
        return x.new_zeros(x.size(0)) + random.uniform(0, 1)
    assert min_zoom <= max_zoom
    s = 1 / _draw_mask(x, _def_draw_b if batch else _def_draw, draw=draw, p=p, neutral=1.0, batch=batch)
    def_draw_c = _def_draw_ctr_b if batch else _def_draw_ctr
    col_pct = _draw_mask(x, def_draw_c, draw=draw_x, p=1.0, batch=batch)
    row_pct = _draw_mask(x, def_draw_c, draw=draw_y, p=1.0, batch=batch)
    col_c = (1 - s) * (2 * col_pct - 1)
    row_c = (1 - s) * (2 * row_pct - 1)
    return affine_mat(s, t0(s), col_c, t0(s), s, row_c)

@patch
@delegates(zoom_mat)
def zoom(x: TensorImage | TensorMask | TensorPoint | TensorBBox, size: int | tuple=None, mode: str='bilinear', pad_mode=PadMode.Reflection, align_corners: bool=True, **kwargs):
    if False:
        while True:
            i = 10
    (x0, mode, pad_mode) = _get_default(x, mode, pad_mode)
    return x.affine_coord(mat=zoom_mat(x0, **kwargs)[:, :2], sz=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners)

class Zoom(AffineCoordTfm):
    """Apply a random zoom of at most `max_zoom` with probability `p` to a batch of images"""

    def __init__(self, min_zoom: float=1.0, max_zoom: float=1.1, p: float=0.5, draw: float | MutableSequence | callable=None, draw_x: float | MutableSequence | callable=None, draw_y: float | MutableSequence | callable=None, size: int | tuple=None, mode='bilinear', pad_mode=PadMode.Reflection, batch=False, align_corners=True):
        if False:
            return 10
        aff_fs = partial(zoom_mat, min_zoom=min_zoom, max_zoom=max_zoom, p=p, draw=draw, draw_x=draw_x, draw_y=draw_y, batch=batch)
        super().__init__(aff_fs, size=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners)

def solve(A, B):
    if False:
        i = 10
        return i + 15
    return torch.linalg.solve(A, B)

def find_coeffs(p1: Tensor, p2: Tensor):
    if False:
        i = 10
        return i + 15
    'Find coefficients for warp tfm from `p1` to `p2`'
    m = []
    p = p1[:, 0, 0]
    for i in range(p1.shape[1]):
        m.append(stack([p2[:, i, 0], p2[:, i, 1], t1(p), t0(p), t0(p), t0(p), -p1[:, i, 0] * p2[:, i, 0], -p1[:, i, 0] * p2[:, i, 1]]))
        m.append(stack([t0(p), t0(p), t0(p), p2[:, i, 0], p2[:, i, 1], t1(p), -p1[:, i, 1] * p2[:, i, 0], -p1[:, i, 1] * p2[:, i, 1]]))
    A = stack(m).permute(2, 0, 1)
    B = p1.view(p1.shape[0], 8, 1)
    return solve(A, B)

def apply_perspective(coords: Tensor, coeffs: Tensor):
    if False:
        while True:
            i = 10
    'Apply perspective tranform on `coords` with `coeffs`'
    sz = coords.shape
    coords = coords.view(sz[0], -1, 2)
    coeffs = torch.cat([coeffs, t1(coeffs[:, :1])], dim=1).view(coeffs.shape[0], 3, 3)
    coords1 = coords @ coeffs[..., :2].transpose(1, 2) + coeffs[..., 2].unsqueeze(1)
    if (coords1[..., 2] == 0.0).any():
        return coords[..., :2].view(*sz)
    coords = coords1 / coords1[..., 2].unsqueeze(-1)
    return coords[..., :2].view(*sz)

class _WarpCoord:

    def __init__(self, magnitude=0.2, p=0.5, draw_x=None, draw_y=None, batch=False):
        if False:
            return 10
        store_attr()
        self.coeffs = None

    def _def_draw(self, x):
        if False:
            return 10
        if not self.batch:
            return x.new_empty(x.size(0)).uniform_(-self.magnitude, self.magnitude)
        return x.new_zeros(x.size(0)) + random.uniform(-self.magnitude, self.magnitude)

    def before_call(self, x):
        if False:
            print('Hello World!')
        x_t = _draw_mask(x, self._def_draw, self.draw_x, p=self.p, batch=self.batch)
        y_t = _draw_mask(x, self._def_draw, self.draw_y, p=self.p, batch=self.batch)
        orig_pts = torch.tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=x.dtype, device=x.device)
        self.orig_pts = orig_pts.unsqueeze(0).expand(x.size(0), 4, 2)
        targ_pts = stack([stack([-1 - y_t, -1 - x_t]), stack([-1 + y_t, 1 + x_t]), stack([1 + y_t, -1 + x_t]), stack([1 - y_t, 1 - x_t])])
        self.targ_pts = targ_pts.permute(2, 0, 1)

    def __call__(self, x, invert=False):
        if False:
            return 10
        coeffs = find_coeffs(self.targ_pts, self.orig_pts) if invert else find_coeffs(self.orig_pts, self.targ_pts)
        return apply_perspective(x, coeffs)

@patch
@delegates(_WarpCoord.__init__)
def warp(x: TensorImage | TensorMask | TensorPoint | TensorBBox, size: int | tuple=None, mode: str='bilinear', pad_mode=PadMode.Reflection, align_corners: bool=True, **kwargs):
    if False:
        while True:
            i = 10
    (x0, mode, pad_mode) = _get_default(x, mode, pad_mode)
    coord_tfm = _WarpCoord(**kwargs)
    coord_tfm.before_call(x0)
    return x.affine_coord(coord_tfm=coord_tfm, sz=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners)

class Warp(AffineCoordTfm):
    """Apply perspective warping with `magnitude` and `p` on a batch of matrices"""

    def __init__(self, magnitude: float=0.2, p: float=0.5, draw_x: float | MutableSequence | callable=None, draw_y: float | MutableSequence | callable=None, size: int | tuple=None, mode: str='bilinear', pad_mode=PadMode.Reflection, batch: bool=False, align_corners: bool=True):
        if False:
            i = 10
            return i + 15
        store_attr()
        coord_fs = _WarpCoord(magnitude=magnitude, p=p, draw_x=draw_x, draw_y=draw_y, batch=batch)
        super().__init__(coord_fs=coord_fs, size=size, mode=mode, pad_mode=pad_mode, align_corners=align_corners)

@patch
def lighting(x: TensorImage, func):
    if False:
        i = 10
        return i + 15
    return torch.sigmoid(func(logit(x)))

class SpaceTfm(RandTransform):
    """Apply `fs` to the logits"""
    order = 40

    def __init__(self, fs: callable | MutableSequence, space_fn: callable, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.space_fn = space_fn
        self.fs = L(fs)

    def before_call(self, b, split_idx: int):
        if False:
            for i in range(10):
                print('nop')
        self.do = True
        while isinstance(b, tuple):
            b = b[0]
        for t in self.fs:
            t.before_call(b)

    def compose(self, tfm: callable):
        if False:
            return 10
        'Compose `self` with another `LightingTransform`'
        self.fs += tfm.fs

    def encodes(self, x: TensorImage):
        if False:
            for i in range(10):
                print('nop')
        return self.space_fn(x, partial(compose_tfms, tfms=self.fs))

class LightingTfm(SpaceTfm):
    """Apply `fs` to the logits"""
    order = 40

    def __init__(self, fs: callable | MutableSequence, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(fs, TensorImage.lighting, **kwargs)

class _BrightnessLogit:

    def __init__(self, max_lighting=0.2, p=0.75, draw=None, batch=False):
        if False:
            print('Hello World!')
        store_attr()

    def _def_draw(self, x):
        if False:
            for i in range(10):
                print('nop')
        if not self.batch:
            return x.new_empty(x.size(0)).uniform_(0.5 * (1 - self.max_lighting), 0.5 * (1 + self.max_lighting))
        return x.new_zeros(x.size(0)) + random.uniform(0.5 * (1 - self.max_lighting), 0.5 * (1 + self.max_lighting))

    def before_call(self, x):
        if False:
            i = 10
            return i + 15
        self.change = _draw_mask(x, self._def_draw, draw=self.draw, p=self.p, neutral=0.5, batch=self.batch)

    def __call__(self, x):
        if False:
            print('Hello World!')
        return x.add_(logit(self.change[:, None, None, None]))

@patch
@delegates(_BrightnessLogit.__init__)
def brightness(x: TensorImage, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    func = _BrightnessLogit(**kwargs)
    func.before_call(x)
    return x.lighting(func)

class Brightness(LightingTfm):

    def __init__(self, max_lighting: float=0.2, p: float=0.75, draw: float | MutableSequence | callable=None, batch=False):
        if False:
            print('Hello World!')
        'Apply change in brightness of `max_lighting` to batch of images with probability `p`.'
        store_attr()
        super().__init__(_BrightnessLogit(max_lighting, p, draw, batch))

class _ContrastLogit:

    def __init__(self, max_lighting=0.2, p=0.75, draw=None, batch=False):
        if False:
            i = 10
            return i + 15
        store_attr()

    def _def_draw(self, x):
        if False:
            print('Hello World!')
        if not self.batch:
            res = x.new_empty(x.size(0)).uniform_(math.log(1 - self.max_lighting), -math.log(1 - self.max_lighting))
        else:
            res = x.new_zeros(x.size(0)) + random.uniform(math.log(1 - self.max_lighting), -math.log(1 - self.max_lighting))
        return torch.exp(res)

    def before_call(self, x):
        if False:
            i = 10
            return i + 15
        self.change = _draw_mask(x, self._def_draw, draw=self.draw, p=self.p, neutral=1.0, batch=self.batch)

    def __call__(self, x):
        if False:
            print('Hello World!')
        return x.mul_(self.change[:, None, None, None])

@patch
@delegates(_ContrastLogit.__init__)
def contrast(x: TensorImage, **kwargs):
    if False:
        i = 10
        return i + 15
    func = _ContrastLogit(**kwargs)
    func.before_call(x)
    return x.lighting(func)

class Contrast(LightingTfm):
    """Apply change in contrast of `max_lighting` to batch of images with probability `p`."""

    def __init__(self, max_lighting=0.2, p=0.75, draw: float | MutableSequence | callable=None, batch=False):
        if False:
            for i in range(10):
                print('nop')
        store_attr()
        super().__init__(_ContrastLogit(max_lighting, p, draw, batch))

def grayscale(x):
    if False:
        while True:
            i = 10
    'Tensor to grayscale tensor. Uses the ITU-R 601-2 luma transform. '
    return (x * torch.tensor([0.2989, 0.587, 0.114], device=x.device)[..., None, None]).sum(1)[:, None]

class _SaturationLogit:

    def __init__(self, max_lighting=0.2, p=0.75, draw=None, batch=False):
        if False:
            while True:
                i = 10
        store_attr()

    def _def_draw(self, x):
        if False:
            for i in range(10):
                print('nop')
        if not self.batch:
            res = x.new_empty(x.size(0)).uniform_(math.log(1 - self.max_lighting), -math.log(1 - self.max_lighting))
        else:
            res = x.new_zeros(x.size(0)) + random.uniform(math.log(1 - self.max_lighting), -math.log(1 - self.max_lighting))
        return torch.exp(res)

    def before_call(self, x):
        if False:
            i = 10
            return i + 15
        self.change = _draw_mask(x, self._def_draw, draw=self.draw, p=self.p, neutral=1.0, batch=self.batch)

    def __call__(self, x):
        if False:
            while True:
                i = 10
        gs = grayscale(x)
        gs.mul_(1 - self.change[:, None, None, None])
        x.mul_(self.change[:, None, None, None])
        return x.add_(gs)

@patch
@delegates(_SaturationLogit.__init__)
def saturation(x: TensorImage, **kwargs):
    if False:
        print('Hello World!')
    func = _SaturationLogit(**kwargs)
    func.before_call(x)
    return x.lighting(func)

class Saturation(LightingTfm):
    """Apply change in saturation of `max_lighting` to batch of images with probability `p`."""

    def __init__(self, max_lighting: float=0.2, p: float=0.75, draw: float | MutableSequence | callable=None, batch: bool=False):
        if False:
            return 10
        store_attr()
        super().__init__(_SaturationLogit(max_lighting, p, draw, batch))

def rgb2hsv(img: Tensor):
    if False:
        print('Hello World!')
    'Converts a RGB image to an HSV image. Note: Will not work on logit space images.'
    (r, g, b) = img.unbind(1)
    maxc = torch.max(img, dim=1)[0]
    minc = torch.min(img, dim=1)[0]
    eqc = maxc == minc
    cr = maxc - minc
    s = cr / torch.where(eqc, maxc.new_ones(()), maxc)
    cr_divisor = torch.where(eqc, maxc.new_ones(()), cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = torch.fmod(h / 6.0 + 1.0, 1.0)
    return torch.stack((h, s, maxc), dim=1)

def hsv2rgb(img: Tensor):
    if False:
        i = 10
        return i + 15
    'Converts a HSV image to an RGB image.'
    (h, s, v) = img.unbind(1)
    i = torch.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.to(dtype=torch.int32)
    p = torch.clamp(v * (1.0 - s), 0.0, 1.0)
    q = torch.clamp(v * (1.0 - s * f), 0.0, 1.0)
    t = torch.clamp(v * (1.0 - s * (1.0 - f)), 0.0, 1.0)
    i = i % 6
    mask = i[:, None] == torch.arange(6, device=i.device)[:, None, None][None]
    a1 = torch.stack((v, q, p, p, t, v), dim=1)
    a2 = torch.stack((t, v, v, q, p, p), dim=1)
    a3 = torch.stack((p, p, t, v, v, q), dim=1)
    a4 = torch.stack((a1, a2, a3), dim=1)
    return torch.einsum('nijk, nxijk -> nxjk', mask.to(dtype=img.dtype), a4)

@patch
def hsv(x: TensorImage, func):
    if False:
        print('Hello World!')
    return TensorImage(hsv2rgb(func(rgb2hsv(x))))

class HSVTfm(SpaceTfm):
    """Apply `fs` to the images in HSV space"""

    def __init__(self, fs, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(fs, TensorImage.hsv, **kwargs)

class _Hue:

    def __init__(self, max_hue=0.1, p=0.75, draw=None, batch=False):
        if False:
            while True:
                i = 10
        store_attr()

    def _def_draw(self, x):
        if False:
            for i in range(10):
                print('nop')
        if not self.batch:
            res = x.new_empty(x.size(0)).uniform_(math.log(1 - self.max_hue), -math.log(1 - self.max_hue))
        else:
            res = x.new_zeros(x.size(0)) + random.uniform(math.log(1 - self.max_hue), -math.log(1 - self.max_hue))
        return torch.exp(res)

    def before_call(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.change = _draw_mask(x, self._def_draw, draw=self.draw, p=self.p, neutral=0.0, batch=self.batch)

    def __call__(self, x):
        if False:
            while True:
                i = 10
        (h, s, v) = x.unbind(1)
        h += self.change[:, None, None]
        h = h % 1.0
        return x.set_(torch.stack((h, s, v), dim=1))

@patch
@delegates(_Hue.__init__)
def hue(x: TensorImage, **kwargs):
    if False:
        print('Hello World!')
    func = _Hue(**kwargs)
    func.before_call(x)
    return TensorImage(x.hsv(func))

class Hue(HSVTfm):
    """Apply change in hue of `max_hue` to batch of images with probability `p`."""

    def __init__(self, max_hue: float=0.1, p: float=0.75, draw: float | MutableSequence | callable=None, batch=False):
        if False:
            return 10
        super().__init__(_Hue(max_hue, p, draw, batch))

def cutout_gaussian(x: Tensor, areas: list):
    if False:
        i = 10
        return i + 15
    'Replace all `areas` in `x` with N(0,1) noise'
    (chan, img_h, img_w) = x.shape[-3:]
    for (rl, rh, cl, ch) in areas:
        x[..., rl:rh, cl:ch].normal_()
    return x

def norm_apply_denorm(x: Tensor, f: callable, nrm: callable):
    if False:
        while True:
            i = 10
    'Normalize `x` with `nrm`, then apply `f`, then denormalize'
    y = f(nrm(x.clone()))
    return nrm.decode(y).clamp(0, 1)

def _slice(area, sz):
    if False:
        i = 10
        return i + 15
    bound = int(round(math.sqrt(area)))
    loc = random.randint(0, max(sz - bound, 0))
    return (loc, loc + bound)

class RandomErasing(RandTransform):
    """Randomly selects a rectangle region in an image and randomizes its pixels."""
    order = 100

    def __init__(self, p: float=0.5, sl: float=0.0, sh: float=0.3, min_aspect: float=0.3, max_count: int=1):
        if False:
            print('Hello World!')
        store_attr()
        super().__init__(p=p)
        self.log_ratio = (math.log(min_aspect), math.log(1 / min_aspect))

    def _bounds(self, area, img_h, img_w):
        if False:
            print('Hello World!')
        r_area = random.uniform(self.sl, self.sh) * area
        aspect = math.exp(random.uniform(*self.log_ratio))
        return _slice(r_area * aspect, img_h) + _slice(r_area / aspect, img_w)

    def encodes(self, x: TensorImage):
        if False:
            print('Hello World!')
        count = random.randint(1, self.max_count)
        (_, img_h, img_w) = x.shape[-3:]
        area = img_h * img_w / count
        areas = [self._bounds(area, img_h, img_w) for _ in range(count)]
        return cutout_gaussian(x, areas)

def _compose_same_tfms(tfms):
    if False:
        while True:
            i = 10
    tfms = L(tfms)
    if len(tfms) == 0:
        return None
    res = tfms[0]
    for tfm in tfms[1:]:
        res.compose(tfm)
    return res

def setup_aug_tfms(tfms):
    if False:
        while True:
            i = 10
    'Go through `tfms` and combines together affine/coord or lighting transforms'
    aff_tfms = [tfm for tfm in tfms if isinstance(tfm, AffineCoordTfm)]
    lig_tfms = [tfm for tfm in tfms if isinstance(tfm, LightingTfm)]
    others = [tfm for tfm in tfms if tfm not in aff_tfms + lig_tfms]
    lig_tfm = _compose_same_tfms(lig_tfms)
    aff_tfm = _compose_same_tfms(aff_tfms)
    res = [aff_tfm] if aff_tfm is not None else []
    if lig_tfm is not None:
        res.append(lig_tfm)
    return res + others

def aug_transforms(mult: float=1.0, do_flip: bool=True, flip_vert: bool=False, max_rotate: float=10.0, min_zoom: float=1.0, max_zoom: float=1.1, max_lighting: float=0.2, max_warp: float=0.2, p_affine: float=0.75, p_lighting: float=0.75, xtra_tfms: list=None, size: int | tuple=None, mode: str='bilinear', pad_mode=PadMode.Reflection, align_corners=True, batch=False, min_scale=1.0):
    if False:
        i = 10
        return i + 15
    'Utility func to easily create a list of flip, rotate, zoom, warp, lighting transforms.'
    (res, tkw) = ([], dict(size=size if min_scale == 1.0 else None, mode=mode, pad_mode=pad_mode, batch=batch, align_corners=align_corners))
    (max_rotate, max_lighting, max_warp) = array([max_rotate, max_lighting, max_warp]) * mult
    if do_flip:
        res.append(Dihedral(p=0.5, **tkw) if flip_vert else Flip(p=0.5, **tkw))
    if max_warp:
        res.append(Warp(magnitude=max_warp, p=p_affine, **tkw))
    if max_rotate:
        res.append(Rotate(max_deg=max_rotate, p=p_affine, **tkw))
    if min_zoom < 1 or max_zoom > 1:
        res.append(Zoom(min_zoom=min_zoom, max_zoom=max_zoom, p=p_affine, **tkw))
    if max_lighting:
        res.append(Brightness(max_lighting=max_lighting, p=p_lighting, batch=batch))
        res.append(Contrast(max_lighting=max_lighting, p=p_lighting, batch=batch))
    if min_scale != 1.0:
        xtra_tfms = RandomResizedCropGPU(size, min_scale=min_scale, ratio=(1, 1)) + L(xtra_tfms)
    return setup_aug_tfms(res + L(xtra_tfms))