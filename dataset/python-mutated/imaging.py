from __future__ import annotations
from ..basics import *
from ..vision.all import *
from ..data.transforms import *
import pydicom, kornia, skimage
from pydicom.dataset import Dataset as DcmDataset
from pydicom.tag import BaseTag as DcmTag
from pydicom.multival import MultiValue as DcmMultiValue
from PIL import Image
try:
    import cv2
    cv2.setNumThreads(0)
except:
    pass
__all__ = ['dicom_windows', 'get_dicom_files', 'TensorDicom', 'PILDicom', 'array_freqhist_bins', 'TensorCTScan', 'PILCTScan', 'uniform_blur2d', 'gauss_blur2d', 'mask2bbox', 'crop_resize', 'DicomSegmentationDataLoaders', 'DcmDataset', 'DcmTag', 'DcmMultiValue', 'dcmread']
_all_ = ['DcmDataset', 'DcmTag', 'DcmMultiValue', 'dcmread', 'get_dicom_files', 'DicomSegmentationDataLoaders']

def get_dicom_files(path, recurse=True, folders=None):
    if False:
        print('Hello World!')
    'Get dicom files in `path` recursively, only in `folders`, if specified.'
    return get_files(path, extensions=['.dcm', '.dicom'], recurse=recurse, folders=folders)

@patch
def dcmread(fn: Path, force=False):
    if False:
        while True:
            i = 10
    'Open a `DICOM` file'
    return pydicom.dcmread(str(fn), force)

class TensorDicom(TensorImage):
    """Inherits from `TensorImage` and converts the `pixel_array` into a `TensorDicom`"""
    _show_args = {'cmap': 'gray'}

class PILDicom(PILBase):
    (_open_args, _tensor_cls, _show_args) = ({}, TensorDicom, TensorDicom._show_args)

    @classmethod
    def create(cls, fn: Path | str | bytes, mode=None) -> None:
        if False:
            i = 10
            return i + 15
        'Open a `DICOM file` from path `fn` or bytes `fn` and load it as a `PIL Image`'
        if isinstance(fn, bytes):
            im = Image.fromarray(pydicom.dcmread(pydicom.filebase.DicomBytesIO(fn)).pixel_array)
        if isinstance(fn, (Path, str)):
            im = Image.fromarray(pydicom.dcmread(fn).pixel_array)
        im.load()
        im = im._new(im.im)
        return cls(im.convert(mode) if mode else im)
PILDicom._tensor_cls = TensorDicom

@patch
def png16read(self: Path):
    if False:
        for i in range(10):
            print('nop')
    return array(Image.open(self), dtype=np.uint16)

@patch(as_prop=True)
def pixels(self: DcmDataset):
    if False:
        while True:
            i = 10
    '`pixel_array` as a tensor'
    return tensor(self.pixel_array.astype(np.float32))

@patch(as_prop=True)
def scaled_px(self: DcmDataset):
    if False:
        for i in range(10):
            print('nop')
    '`pixels` scaled by `RescaleSlope` and `RescaleIntercept`'
    img = self.pixels
    if hasattr(self, 'RescaleSlope') and hasattr(self, 'RescaleIntercept') is not None:
        return img * self.RescaleSlope + self.RescaleIntercept
    else:
        return img

def array_freqhist_bins(self, n_bins=100):
    if False:
        return 10
    'A numpy based function to split the range of pixel values into groups, such that each group has around the same number of pixels'
    imsd = np.sort(self.flatten())
    t = np.array([0.001])
    t = np.append(t, np.arange(n_bins) / n_bins + 1 / 2 / n_bins)
    t = np.append(t, 0.999)
    t = (len(imsd) * t + 0.5).astype(int)
    return np.unique(imsd[t])

@patch
def freqhist_bins(self: Tensor, n_bins=100):
    if False:
        i = 10
        return i + 15
    'A function to split the range of pixel values into groups, such that each group has around the same number of pixels'
    imsd = self.view(-1).sort()[0]
    t = torch.cat([tensor([0.001]), torch.arange(n_bins).float() / n_bins + 1 / 2 / n_bins, tensor([0.999])])
    t = (len(imsd) * t).long()
    return imsd[t].unique()

@patch
def hist_scaled_pt(self: Tensor, brks=None):
    if False:
        for i in range(10):
            print('nop')
    if brks is None:
        brks = self.freqhist_bins()
    brks = brks.to(self.device)
    ys = torch.linspace(0.0, 1.0, len(brks)).to(self.device)
    return self.flatten().interp_1d(brks, ys).reshape(self.shape).clamp(0.0, 1.0)

@patch
def hist_scaled(self: Tensor, brks=None):
    if False:
        for i in range(10):
            print('nop')
    'Scales a tensor using `freqhist_bins` to values between 0 and 1'
    if self.device.type == 'cuda':
        return self.hist_scaled_pt(brks)
    if brks is None:
        brks = self.freqhist_bins()
    ys = np.linspace(0.0, 1.0, len(brks))
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0.0, 1.0)

@patch
def hist_scaled(self: DcmDataset, brks=None, min_px=None, max_px=None):
    if False:
        i = 10
        return i + 15
    'Pixels scaled to a `min_px` and `max_px` value'
    px = self.scaled_px
    if min_px is not None:
        px[px < min_px] = min_px
    if max_px is not None:
        px[px > max_px] = max_px
    return px.hist_scaled(brks=brks)

@patch
def windowed(self: Tensor, w, l):
    if False:
        while True:
            i = 10
    'Scale pixel intensity by window width and window level'
    px = self.clone()
    px_min = l - w // 2
    px_max = l + w // 2
    px[px < px_min] = px_min
    px[px > px_max] = px_max
    return (px - px_min) / (px_max - px_min)

@patch
def windowed(self: DcmDataset, w, l):
    if False:
        while True:
            i = 10
    return self.scaled_px.windowed(w, l)
dicom_windows = types.SimpleNamespace(brain=(80, 40), subdural=(254, 100), stroke=(8, 32), brain_bone=(2800, 600), brain_soft=(375, 40), lungs=(1500, -600), mediastinum=(350, 50), abdomen_soft=(400, 50), liver=(150, 30), spine_soft=(250, 50), spine_bone=(1800, 400))

class TensorCTScan(TensorImageBW):
    """Inherits from `TensorImageBW` and converts the `pixel_array` into a `TensorCTScan`"""
    _show_args = {'cmap': 'bone'}

class PILCTScan(PILBase):
    (_open_args, _tensor_cls, _show_args) = ({}, TensorCTScan, TensorCTScan._show_args)

@patch
@delegates(show_image)
def show(self: DcmDataset, scale=True, cmap=plt.cm.bone, min_px=-1100, max_px=None, **kwargs):
    if False:
        print('Hello World!')
    'Display a normalized dicom image by default'
    px = self.windowed(*scale) if isinstance(scale, tuple) else self.hist_scaled(min_px=min_px, max_px=max_px, brks=scale) if isinstance(scale, (ndarray, Tensor)) else self.hist_scaled(min_px=min_px, max_px=max_px) if scale else self.scaled_px
    show_image(px, cmap=cmap, **kwargs)

@patch
def show(self: DcmDataset, frames=1, scale=True, cmap=plt.cm.bone, min_px=-1100, max_px=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'Adds functionality to view dicom images where each file may have more than 1 frame'
    px = self.windowed(*scale) if isinstance(scale, tuple) else self.hist_scaled(min_px=min_px, max_px=max_px, brks=scale) if isinstance(scale, (ndarray, Tensor)) else self.hist_scaled(min_px=min_px, max_px=max_px) if scale else self.scaled_px
    if px.ndim > 2:
        gh = []
        p = px.shape
        print(f'{p[0]} frames per file')
        for i in range(frames):
            u = px[i]
            gh.append(u)
        show_images(gh, **kwargs)
    else:
        show_image(px, cmap=cmap, **kwargs)

@patch
def pct_in_window(dcm: DcmDataset, w, l):
    if False:
        while True:
            i = 10
    '% of pixels in the window `(w,l)`'
    px = dcm.scaled_px
    return ((px > l - w // 2) & (px < l + w // 2)).float().mean().item()

def uniform_blur2d(x, s):
    if False:
        print('Hello World!')
    'Uniformly apply blurring'
    w = x.new_ones(1, 1, 1, s) / s
    x = unsqueeze(x, dim=0, n=4 - x.dim())
    r = F.conv2d(x, w, padding=s // 2)
    r = F.conv2d(r, w.transpose(-1, -2), padding=s // 2).cpu()[:, 0]
    return r.squeeze()

def gauss_blur2d(x, s):
    if False:
        return 10
    'Apply gaussian_blur2d kornia filter'
    s2 = int(s / 4) * 2 + 1
    x2 = unsqueeze(x, dim=0, n=4 - x.dim())
    res = kornia.filters.gaussian_blur2d(x2, (s2, s2), (s, s), 'replicate')
    return res.squeeze()

@patch
def mask_from_blur(x: Tensor, window, sigma=0.3, thresh=0.05, remove_max=True):
    if False:
        for i in range(10):
            print('nop')
    'Create a mask from the blurred image'
    p = x.windowed(*window)
    if remove_max:
        p[p == 1] = 0
    return gauss_blur2d(p, s=sigma * x.shape[-1]) > thresh

@patch
def mask_from_blur(x: DcmDataset, window, sigma=0.3, thresh=0.05, remove_max=True):
    if False:
        return 10
    'Create a mask from the blurred image'
    return to_device(x.scaled_px).mask_from_blur(window, sigma, thresh, remove_max=remove_max)

def _px_bounds(x, dim):
    if False:
        i = 10
        return i + 15
    c = x.sum(dim).nonzero().cpu()
    (idxs, vals) = torch.unique(c[:, 0], return_counts=True)
    vs = torch.split_with_sizes(c[:, 1], tuple(vals))
    d = {k.item(): v for (k, v) in zip(idxs, vs)}
    default_u = tensor([0, x.shape[-1] - 1])
    b = [d.get(o, default_u) for o in range(x.shape[0])]
    b = [tensor([o.min(), o.max()]) for o in b]
    return torch.stack(b)

def mask2bbox(mask):
    if False:
        print('Hello World!')
    no_batch = mask.dim() == 2
    if no_batch:
        mask = mask[None]
    bb1 = _px_bounds(mask, -1).t()
    bb2 = _px_bounds(mask, -2).t()
    res = torch.stack([bb1, bb2], dim=1).to(mask.device)
    return res[..., 0] if no_batch else res

def _bbs2sizes(crops, init_sz, use_square=True):
    if False:
        for i in range(10):
            print('nop')
    bb = crops.flip(1)
    szs = bb[1] - bb[0]
    if use_square:
        szs = szs.max(0)[0][None].repeat((2, 1))
    overs = szs + bb[0] > init_sz
    bb[0][overs] = init_sz - szs[overs]
    lows = bb[0] / float(init_sz)
    return (lows, szs / float(init_sz))

def crop_resize(x, crops, new_sz):
    if False:
        i = 10
        return i + 15
    bs = x.shape[0]
    (lows, szs) = _bbs2sizes(crops, x.shape[-1])
    if not isinstance(new_sz, (list, tuple)):
        new_sz = (new_sz, new_sz)
    id_mat = tensor([[1.0, 0, 0], [0, 1, 0]])[None].repeat((bs, 1, 1)).to(x.device)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        sp = F.affine_grid(id_mat, (bs, 1, *new_sz)) + 1.0
        grid = sp * unsqueeze(szs.t(), 1, n=2) + unsqueeze(lows.t() * 2.0, 1, n=2)
        return F.grid_sample(x.unsqueeze(1), grid - 1)

@patch
def to_nchan(x: Tensor, wins, bins=None):
    if False:
        return 10
    res = [x.windowed(*win) for win in wins]
    if not isinstance(bins, int) or bins != 0:
        res.append(x.hist_scaled(bins).clamp(0, 1))
    dim = [0, 1][x.dim() == 3]
    return TensorCTScan(torch.stack(res, dim=dim))

@patch
def to_nchan(x: DcmDataset, wins, bins=None):
    if False:
        while True:
            i = 10
    return x.scaled_px.to_nchan(wins, bins)

@patch
def to_3chan(x: Tensor, win1, win2, bins=None):
    if False:
        while True:
            i = 10
    return x.to_nchan([win1, win2], bins=bins)

@patch
def to_3chan(x: DcmDataset, win1, win2, bins=None):
    if False:
        return 10
    return x.scaled_px.to_3chan(win1, win2, bins)

@patch
def save_jpg(x: Tensor | DcmDataset, path, wins, bins=None, quality=90):
    if False:
        return 10
    'Save tensor or dicom image into `jpg` format'
    fn = Path(path).with_suffix('.jpg')
    x = (x.to_nchan(wins, bins) * 255).byte()
    im = Image.fromarray(x.permute(1, 2, 0).numpy(), mode=['RGB', 'CMYK'][x.shape[0] == 4])
    im.save(fn, quality=quality)

@patch
def to_uint16(x: Tensor | DcmDataset, bins=None):
    if False:
        for i in range(10):
            print('nop')
    'Convert into a unit16 array'
    d = x.hist_scaled(bins).clamp(0, 1) * 2 ** 16
    return d.numpy().astype(np.uint16)

@patch
def save_tif16(x: Tensor | DcmDataset, path, bins=None, compress=True):
    if False:
        for i in range(10):
            print('nop')
    'Save tensor or dicom image into `tiff` format'
    fn = Path(path).with_suffix('.tif')
    Image.fromarray(x.to_uint16(bins)).save(str(fn), compression='tiff_deflate' if compress else None)

@patch
def set_pixels(self: DcmDataset, px):
    if False:
        print('Hello World!')
    self.PixelData = px.tobytes()
    (self.Rows, self.Columns) = px.shape
DcmDataset.pixel_array = property(DcmDataset.pixel_array.fget, set_pixels)

@patch
def zoom(self: DcmDataset, ratio):
    if False:
        print('Hello World!')
    'Zoom image by specified ratio'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        self.set_pixels(ndimage.zoom(self.pixel_array, ratio))

@patch
def zoom_to(self: DcmDataset, sz):
    if False:
        for i in range(10):
            print('nop')
    'Change image size to specified pixel size'
    if not isinstance(sz, (list, tuple)):
        sz = (sz, sz)
    (rows, cols) = sz
    self.zoom((rows / self.Rows, cols / self.Columns))

@patch(as_prop=True)
def shape(self: DcmDataset):
    if False:
        i = 10
        return i + 15
    'Returns the shape of a dicom image as rows and columns'
    return (self.Rows, self.Columns)

def _cast_dicom_special(x):
    if False:
        return 10
    cls = type(x)
    if not cls.__module__.startswith('pydicom'):
        return x
    if cls.__base__ == object:
        return x
    return cls.__base__(x)

def _split_elem(vals):
    if False:
        return 10
    res = dict()
    for val in vals:
        (k, v) = (val.keyword, val.value)
        if not isinstance(v, DcmMultiValue):
            res[k] = v
            continue
        res[f'Multi{k}'] = 1
        for (i, o) in enumerate(v):
            res[f"{k}{('' if i == 0 else i)}"] = o
    return {k: _cast_dicom_special(v) for (k, v) in res.items()}

@patch
def as_dict(self: DcmDataset, px_summ=True, window=dicom_windows.brain):
    if False:
        return 10
    'Convert the header of a dicom into a dictionary'
    pxdata = (32736, 16)
    vals = [self[o] for o in self.keys() if o != pxdata]
    res = _split_elem(vals)
    res['fname'] = self.filename
    if not px_summ:
        return res
    stats = ('min', 'max', 'mean', 'std')
    try:
        pxs = self.pixel_array
        for f in stats:
            res['img_' + f] = getattr(pxs, f)()
        res['img_pct_window'] = self.pct_in_window(*window)
    except Exception as e:
        for f in stats:
            res['img_' + f] = 0
        print(res, e)
    return res

def _dcm2dict(fn, window=dicom_windows.brain, px_summ=True, **kwargs):
    if False:
        i = 10
        return i + 15
    return fn.dcmread().as_dict(window=window, px_summ=px_summ, **kwargs)

@delegates(parallel)
def _from_dicoms(cls, fns, n_workers=0, **kwargs):
    if False:
        return 10
    return pd.DataFrame(parallel(_dcm2dict, fns, n_workers=n_workers, **kwargs))
pd.DataFrame.from_dicoms = classmethod(_from_dicoms)

class DicomSegmentationDataLoaders(DataLoaders):
    """Basic wrapper around DICOM `DataLoaders` with factory methods for segmentation problems"""

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_label_func(cls, path, fnames, label_func, valid_pct=0.2, seed=None, codes=None, item_tfms=None, batch_tfms=None, **kwargs):
        if False:
            print('Hello World!')
        'Create from list of `fnames` in `path`s with `label_func`.'
        dblock = DataBlock(blocks=(ImageBlock(cls=PILDicom), MaskBlock(codes=codes)), splitter=RandomSplitter(valid_pct, seed=seed), get_y=label_func, item_tfms=item_tfms, batch_tfms=batch_tfms)
        res = cls.from_dblock(dblock, fnames, path=path, **kwargs)
        return res