import os
import pkgutil
import warnings
from importlib import import_module
import healpy as hp
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torch.nn import functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils import model_zoo
try:
    from apex import amp
except ImportError:
    amp = None

@torch.no_grad()
def compute_hp_neighmap(nsides):
    if False:
        while True:
            i = 10
    " Precomputing the neighborhood indices table of each pixel on spherical surface.\n\n        Precomputing the neighborhood indices of each pixel on spherical surface.\n        The neighbor map provides the surrounding pixel ids for each pixel.\n        Hierarchical Equal Area isoLatitude Pixelization (HEALPix) is used.\n        The tool package 'healpy' can be found on https://github.com/healpy/healpy\n        It may take a little long time for high resolution（eg. nside = 128)\n\n        Args:\n            nsides: the pixelization resolution, could be 4, 8, 16, 32, 64, 128\n            which corresponds to pixle num 12 x nside^2\n\n        Returns:\n            neighbors: a dict-like neighborhood map corresponding to each resolution.\n    "
    neighbours = {}
    for i in range(len(nsides)):
        npix = hp.nside2npix(nsides[i])
        neighbour_map = torch.ones(9 * npix, dtype=torch.long, requires_grad=False)
        for p in range(npix):
            local_neighbor = hp.pixelfunc.get_all_neighbours(nsides[i], p, nest=True)
            local_neighbor = np.insert(local_neighbor, 4, p)
            ind = np.where(local_neighbor == -1)[0]
            local_neighbor[ind] = npix
            neighbour_map[9 * p:9 * p + 9] = torch.tensor(local_neighbor)
        neighbours[nsides[i]] = neighbour_map
    return neighbours

def precompute_pixelization_maps(nsides, initial_img_size=(128, 256)):
    if False:
        return 10
    ' Precomputing the mapping from multi-resolution ERP image to multi-resolution spherical pixels.\n\n        Precomputing the mapping from multi-resolution(sacle factor = 2) ERP image to\n        multi-resolution(sacle factor = 4)spherical surface. Each pixel on spherical surface\n        is back projected to ERP image and the indices are recorded and returned.\n        Bilinear interpolation is used.\n        Args:\n            nsides: the pixelization resolution, could be [32, 16, 8, 4]\n            initial_img_size: the initial image size, scale factor = 2\n            Returns:\n                index maps: a list-like indice maps corresponding to each nside.\n    '
    index_maps = []
    (ini_h, ini_w) = (initial_img_size[0], initial_img_size[1])
    for i in range(len(nsides)):
        (h, w) = (ini_h // 2 ** i, ini_w // 2 ** i)
        pixel_num_sp = hp.nside2npix(nsides[i])
        pixel_idx = np.arange(pixel_num_sp)
        sp_ll = hp.pix2ang(nsides[i], pixel_idx, nest=True, lonlat=True)
        (x, y) = (sp_ll[0] / 360.0 * w, (sp_ll[1] + 90.0) / 180.0 * h)
        (x, y) = (x.reshape(1, -1), y.reshape(1, -1))
        (x0, y0) = (np.floor(x).astype(np.int32), np.floor(y).astype(np.int32))
        (x1, y1) = (x0 + 1, y0 + 1)
        (x0, y0) = (np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1))
        (x1, y1) = (np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1))
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)
        index_maps.append(torch.from_numpy(np.concatenate([x0, y0, x1, y1, wa, wb, wc, wd], axis=0)).cuda())
    return index_maps

def precompute_position_encoding(nsides):
    if False:
        while True:
            i = 10
    'Precomputing spherical coordinates of spherical pixels for each nside'
    pos_encodings = []
    for i in range(len(nsides)):
        pixel_num_sp = hp.nside2npix(nsides[i])
        pixel_idx = np.arange(pixel_num_sp)
        (dir_x, dir_y, dir_z) = hp.pix2vec(nsides[i], pixel_idx, nest=True)
        (dir_x, dir_y, dir_z) = (dir_x.reshape(1, -1), dir_y.reshape(1, -1), dir_y.reshape(1, -1))
        pos_encodings.append(torch.from_numpy(np.concatenate([dir_x, dir_y, dir_z], axis=0)).unsqueeze(1).cuda())
    return pos_encodings

def auto_resume_helper(output_dir):
    if False:
        while True:
            i = 10
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def load_checkpoint_file(cfg, model, optimizer, lr_scheduler):
    if False:
        i = 10
        return i + 15
    if cfg.TRAIN.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(cfg.TRAIN.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(cfg.TRAIN.RESUME, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    rel_error = float('inf')
    if not cfg.EVAL_MODE and 'optimizer' in checkpoint and ('lr_scheduler' in checkpoint) and ('epoch' in checkpoint):
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.defrost()
        cfg.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        cfg.freeze()
        if 'amp' in checkpoint and cfg.AMP_OPT_LEVEL != 'O0' and (checkpoint['config'].AMP_OPT_LEVEL != 'O0'):
            amp.load_state_dict(checkpoint['amp'])
        if 'rel_error' in checkpoint:
            rel_error = checkpoint['rel_error']
    del checkpoint
    torch.cuda.empty_cache()
    return rel_error

def save_checkpoint(cfg, epoch, model, rel_error, optimizer, lr_scheduler, out_dir):
    if False:
        print('Hello World!')
    save_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'rel_error': rel_error, 'epoch': epoch, 'config': cfg}
    if cfg.AMP_OPT_LEVEL != 'O0':
        save_state['amp'] = amp.state_dict()
    save_path = os.path.join(out_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(save_state, save_path)

def reduce_tensor(tensor):
    if False:
        i = 10
        return i + 15
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def get_grad_norm(parameters, norm_type=2):
    if False:
        i = 10
        return i + 15
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm

def to_numpy(tensor):
    if False:
        for i in range(10):
            print('nop')
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
TORCH_VERSION = torch.__version__

def is_module_wrapper(module):
    if False:
        while True:
            i = 10
    module_wrappers = (DataParallel, DistributedDataParallel)
    return isinstance(module, module_wrappers)

def get_dist_info():
    if False:
        while True:
            i = 10
    if TORCH_VERSION < '1.0':
        initialized = dist._initialized
    elif dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return (rank, world_size)

def load_state_dict(module, state_dict, strict=False):
    if False:
        i = 10
        return i + 15
    "Load state_dict to a module.\n\n    This method is modified from :meth:`torch.nn.Module.load_state_dict`.\n    Default value for ``strict`` is set to ``False`` and the message for\n    param mismatch will be shown even if strict is False.\n\n    Args:\n        module (Module): Module that receives the state_dict.\n        state_dict (OrderedDict): Weights.\n        strict (bool): whether to strictly enforce that the keys\n            in :attr:`state_dict` match the keys returned by this module's\n            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.\n    "
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        if False:
            i = 10
            return i + 15
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, all_missing_keys, unexpected_keys, err_msg)
        for (name, child) in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    load(module)
    load = None
    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]
    if unexpected_keys:
        err_msg.append(f"unexpected key in source state_dict: {', '.join(unexpected_keys)}\n")
    if missing_keys:
        err_msg.append(f"missing keys in source state_dict: {', '.join(missing_keys)}\n")
    (rank, _) = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, '===== The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)

def load_url_dist(url, model_dir=None):
    if False:
        for i in range(10):
            print('nop')
    'In distributed setting, this function only download checkpoint at local\n    rank 0.'
    (rank, world_size) = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    return checkpoint

def get_torchvision_models():
    if False:
        while True:
            i = 10
    model_urls = dict()
    for (_, name, ispkg) in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls

def _load_checkpoint(filename, map_location=None):
    if False:
        print('Hello World!')
    'Load checkpoint from somewhere (modelzoo, file, url).\n\n    Args:\n        filename (str): Accept local filepath, URL, ``torchvision://xxx``,\n            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for\n            details.\n        map_location (str | None): Same as :func:`torch.load`. Default: None.\n\n    Returns:\n        dict | OrderedDict: The loaded checkpoint. It can be either an\n            OrderedDict storing model weights or a dict containing other\n            information, which depends on the checkpoint.\n    '
    if filename.startswith('modelzoo://'):
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_url_dist(model_urls[model_name])
    else:
        if not os.path.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint

def load_checkpoint(model, filename, map_location='cpu', strict=False):
    if False:
        i = 10
        return i + 15
    'Load checkpoint from a file or URI.\n\n    Args:\n        model (Module): Module to load checkpoint.\n        filename (str): Accept local filepath, URL, ``torchvision://xxx``,\n            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for\n            details.\n        map_location (str): Same as :func:`torch.load`.\n        strict (bool): Whether to allow different params for the model and\n            checkpoint.\n\n    Returns:\n        dict or OrderedDict: The loaded checkpoint.\n    '
    checkpoint = _load_checkpoint(filename, map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    tmp = {}
    for (k, v) in state_dict.items():
        if k.startswith('module.'):
            tmp[k[7:]] = v
        else:
            tmp[k] = v
    state_dict = tmp
    del tmp
    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for (k, v) in state_dict.items() if k.startswith('encoder.')}
    if state_dict.get('absolute_pos_embed') is not None:
        absolute_pos_embed = state_dict['absolute_pos_embed']
        (N1, L, C1) = absolute_pos_embed.size()
        (N2, C2, H, W) = model.absolute_pos_embed.size()
        if N1 != N2 or C1 != C2 or L != H * W:
            print('Error in loading absolute_pos_embed, pass')
        else:
            state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2)
    relative_position_bias_table_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k]
    for table_key in relative_position_bias_table_keys:
        table_pretrained = state_dict[table_key]
        table_current = model.state_dict()[table_key]
        (L1, nH1) = table_pretrained.size()
        (L2, nH2) = table_current.size()
        if nH1 != nH2:
            print(f'Error in loading {table_key}, pass')
        elif L1 != L2:
            S1 = int(L1 ** 0.5)
            S2 = int(L2 ** 0.5)
            table_pretrained_resized = F.interpolate(table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
            state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0)
    load_state_dict(model, state_dict, strict)
    return checkpoint

def render_depth_map(hp_data, image_to_sp):
    if False:
        i = 10
        return i + 15
    return hp_data[:, :, :, image_to_sp].squeeze(2)

def compute_hp_info(nside=128, img_size=(512, 1024)):
    if False:
        i = 10
        return i + 15
    hp_info = {}
    (h, w) = (img_size[0], img_size[1])
    pixel_num_sp = hp.nside2npix(nside)
    pixel_idx = np.arange(pixel_num_sp)
    sp_ll = hp.pix2ang(nside, pixel_idx, nest=True, lonlat=True)
    (x, y) = (sp_ll[0] / 360.0 * w, (sp_ll[1] + 90.0) / 180.0 * h)
    (x, y) = (x.reshape(1, -1), y.reshape(1, -1))
    (x0, y0) = (np.floor(x).astype(np.int32), np.floor(y).astype(np.int32))
    (x1, y1) = (x0 + 1, y0 + 1)
    (x0, y0) = (np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1))
    (x1, y1) = (np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1))
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    sp_to_image = np.concatenate([x0, y0, x1, y1, wa, wb, wc, wd], axis=0)
    sp_xyz = hp.pix2vec(nside, pixel_idx, nest=True)
    sp_xyz = np.stack([sp_xyz[0], sp_xyz[1], sp_xyz[2]], axis=1)
    theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    theta = np.pi - np.repeat(theta, w, axis=1)
    theta = theta.flatten()
    phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w
    phi = np.repeat(phi, h, axis=0)
    phi = phi.flatten()
    image_to_sp = hp.pixelfunc.ang2pix(nside, theta, phi, nest=True, lonlat=False)
    hp_info['hp_dir'] = sp_xyz
    hp_info['hp_pix_num'] = pixel_num_sp
    hp_info['hp_to_image_map'] = sp_to_image
    hp_info['image_to_sp_map'] = torch.from_numpy(image_to_sp.reshape(h, w))
    return hp_info