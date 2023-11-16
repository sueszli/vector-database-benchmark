import os
import re
import sys
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def read_pfm(filename):
    if False:
        for i in range(10):
            print('nop')
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match('^(\\d+)\\s(\\d+)\\s$', file.readline().decode('utf-8'))
    if dim_match:
        (width, height) = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float(file.readline().rstrip())
    if scale < 0:
        endian = '<'
        scale = -scale
    else:
        endian = '>'
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return (data, scale)

def save_pfm(filename, image, scale=1):
    if False:
        for i in range(10):
            print('nop')
    file = open(filename, 'wb')
    color = None
    image = np.flipud(image)
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
    if len(image.shape) == 3 and image.shape[2] == 3:
        color = True
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))
    endian = image.dtype.byteorder
    if endian == '<' or (endian == '=' and sys.byteorder == 'little'):
        scale = -scale
    file.write(('%f\n' % scale).encode('utf-8'))
    image.tofile(file)
    file.close()
(S_H, S_W) = (0, 0)

class MVSDataset(Dataset):

    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        if False:
            return 10
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        (self.max_h, self.max_w) = (kwargs['max_h'], kwargs['max_w'])
        self.fix_res = kwargs.get('fix_res', False)
        self.fix_wh = False
        assert self.mode == 'test'
        self.metas = self.build_list()

    def build_list(self):
        if False:
            for i in range(10):
                print('nop')
        metas = []
        scans = self.listfile
        interval_scale_dict = {}
        for scan in scans:
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]
            pair_file = '{}/pair.txt'.format(scan)
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        metas.append((scan, ref_view, src_views, scan))
        self.interval_scale = interval_scale_dict
        return metas

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.metas)

    def read_cam_file(self, filename, interval_scale):
        if False:
            i = 10
            return i + 15
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])
        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths
        depth_interval *= interval_scale
        return (intrinsics, extrinsics, depth_min, depth_interval)

    def read_img(self, filename):
        if False:
            i = 10
            return i + 15
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32) / 255.0
        return np_img

    def read_depth(self, filename):
        if False:
            return 10
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=32):
        if False:
            while True:
                i = 10
        (h, w) = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            (new_w, new_h) = (scale * w // base * base, scale * h // base * base)
        else:
            (new_w, new_h) = (1.0 * w // base * base, 1.0 * h // base * base)
        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h
        img = cv2.resize(img, (int(new_w), int(new_h)))
        return (img, intrinsics)

    def __getitem__(self, idx):
        if False:
            return 10
        global S_H, S_W
        meta = self.metas[idx]
        (scan, ref_view, src_views, scene_name) = meta
        view_ids = [ref_view] + src_views[:self.nviews - 1]
        imgs = []
        depth_values = None
        proj_matrices = []
        for (i, vid) in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images_post/{:0>8}.jpg'.format(scan, vid))
            if not os.path.exists(img_filename):
                img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            img = self.read_img(img_filename)
            (intrinsics, extrinsics, depth_min, depth_interval) = self.read_cam_file(proj_mat_filename, interval_scale=self.interval_scale[scene_name])
            (img, intrinsics) = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h)
            if self.fix_res:
                (S_H, S_W) = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True
            if i == 0:
                if not self.fix_wh:
                    (S_H, S_W) = img.shape[:2]
            (c_h, c_w) = img.shape[:2]
            if c_h != S_H or c_w != S_W:
                scale_h = 1.0 * S_H / c_h
                scale_w = 1.0 * S_W / c_w
                img = cv2.resize(img, (S_W, S_H))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h
            imgs.append(img)
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)
            if i == 0:
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval, dtype=np.float32)
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
        proj_matrices_ms = {'stage1': proj_matrices, 'stage2': stage2_pjmats, 'stage3': stage3_pjmats}
        return {'imgs': imgs, 'proj_matrices': proj_matrices_ms, 'depth_values': depth_values, 'filename': scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + '{}'}