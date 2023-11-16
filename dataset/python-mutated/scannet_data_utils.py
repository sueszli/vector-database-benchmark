import os
from concurrent import futures as futures
from os import path as osp
import mmcv
import numpy as np

class ScanNetData(object):
    """ScanNet data.

    Generate scannet infos for scannet_converter.

    Args:
        root_path (str): Root path of the raw data.
        split (str, optional): Set split type of the data. Default: 'train'.
    """

    def __init__(self, root_path, split='train'):
        if False:
            return 10
        self.root_dir = root_path
        self.split = split
        self.split_dir = osp.join(root_path)
        self.classes = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'garbagebin']
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.cat_ids2class = {nyu40id: i for (i, nyu40id) in enumerate(list(self.cat_ids))}
        assert split in ['train', 'val', 'test']
        split_file = osp.join(self.root_dir, 'meta_data', f'scannetv2_{split}.txt')
        mmcv.check_file_exist(split_file)
        self.sample_id_list = mmcv.list_from_file(split_file)
        self.test_mode = split == 'test'

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.sample_id_list)

    def get_aligned_box_label(self, idx):
        if False:
            i = 10
            return i + 15
        box_file = osp.join(self.root_dir, 'scannet_instance_data', f'{idx}_aligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_unaligned_box_label(self, idx):
        if False:
            print('Hello World!')
        box_file = osp.join(self.root_dir, 'scannet_instance_data', f'{idx}_unaligned_bbox.npy')
        mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_axis_align_matrix(self, idx):
        if False:
            while True:
                i = 10
        matrix_file = osp.join(self.root_dir, 'scannet_instance_data', f'{idx}_axis_align_matrix.npy')
        mmcv.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def get_images(self, idx):
        if False:
            print('Hello World!')
        paths = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.jpg'):
                paths.append(osp.join('posed_images', idx, file))
        return paths

    def get_extrinsics(self, idx):
        if False:
            for i in range(10):
                print('nop')
        extrinsics = []
        path = osp.join(self.root_dir, 'posed_images', idx)
        for file in sorted(os.listdir(path)):
            if file.endswith('.txt') and (not file == 'intrinsic.txt'):
                extrinsics.append(np.loadtxt(osp.join(path, file)))
        return extrinsics

    def get_intrinsics(self, idx):
        if False:
            i = 10
            return i + 15
        matrix_file = osp.join(self.root_dir, 'posed_images', idx, 'intrinsic.txt')
        mmcv.check_file_exist(matrix_file)
        return np.loadtxt(matrix_file)

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        if False:
            return 10
        'Get data infos.\n\n        This method gets information from the raw data.\n\n        Args:\n            num_workers (int, optional): Number of threads to be used.\n                Default: 4.\n            has_label (bool, optional): Whether the data has label.\n                Default: True.\n            sample_id_list (list[int], optional): Index list of the sample.\n                Default: None.\n\n        Returns:\n            infos (list[dict]): Information of the raw data.\n        '

        def process_single_scene(sample_idx):
            if False:
                print('Hello World!')
            print(f'{self.split} sample_idx: {sample_idx}')
            info = dict()
            pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            pts_filename = osp.join(self.root_dir, 'scannet_instance_data', f'{sample_idx}_vert.npy')
            points = np.load(pts_filename)
            mmcv.mkdir_or_exist(osp.join(self.root_dir, 'points'))
            points.tofile(osp.join(self.root_dir, 'points', f'{sample_idx}.bin'))
            info['pts_path'] = osp.join('points', f'{sample_idx}.bin')
            if os.path.exists(osp.join(self.root_dir, 'posed_images')):
                info['intrinsics'] = self.get_intrinsics(sample_idx)
                all_extrinsics = self.get_extrinsics(sample_idx)
                all_img_paths = self.get_images(sample_idx)
                (extrinsics, img_paths) = ([], [])
                for (extrinsic, img_path) in zip(all_extrinsics, all_img_paths):
                    if np.all(np.isfinite(extrinsic)):
                        img_paths.append(img_path)
                        extrinsics.append(extrinsic)
                info['extrinsics'] = extrinsics
                info['img_paths'] = img_paths
            if not self.test_mode:
                pts_instance_mask_path = osp.join(self.root_dir, 'scannet_instance_data', f'{sample_idx}_ins_label.npy')
                pts_semantic_mask_path = osp.join(self.root_dir, 'scannet_instance_data', f'{sample_idx}_sem_label.npy')
                pts_instance_mask = np.load(pts_instance_mask_path).astype(np.int64)
                pts_semantic_mask = np.load(pts_semantic_mask_path).astype(np.int64)
                mmcv.mkdir_or_exist(osp.join(self.root_dir, 'instance_mask'))
                mmcv.mkdir_or_exist(osp.join(self.root_dir, 'semantic_mask'))
                pts_instance_mask.tofile(osp.join(self.root_dir, 'instance_mask', f'{sample_idx}.bin'))
                pts_semantic_mask.tofile(osp.join(self.root_dir, 'semantic_mask', f'{sample_idx}.bin'))
                info['pts_instance_mask_path'] = osp.join('instance_mask', f'{sample_idx}.bin')
                info['pts_semantic_mask_path'] = osp.join('semantic_mask', f'{sample_idx}.bin')
            if has_label:
                annotations = {}
                aligned_box_label = self.get_aligned_box_label(sample_idx)
                unaligned_box_label = self.get_unaligned_box_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]
                    unaligned_box = unaligned_box_label[:, :-1]
                    classes = aligned_box_label[:, -1]
                    annotations['name'] = np.array([self.label2cat[self.cat_ids2class[classes[i]]] for i in range(annotations['gt_num'])])
                    annotations['location'] = aligned_box[:, :3]
                    annotations['dimensions'] = aligned_box[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = aligned_box
                    annotations['unaligned_location'] = unaligned_box[:, :3]
                    annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]
                    annotations['unaligned_gt_boxes_upright_depth'] = unaligned_box
                    annotations['index'] = np.arange(annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([self.cat_ids2class[classes[i]] for i in range(annotations['gt_num'])])
                axis_align_matrix = self.get_axis_align_matrix(sample_idx)
                annotations['axis_align_matrix'] = axis_align_matrix
                info['annos'] = annotations
            return info
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

class ScanNetSegData(object):
    """ScanNet dataset used to generate infos for semantic segmentation task.

    Args:
        data_root (str): Root path of the raw data.
        ann_file (str): The generated scannet infos.
        split (str, optional): Set split type of the data. Default: 'train'.
        num_points (int, optional): Number of points in each data input.
            Default: 8192.
        label_weight_func (function, optional): Function to compute the
            label weight. Default: None.
    """

    def __init__(self, data_root, ann_file, split='train', num_points=8192, label_weight_func=None):
        if False:
            return 10
        self.data_root = data_root
        self.data_infos = mmcv.load(ann_file)
        self.split = split
        assert split in ['train', 'val', 'test']
        self.num_points = num_points
        self.all_ids = np.arange(41)
        self.cat_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        self.ignore_index = len(self.cat_ids)
        self.cat_id2class = np.ones((self.all_ids.shape[0],), dtype=np.int) * self.ignore_index
        for (i, cat_id) in enumerate(self.cat_ids):
            self.cat_id2class[cat_id] = i
        self.label_weight_func = (lambda x: 1.0 / np.log(1.2 + x)) if label_weight_func is None else label_weight_func

    def get_seg_infos(self):
        if False:
            while True:
                i = 10
        if self.split == 'test':
            return
        (scene_idxs, label_weight) = self.get_scene_idxs_and_label_weight()
        save_folder = osp.join(self.data_root, 'seg_info')
        mmcv.mkdir_or_exist(save_folder)
        np.save(osp.join(save_folder, f'{self.split}_resampled_scene_idxs.npy'), scene_idxs)
        np.save(osp.join(save_folder, f'{self.split}_label_weight.npy'), label_weight)
        print(f'{self.split} resampled scene index and label weight saved')

    def _convert_to_label(self, mask):
        if False:
            while True:
                i = 10
        'Convert class_id in loaded segmentation mask to label.'
        if isinstance(mask, str):
            if mask.endswith('npy'):
                mask = np.load(mask)
            else:
                mask = np.fromfile(mask, dtype=np.int64)
        label = self.cat_id2class[mask]
        return label

    def get_scene_idxs_and_label_weight(self):
        if False:
            i = 10
            return i + 15
        'Compute scene_idxs for data sampling and label weight for loss\n        calculation.\n\n        We sample more times for scenes with more points. Label_weight is\n        inversely proportional to number of class points.\n        '
        num_classes = len(self.cat_ids)
        num_point_all = []
        label_weight = np.zeros((num_classes + 1,))
        for data_info in self.data_infos:
            label = self._convert_to_label(osp.join(self.data_root, data_info['pts_semantic_mask_path']))
            num_point_all.append(label.shape[0])
            (class_count, _) = np.histogram(label, range(num_classes + 2))
            label_weight += class_count
        sample_prob = np.array(num_point_all) / float(np.sum(num_point_all))
        num_iter = int(np.sum(num_point_all) / float(self.num_points))
        scene_idxs = []
        for idx in range(len(self.data_infos)):
            scene_idxs.extend([idx] * int(round(sample_prob[idx] * num_iter)))
        scene_idxs = np.array(scene_idxs).astype(np.int32)
        label_weight = label_weight[:-1].astype(np.float32)
        label_weight = label_weight / label_weight.sum()
        label_weight = self.label_weight_func(label_weight).astype(np.float32)
        return (scene_idxs, label_weight)