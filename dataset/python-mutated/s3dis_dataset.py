from os import path as osp
import numpy as np
from mmdet3d.core import show_seg_result
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmseg.datasets import DATASETS as SEG_DATASETS
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .custom_3d_seg import Custom3DSegDataset
from .pipelines import Compose

@DATASETS.register_module()
class S3DISDataset(Custom3DDataset):
    """S3DIS Dataset for Detection Task.

    This class is the inner dataset for S3DIS. Since S3DIS has 6 areas, we
    often train on 5 of them and test on the remaining one. The one for
    test is Area_5 as suggested in `GSDN <https://arxiv.org/abs/2006.12356>`_.
    To concatenate 5 areas during training
    `mmdet.datasets.dataset_wrappers.ConcatDataset` should be used.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Depth' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('table', 'chair', 'sofa', 'bookcase', 'board')

    def __init__(self, data_root, ann_file, pipeline=None, classes=None, modality=None, box_type_3d='Depth', filter_empty_gt=True, test_mode=False, *kwargs):
        if False:
            print('Hello World!')
        super().__init__(*kwargs, data_root=data_root, ann_file=ann_file, pipeline=pipeline, classes=classes, modality=modality, box_type_3d=box_type_3d, filter_empty_gt=filter_empty_gt, test_mode=test_mode)

    def get_ann_info(self, index):
        if False:
            return 10
        'Get annotation info according to the given index.\n\n        Args:\n            index (int): Index of the annotation data to get.\n\n        Returns:\n            dict: annotation information consists of the following keys:\n\n                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):\n                    3D ground truth bboxes\n                - gt_labels_3d (np.ndarray): Labels of ground truths.\n                - pts_instance_mask_path (str): Path of instance masks.\n                - pts_semantic_mask_path (str): Path of semantic masks.\n        '
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(np.float32)
            gt_labels_3d = info['annos']['class'].astype(np.int64)
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0,), dtype=np.int64)
        gt_bboxes_3d = DepthInstance3DBoxes(gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], with_yaw=False, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        pts_instance_mask_path = osp.join(self.data_root, info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root, info['pts_semantic_mask_path'])
        anns_results = dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, pts_instance_mask_path=pts_instance_mask_path, pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def get_data_info(self, index):
        if False:
            return 10
        'Get data info according to the given index.\n\n        Args:\n            index (int): Index of the sample data to get.\n\n        Returns:\n            dict: Data information that will be passed to the data\n                preprocessing pipelines. It includes the following keys:\n\n                - pts_filename (str): Filename of point clouds.\n                - file_name (str): Filename of point clouds.\n                - ann_info (dict): Annotation info.\n        '
        info = self.data_infos[index]
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(pts_filename=pts_filename)
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def _build_default_pipeline(self):
        if False:
            while True:
                i = 10
        'Build the default pipeline for this dataset.'
        pipeline = [dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]), dict(type='DefaultFormatBundle3D', class_names=self.CLASSES, with_label=False), dict(type='Collect3D', keys=['points'])]
        return Compose(pipeline)

class _S3DISSegDataset(Custom3DSegDataset):
    """S3DIS Dataset for Semantic Segmentation Task.

    This class is the inner dataset for S3DIS. Since S3DIS has 6 areas, we
    often train on 5 of them and test on the remaining one.
    However, there is not a fixed train-test split of S3DIS. People often test
    on Area_5 as suggested by `SEGCloud <https://arxiv.org/abs/1710.07563>`_.
    But many papers also report the average results of 6-fold cross validation
    over the 6 areas (e.g. `DGCNN <https://arxiv.org/abs/1801.07829>`_).
    Therefore, we use an inner dataset for one area, and further use a dataset
    wrapper to concat all the provided data in different areas.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        palette (list[list[int]], optional): The palette of segmentation map.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.CLASSES).
            Defaults to None.
        scene_idxs (np.ndarray | str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
    """
    CLASSES = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')
    VALID_CLASS_IDS = tuple(range(13))
    ALL_CLASS_IDS = tuple(range(14))
    PALETTE = [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0], [255, 0, 255], [100, 100, 255], [200, 200, 100], [170, 120, 200], [255, 0, 0], [200, 100, 100], [10, 200, 100], [200, 200, 200], [50, 50, 50]]

    def __init__(self, data_root, ann_file, pipeline=None, classes=None, palette=None, modality=None, test_mode=False, ignore_index=None, scene_idxs=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline, classes=classes, palette=palette, modality=modality, test_mode=test_mode, ignore_index=ignore_index, scene_idxs=scene_idxs, **kwargs)

    def get_ann_info(self, index):
        if False:
            for i in range(10):
                print('nop')
        'Get annotation info according to the given index.\n\n        Args:\n            index (int): Index of the annotation data to get.\n\n        Returns:\n            dict: annotation information consists of the following keys:\n\n                - pts_semantic_mask_path (str): Path of semantic masks.\n        '
        info = self.data_infos[index]
        pts_semantic_mask_path = osp.join(self.data_root, info['pts_semantic_mask_path'])
        anns_results = dict(pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results

    def _build_default_pipeline(self):
        if False:
            print('Hello World!')
        'Build the default pipeline for this dataset.'
        pipeline = [dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]), dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False, with_mask_3d=False, with_seg_3d=True), dict(type='PointSegClassMapping', valid_cat_ids=self.VALID_CLASS_IDS, max_cat_id=np.max(self.ALL_CLASS_IDS)), dict(type='DefaultFormatBundle3D', with_label=False, class_names=self.CLASSES), dict(type='Collect3D', keys=['points', 'pts_semantic_mask'])]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        if False:
            for i in range(10):
                print('nop')
        'Results visualization.\n\n        Args:\n            results (list[dict]): List of bounding boxes results.\n            out_dir (str): Output directory of visualization result.\n            show (bool): Visualize the results online.\n            pipeline (list[dict], optional): raw data loading for showing.\n                Default: None.\n        '
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for (i, result) in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            (points, gt_sem_mask) = self._extract_data(i, pipeline, ['points', 'pts_semantic_mask'], load_annos=True)
            points = points.numpy()
            pred_sem_mask = result['semantic_mask'].numpy()
            show_seg_result(points, gt_sem_mask, pred_sem_mask, out_dir, file_name, np.array(self.PALETTE), self.ignore_index, show)

    def get_scene_idxs(self, scene_idxs):
        if False:
            i = 10
            return i + 15
        'Compute scene_idxs for data sampling.\n\n        We sample more times for scenes with more points.\n        '
        if not self.test_mode and scene_idxs is None:
            raise NotImplementedError('please provide re-sampled scene indexes for training')
        return super().get_scene_idxs(scene_idxs)

@DATASETS.register_module()
@SEG_DATASETS.register_module()
class S3DISSegDataset(_S3DISSegDataset):
    """S3DIS Dataset for Semantic Segmentation Task.

    This class serves as the API for experiments on the S3DIS Dataset.
    It wraps the provided datasets of different areas.
    We don't use `mmdet.datasets.dataset_wrappers.ConcatDataset` because we
    need to concat the `scene_idxs` of different areas.

    Please refer to the `google form <https://docs.google.com/forms/d/e/1FAIpQL
    ScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1>`_ for
    data downloading.

    Args:
        data_root (str): Path of dataset root.
        ann_files (list[str]): Path of several annotation files.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        palette (list[list[int]], optional): The palette of segmentation map.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.CLASSES).
            Defaults to None.
        scene_idxs (list[np.ndarray] | list[str], optional): Precomputed index
            to load data. For scenes with many points, we may sample it several
            times. Defaults to None.
    """

    def __init__(self, data_root, ann_files, pipeline=None, classes=None, palette=None, modality=None, test_mode=False, ignore_index=None, scene_idxs=None, **kwargs):
        if False:
            i = 10
            return i + 15
        ann_files = self._check_ann_files(ann_files)
        scene_idxs = self._check_scene_idxs(scene_idxs, len(ann_files))
        super().__init__(data_root=data_root, ann_file=ann_files[0], pipeline=pipeline, classes=classes, palette=palette, modality=modality, test_mode=test_mode, ignore_index=ignore_index, scene_idxs=scene_idxs[0], **kwargs)
        datasets = [_S3DISSegDataset(data_root=data_root, ann_file=ann_files[i], pipeline=pipeline, classes=classes, palette=palette, modality=modality, test_mode=test_mode, ignore_index=ignore_index, scene_idxs=scene_idxs[i], **kwargs) for i in range(len(ann_files))]
        self.concat_data_infos([dst.data_infos for dst in datasets])
        self.concat_scene_idxs([dst.scene_idxs for dst in datasets])
        if not self.test_mode:
            self._set_group_flag()

    def concat_data_infos(self, data_infos):
        if False:
            print('Hello World!')
        'Concat data_infos from several datasets to form self.data_infos.\n\n        Args:\n            data_infos (list[list[dict]])\n        '
        self.data_infos = [info for one_data_infos in data_infos for info in one_data_infos]

    def concat_scene_idxs(self, scene_idxs):
        if False:
            while True:
                i = 10
        'Concat scene_idxs from several datasets to form self.scene_idxs.\n\n        Needs to manually add offset to scene_idxs[1, 2, ...].\n\n        Args:\n            scene_idxs (list[np.ndarray])\n        '
        self.scene_idxs = np.array([], dtype=np.int32)
        offset = 0
        for one_scene_idxs in scene_idxs:
            self.scene_idxs = np.concatenate([self.scene_idxs, one_scene_idxs + offset]).astype(np.int32)
            offset = np.unique(self.scene_idxs).max() + 1

    @staticmethod
    def _duplicate_to_list(x, num):
        if False:
            while True:
                i = 10
        'Repeat x `num` times to form a list.'
        return [x for _ in range(num)]

    def _check_ann_files(self, ann_file):
        if False:
            i = 10
            return i + 15
        'Make ann_files as list/tuple.'
        if not isinstance(ann_file, (list, tuple)):
            ann_file = self._duplicate_to_list(ann_file, 1)
        return ann_file

    def _check_scene_idxs(self, scene_idx, num):
        if False:
            i = 10
            return i + 15
        'Make scene_idxs as list/tuple.'
        if scene_idx is None:
            return self._duplicate_to_list(scene_idx, num)
        if isinstance(scene_idx, str):
            return self._duplicate_to_list(scene_idx, num)
        if isinstance(scene_idx[0], str):
            return scene_idx
        if isinstance(scene_idx[0], (list, tuple, np.ndarray)):
            return scene_idx
        return self._duplicate_to_list(scene_idx, num)