import tempfile
import warnings
from os import path as osp
import mmcv
import numpy as np
from torch.utils.data import Dataset
from ..core.bbox import get_box_type
from .builder import DATASETS
from .pipelines import Compose
from .utils import extract_result_dict, get_loading_pipeline

@DATASETS.register_module()
class Custom3DDataset(Dataset):
    """Customized 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.

    .. code-block:: none

    [
        {'sample_idx':
         'lidar_points': {'lidar_path': velodyne_path,
                           ....
                         },
         'annos': {'box_type_3d':  (str)  'LiDAR/Camera/Depth'
                   'gt_bboxes_3d':  <np.ndarray> (n, 7)
                   'gt_names':  [list]
                   ....
               }
         'calib': { .....}
         'images': { .....}
        }
    ]

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
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    def __init__(self, data_root, ann_file, pipeline=None, classes=None, modality=None, box_type_3d='LiDAR', filter_empty_gt=True, test_mode=False, file_client_args=dict(backend='disk')):
        if False:
            return 10
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        (self.box_type_3d, self.box_mode_3d) = get_box_type(box_type_3d)
        self.CLASSES = self.get_classes(classes)
        self.file_client = mmcv.FileClient(**file_client_args)
        self.cat2id = {name: i for (i, name) in enumerate(self.CLASSES)}
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(open(local_path, 'rb'))
        else:
            warnings.warn(f'The used MMCV version does not have get_local_path. We treat the {self.ann_file} as local paths and it might cause errors if the path is not a local path. Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        if not self.test_mode:
            self._set_group_flag()

    def load_annotations(self, ann_file):
        if False:
            return 10
        'Load annotations from ann_file.\n\n        Args:\n            ann_file (str): Path of the annotation file.\n\n        Returns:\n            list[dict]: List of annotations.\n        '
        return mmcv.load(ann_file, file_format='pkl')

    def get_data_info(self, index):
        if False:
            return 10
        'Get data info according to the given index.\n\n        Args:\n            index (int): Index of the sample data to get.\n\n        Returns:\n            dict: Data information that will be passed to the data\n                preprocessing pipelines. It includes the following keys:\n\n                - sample_idx (str): Sample index.\n                - pts_filename (str): Filename of point clouds.\n                - file_name (str): Filename of point clouds.\n                - ann_info (dict): Annotation info.\n        '
        info = self.data_infos[index]
        sample_idx = info['sample_idx']
        pts_filename = osp.join(self.data_root, info['lidar_points']['lidar_path'])
        input_dict = dict(pts_filename=pts_filename, sample_idx=sample_idx, file_name=pts_filename)
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        if False:
            while True:
                i = 10
        'Get annotation info according to the given index.\n\n        Args:\n            index (int): Index of the annotation data to get.\n\n        Returns:\n            dict: Annotation information consists of the following keys:\n\n                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):\n                    3D ground truth bboxes\n                - gt_labels_3d (np.ndarray): Labels of ground truths.\n                - gt_names (list[str]): Class names of ground truths.\n        '
        info = self.data_infos[index]
        gt_bboxes_3d = info['annos']['gt_bboxes_3d']
        gt_names_3d = info['annos']['gt_names']
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)
        ori_box_type_3d = info['annos']['box_type_3d']
        (ori_box_type_3d, _) = get_box_type(ori_box_type_3d)
        gt_bboxes_3d = ori_box_type_3d(gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        anns_results = dict(gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, gt_names=gt_names_3d)
        return anns_results

    def pre_pipeline(self, results):
        if False:
            print('Hello World!')
        'Initialization before data preparation.\n\n        Args:\n            results (dict): Dict before data preprocessing.\n\n                - img_fields (list): Image fields.\n                - bbox3d_fields (list): 3D bounding boxes fields.\n                - pts_mask_fields (list): Mask fields of points.\n                - pts_seg_fields (list): Mask fields of point segments.\n                - bbox_fields (list): Fields of bounding boxes.\n                - mask_fields (list): Fields of masks.\n                - seg_fields (list): Segment fields.\n                - box_type_3d (str): 3D box type.\n                - box_mode_3d (str): 3D box mode.\n        '
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_train_data(self, index):
        if False:
            return 10
        'Training data preparation.\n\n        Args:\n            index (int): Index for accessing the target data.\n\n        Returns:\n            dict: Training data dict of the corresponding index.\n        '
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def prepare_test_data(self, index):
        if False:
            print('Hello World!')
        'Prepare data for testing.\n\n        Args:\n            index (int): Index for accessing the target data.\n\n        Returns:\n            dict: Testing data dict of the corresponding index.\n        '
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @classmethod
    def get_classes(cls, classes=None):
        if False:
            for i in range(10):
                print('nop')
        'Get class names of current dataset.\n\n        Args:\n            classes (Sequence[str] | str): If classes is None, use\n                default CLASSES defined by builtin dataset. If classes is a\n                string, take it as a file name. The file contains the name of\n                classes where each line contains one class name. If classes is\n                a tuple or list, override the CLASSES defined by the dataset.\n\n        Return:\n            list[str]: A list of class names.\n        '
        if classes is None:
            return cls.CLASSES
        if isinstance(classes, str):
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def format_results(self, outputs, pklfile_prefix=None, submission_prefix=None):
        if False:
            return 10
        'Format the results to pkl file.\n\n        Args:\n            outputs (list[dict]): Testing results of the dataset.\n            pklfile_prefix (str): The prefix of pkl files. It includes\n                the file path and the prefix of filename, e.g., "a/b/prefix".\n                If not specified, a temp file will be created. Default: None.\n\n        Returns:\n            tuple: (outputs, tmp_dir), outputs is the detection results,\n                tmp_dir is the temporal directory created for saving json\n                files when ``jsonfile_prefix`` is not specified.\n        '
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
            out = f'{pklfile_prefix}.pkl'
        mmcv.dump(outputs, out)
        return (outputs, tmp_dir)

    def evaluate(self, results, metric=None, iou_thr=(0.25, 0.5), logger=None, show=False, out_dir=None, pipeline=None):
        if False:
            i = 10
            return i + 15
        'Evaluate.\n\n        Evaluation in indoor protocol.\n\n        Args:\n            results (list[dict]): List of results.\n            metric (str | list[str], optional): Metrics to be evaluated.\n                Defaults to None.\n            iou_thr (list[float]): AP IoU thresholds. Defaults to (0.25, 0.5).\n            logger (logging.Logger | str, optional): Logger used for printing\n                related information during evaluation. Defaults to None.\n            show (bool, optional): Whether to visualize.\n                Default: False.\n            out_dir (str, optional): Path to save the visualization results.\n                Default: None.\n            pipeline (list[dict], optional): raw data loading for showing.\n                Default: None.\n\n        Returns:\n            dict: Evaluation results.\n        '
        from mmdet3d.core.evaluation import indoor_eval
        assert isinstance(results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(results[0], dict), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat_id for (i, cat_id) in enumerate(self.CLASSES)}
        ret_dict = indoor_eval(gt_annos, results, iou_thr, label2cat, logger=logger, box_type_3d=self.box_type_3d, box_mode_3d=self.box_mode_3d)
        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return ret_dict

    def _build_default_pipeline(self):
        if False:
            while True:
                i = 10
        'Build the default pipeline for this dataset.'
        raise NotImplementedError(f'_build_default_pipeline is not implemented for dataset {self.__class__.__name__}')

    def _get_pipeline(self, pipeline):
        if False:
            return 10
        'Get data loading pipeline in self.show/evaluate function.\n\n        Args:\n            pipeline (list[dict]): Input pipeline. If None is given,\n                get from self.pipeline.\n        '
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn('Use default pipeline for data loading, this may cause errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)

    def _extract_data(self, index, pipeline, key, load_annos=False):
        if False:
            while True:
                i = 10
        'Load data using input pipeline and extract data according to key.\n\n        Args:\n            index (int): Index for accessing the target data.\n            pipeline (:obj:`Compose`): Composed data loading pipeline.\n            key (str | list[str]): One single or a list of data key.\n            load_annos (bool): Whether to load data annotations.\n                If True, need to set self.test_mode as False before loading.\n\n        Returns:\n            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:\n                A single or a list of loaded data.\n        '
        assert pipeline is not None, 'data loading pipeline is not provided'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode
        return data

    def __len__(self):
        if False:
            return 10
        'Return the length of data infos.\n\n        Returns:\n            int: Length of data infos.\n        '
        return len(self.data_infos)

    def _rand_another(self, idx):
        if False:
            for i in range(10):
                print('nop')
        'Randomly get another item with the same flag.\n\n        Returns:\n            int: Another index of item with the same flag.\n        '
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        'Get item from infos according to the given index.\n\n        Returns:\n            dict: Data dictionary of the corresponding index.\n        '
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _set_group_flag(self):
        if False:
            return 10
        'Set flag according to image aspect ratio.\n\n        Images with aspect ratio greater than 1 will be set as group 1,\n        otherwise group 0. In 3D datasets, they are all the same, thus are all\n        zeros.\n        '
        self.flag = np.zeros(len(self), dtype=np.uint8)