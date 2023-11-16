import tempfile
import warnings
from os import path as osp
import mmcv
import numpy as np
from torch.utils.data import Dataset
from mmseg.datasets import DATASETS as SEG_DATASETS
from .builder import DATASETS
from .pipelines import Compose
from .utils import extract_result_dict, get_loading_pipeline

@DATASETS.register_module()
@SEG_DATASETS.register_module()
class Custom3DSegDataset(Dataset):
    """Customized 3D dataset for semantic segmentation task.

    This is the base dataset of ScanNet and S3DIS dataset.

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
            unannotated points. If None is given, set to len(self.CLASSES) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (np.ndarray | str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
    """
    CLASSES = None
    VALID_CLASS_IDS = None
    ALL_CLASS_IDS = None
    PALETTE = None

    def __init__(self, data_root, ann_file, pipeline=None, classes=None, palette=None, modality=None, test_mode=False, ignore_index=None, scene_idxs=None, file_client_args=dict(backend='disk')):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.file_client = mmcv.FileClient(**file_client_args)
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file) as local_path:
                self.data_infos = self.load_annotations(open(local_path, 'rb'))
        else:
            warnings.warn(f'The used MMCV version does not have get_local_path. We treat the {self.ann_file} as local paths and it might cause errors if the path is not a local path. Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos = self.load_annotations(self.ann_file)
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self.ignore_index = len(self.CLASSES) if ignore_index is None else ignore_index
        self.scene_idxs = self.get_scene_idxs(scene_idxs)
        (self.CLASSES, self.PALETTE) = self.get_classes_and_palette(classes, palette)
        if not self.test_mode:
            self._set_group_flag()

    def load_annotations(self, ann_file):
        if False:
            print('Hello World!')
        'Load annotations from ann_file.\n\n        Args:\n            ann_file (str): Path of the annotation file.\n\n        Returns:\n            list[dict]: List of annotations.\n        '
        return mmcv.load(ann_file, file_format='pkl')

    def get_data_info(self, index):
        if False:
            i = 10
            return i + 15
        'Get data info according to the given index.\n\n        Args:\n            index (int): Index of the sample data to get.\n\n        Returns:\n            dict: Data information that will be passed to the data\n                preprocessing pipelines. It includes the following keys:\n\n                - sample_idx (str): Sample index.\n                - pts_filename (str): Filename of point clouds.\n                - file_name (str): Filename of point clouds.\n                - ann_info (dict): Annotation info.\n        '
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(pts_filename=pts_filename, sample_idx=sample_idx, file_name=pts_filename)
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        return input_dict

    def pre_pipeline(self, results):
        if False:
            for i in range(10):
                print('nop')
        'Initialization before data preparation.\n\n        Args:\n            results (dict): Dict before data preprocessing.\n\n                - img_fields (list): Image fields.\n                - pts_mask_fields (list): Mask fields of points.\n                - pts_seg_fields (list): Mask fields of point segments.\n                - mask_fields (list): Fields of masks.\n                - seg_fields (list): Segment fields.\n        '
        results['img_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['bbox3d_fields'] = []

    def prepare_train_data(self, index):
        if False:
            print('Hello World!')
        'Training data preparation.\n\n        Args:\n            index (int): Index for accessing the target data.\n\n        Returns:\n            dict: Training data dict of the corresponding index.\n        '
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        if False:
            for i in range(10):
                print('nop')
        'Prepare data for testing.\n\n        Args:\n            index (int): Index for accessing the target data.\n\n        Returns:\n            dict: Testing data dict of the corresponding index.\n        '
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_classes_and_palette(self, classes=None, palette=None):
        if False:
            while True:
                i = 10
        'Get class names of current dataset.\n\n        This function is taken from MMSegmentation.\n\n        Args:\n            classes (Sequence[str] | str): If classes is None, use\n                default CLASSES defined by builtin dataset. If classes is a\n                string, take it as a file name. The file contains the name of\n                classes where each line contains one class name. If classes is\n                a tuple or list, override the CLASSES defined by the dataset.\n                Defaults to None.\n            palette (Sequence[Sequence[int]]] | np.ndarray):\n                The palette of segmentation map. If None is given, random\n                palette will be generated. Defaults to None.\n        '
        if classes is None:
            self.custom_classes = False
            self.label_map = {cls_id: self.ignore_index for cls_id in self.ALL_CLASS_IDS}
            self.label_map.update({cls_id: i for (i, cls_id) in enumerate(self.VALID_CLASS_IDS)})
            self.label2cat = {i: cat_name for (i, cat_name) in enumerate(self.CLASSES)}
            return (self.CLASSES, self.PALETTE)
        self.custom_classes = True
        if isinstance(classes, str):
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        if self.CLASSES:
            if not set(class_names).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')
            self.VALID_CLASS_IDS = [self.VALID_CLASS_IDS[self.CLASSES.index(cls_name)] for cls_name in class_names]
            self.label_map = {cls_id: self.ignore_index for cls_id in self.ALL_CLASS_IDS}
            self.label_map.update({cls_id: i for (i, cls_id) in enumerate(self.VALID_CLASS_IDS)})
            self.label2cat = {i: cat_name for (i, cat_name) in enumerate(class_names)}
        palette = [self.PALETTE[self.CLASSES.index(cls_name)] for cls_name in class_names]
        return (class_names, palette)

    def get_scene_idxs(self, scene_idxs):
        if False:
            while True:
                i = 10
        'Compute scene_idxs for data sampling.\n\n        We sample more times for scenes with more points.\n        '
        if self.test_mode:
            return np.arange(len(self.data_infos)).astype(np.int32)
        if scene_idxs is None:
            scene_idxs = np.arange(len(self.data_infos))
        if isinstance(scene_idxs, str):
            with self.file_client.get_local_path(scene_idxs) as local_path:
                scene_idxs = np.load(local_path)
        else:
            scene_idxs = np.array(scene_idxs)
        return scene_idxs.astype(np.int32)

    def format_results(self, outputs, pklfile_prefix=None, submission_prefix=None):
        if False:
            print('Hello World!')
        'Format the results to pkl file.\n\n        Args:\n            outputs (list[dict]): Testing results of the dataset.\n            pklfile_prefix (str): The prefix of pkl files. It includes\n                the file path and the prefix of filename, e.g., "a/b/prefix".\n                If not specified, a temp file will be created. Default: None.\n\n        Returns:\n            tuple: (outputs, tmp_dir), outputs is the detection results,\n                tmp_dir is the temporal directory created for saving json\n                files when ``jsonfile_prefix`` is not specified.\n        '
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
            out = f'{pklfile_prefix}.pkl'
        mmcv.dump(outputs, out)
        return (outputs, tmp_dir)

    def evaluate(self, results, metric=None, logger=None, show=False, out_dir=None, pipeline=None):
        if False:
            return 10
        'Evaluate.\n\n        Evaluation in semantic segmentation protocol.\n\n        Args:\n            results (list[dict]): List of results.\n            metric (str | list[str]): Metrics to be evaluated.\n            logger (logging.Logger | str, optional): Logger used for printing\n                related information during evaluation. Defaults to None.\n            show (bool, optional): Whether to visualize.\n                Defaults to False.\n            out_dir (str, optional): Path to save the visualization results.\n                Defaults to None.\n            pipeline (list[dict], optional): raw data loading for showing.\n                Default: None.\n\n        Returns:\n            dict: Evaluation results.\n        '
        from mmdet3d.core.evaluation import seg_eval
        assert isinstance(results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(results[0], dict), f'Expect elements in results to be dict, got {type(results[0])}.'
        load_pipeline = self._get_pipeline(pipeline)
        pred_sem_masks = [result['semantic_mask'] for result in results]
        gt_sem_masks = [self._extract_data(i, load_pipeline, 'pts_semantic_mask', load_annos=True) for i in range(len(self.data_infos))]
        ret_dict = seg_eval(gt_sem_masks, pred_sem_masks, self.label2cat, self.ignore_index, logger=logger)
        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return ret_dict

    def _rand_another(self, idx):
        if False:
            i = 10
            return i + 15
        'Randomly get another item with the same flag.\n\n        Returns:\n            int: Another index of item with the same flag.\n        '
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _build_default_pipeline(self):
        if False:
            return 10
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
            return 10
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
            print('Hello World!')
        'Return the length of scene_idxs.\n\n        Returns:\n            int: Length of data infos.\n        '
        return len(self.scene_idxs)

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        'Get item from infos according to the given index.\n\n        In indoor scene segmentation task, each scene contains millions of\n        points. However, we only sample less than 10k points within a patch\n        each time. Therefore, we use `scene_idxs` to re-sample different rooms.\n\n        Returns:\n            dict: Data dictionary of the corresponding index.\n        '
        scene_idx = self.scene_idxs[idx]
        if self.test_mode:
            return self.prepare_test_data(scene_idx)
        while True:
            data = self.prepare_train_data(scene_idx)
            if data is None:
                idx = self._rand_another(idx)
                scene_idx = self.scene_idxs[idx]
                continue
            return data

    def _set_group_flag(self):
        if False:
            return 10
        'Set flag according to image aspect ratio.\n\n        Images with aspect ratio greater than 1 will be set as group 1,\n        otherwise group 0. In 3D datasets, they are all the same, thus are all\n        zeros.\n        '
        self.flag = np.zeros(len(self), dtype=np.uint8)