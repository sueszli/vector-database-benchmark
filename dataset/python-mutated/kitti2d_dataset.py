import mmcv
import numpy as np
from mmdet.datasets import CustomDataset
from .builder import DATASETS

@DATASETS.register_module()
class Kitti2DDataset(CustomDataset):
    """KITTI 2D Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

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
    CLASSES = ('car', 'pedestrian', 'cyclist')
    "\n    Annotation format:\n    [\n        {\n            'image': {\n                'image_idx': 0,\n                'image_path': 'training/image_2/000000.png',\n                'image_shape': array([ 370, 1224], dtype=int32)\n            },\n            'point_cloud': {\n                 'num_features': 4,\n                 'velodyne_path': 'training/velodyne/000000.bin'\n             },\n             'calib': {\n                 'P0': <np.ndarray> (4, 4),\n                 'P1': <np.ndarray> (4, 4),\n                 'P2': <np.ndarray> (4, 4),\n                 'P3': <np.ndarray> (4, 4),\n                 'R0_rect':4x4 np.array,\n                 'Tr_velo_to_cam': 4x4 np.array,\n                 'Tr_imu_to_velo': 4x4 np.array\n             },\n             'annos': {\n                 'name': <np.ndarray> (n),\n                 'truncated': <np.ndarray> (n),\n                 'occluded': <np.ndarray> (n),\n                 'alpha': <np.ndarray> (n),\n                 'bbox': <np.ndarray> (n, 4),\n                 'dimensions': <np.ndarray> (n, 3),\n                 'location': <np.ndarray> (n, 3),\n                 'rotation_y': <np.ndarray> (n),\n                 'score': <np.ndarray> (n),\n                 'index': array([0], dtype=int32),\n                 'group_ids': array([0], dtype=int32),\n                 'difficulty': array([0], dtype=int32),\n                 'num_points_in_gt': <np.ndarray> (n),\n             }\n        }\n    ]\n    "

    def load_annotations(self, ann_file):
        if False:
            for i in range(10):
                print('nop')
        'Load annotations from ann_file.\n\n        Args:\n            ann_file (str): Path of the annotation file.\n\n        Returns:\n            list[dict]: List of annotations.\n        '
        self.data_infos = mmcv.load(ann_file)
        self.cat2label = {cat_name: i for (i, cat_name) in enumerate(self.CLASSES)}
        return self.data_infos

    def _filter_imgs(self, min_size=32):
        if False:
            print('Hello World!')
        'Filter images without ground truths.'
        valid_inds = []
        for (i, img_info) in enumerate(self.data_infos):
            if len(img_info['annos']['name']) > 0:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, index):
        if False:
            i = 10
            return i + 15
        'Get annotation info according to the given index.\n\n        Args:\n            index (int): Index of the annotation data to get.\n\n        Returns:\n            dict: Annotation information consists of the following keys:\n\n                - bboxes (np.ndarray): Ground truth bboxes.\n                - labels (np.ndarray): Labels of ground truths.\n        '
        info = self.data_infos[index]
        annos = info['annos']
        gt_names = annos['name']
        gt_bboxes = annos['bbox']
        difficulty = annos['difficulty']
        selected = self.keep_arrays_by_name(gt_names, self.CLASSES)
        gt_bboxes = gt_bboxes[selected]
        gt_names = gt_names[selected]
        difficulty = difficulty[selected]
        gt_labels = np.array([self.cat2label[n] for n in gt_names])
        anns_results = dict(bboxes=gt_bboxes.astype(np.float32), labels=gt_labels)
        return anns_results

    def prepare_train_img(self, idx):
        if False:
            i = 10
            return i + 15
        'Training image preparation.\n\n        Args:\n            index (int): Index for accessing the target image data.\n\n        Returns:\n            dict: Training image data dict after preprocessing\n                corresponding to the index.\n        '
        img_raw_info = self.data_infos[idx]['image']
        img_info = dict(filename=img_raw_info['image_path'])
        ann_info = self.get_ann_info(idx)
        if len(ann_info['bboxes']) == 0:
            return None
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        if False:
            i = 10
            return i + 15
        'Prepare data for testing.\n\n        Args:\n            index (int): Index for accessing the target image data.\n\n        Returns:\n            dict: Testing image data dict after preprocessing\n                corresponding to the index.\n        '
        img_raw_info = self.data_infos[idx]['image']
        img_info = dict(filename=img_raw_info['image_path'])
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def drop_arrays_by_name(self, gt_names, used_classes):
        if False:
            while True:
                i = 10
        'Drop irrelevant ground truths by name.\n\n        Args:\n            gt_names (list[str]): Names of ground truths.\n            used_classes (list[str]): Classes of interest.\n\n        Returns:\n            np.ndarray: Indices of ground truths that will be dropped.\n        '
        inds = [i for (i, x) in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        if False:
            print('Hello World!')
        'Keep useful ground truths by name.\n\n        Args:\n            gt_names (list[str]): Names of ground truths.\n            used_classes (list[str]): Classes of interest.\n\n        Returns:\n            np.ndarray: Indices of ground truths that will be keeped.\n        '
        inds = [i for (i, x) in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def reformat_bbox(self, outputs, out=None):
        if False:
            i = 10
            return i + 15
        'Reformat bounding boxes to KITTI 2D styles.\n\n        Args:\n            outputs (list[np.ndarray]): List of arrays storing the inferenced\n                bounding boxes and scores.\n            out (str, optional): The prefix of output file.\n                Default: None.\n\n        Returns:\n            list[dict]: A list of dictionaries with the kitti 2D format.\n        '
        from mmdet3d.core.bbox.transforms import bbox2result_kitti2d
        sample_idx = [info['image']['image_idx'] for info in self.data_infos]
        result_files = bbox2result_kitti2d(outputs, self.CLASSES, sample_idx, out)
        return result_files

    def evaluate(self, result_files, eval_types=None):
        if False:
            i = 10
            return i + 15
        "Evaluation in KITTI protocol.\n\n        Args:\n            result_files (str): Path of result files.\n            eval_types (str, optional): Types of evaluation. Default: None.\n                KITTI dataset only support 'bbox' evaluation type.\n\n        Returns:\n            tuple (str, dict): Average precision results in str format\n                and average precision results in dict format.\n        "
        from mmdet3d.core.evaluation import kitti_eval
        eval_types = ['bbox'] if not eval_types else eval_types
        assert eval_types in ('bbox', ['bbox']), 'KITTI data set only evaluate bbox'
        gt_annos = [info['annos'] for info in self.data_infos]
        (ap_result_str, ap_dict) = kitti_eval(gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
        return (ap_result_str, ap_dict)