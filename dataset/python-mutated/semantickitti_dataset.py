from os import path as osp
from .builder import DATASETS
from .custom_3d import Custom3DDataset

@DATASETS.register_module()
class SemanticKITTIDataset(Custom3DDataset):
    """SemanticKITTI Dataset.

    This class serves as the API for experiments on the SemanticKITTI Dataset
    Please refer to <http://www.semantic-kitti.org/dataset.html>`_
    for data downloading

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): NO 3D box for this dataset.
            You can choose any type
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """
    CLASSES = ('unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunck', 'terrian', 'pole', 'traffic-sign')

    def __init__(self, data_root, ann_file, pipeline=None, classes=None, modality=None, box_type_3d='Lidar', filter_empty_gt=False, test_mode=False):
        if False:
            i = 10
            return i + 15
        super().__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline, classes=classes, modality=modality, box_type_3d=box_type_3d, filter_empty_gt=filter_empty_gt, test_mode=test_mode)

    def get_data_info(self, index):
        if False:
            for i in range(10):
                print('nop')
        'Get data info according to the given index.\n        Args:\n            index (int): Index of the sample data to get.\n\n        Returns:\n            dict: Data information that will be passed to the data\n                preprocessing pipelines. It includes the following keys:\n                - sample_idx (str): Sample index.\n                - pts_filename (str): Filename of point clouds.\n                - file_name (str): Filename of point clouds.\n                - ann_info (dict): Annotation info.\n        '
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
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
        'Get annotation info according to the given index.\n\n        Args:\n            index (int): Index of the annotation data to get.\n\n        Returns:\n            dict: annotation information consists of the following keys:\n\n                - pts_semantic_mask_path (str): Path of semantic masks.\n        '
        info = self.data_infos[index]
        pts_semantic_mask_path = osp.join(self.data_root, info['pts_semantic_mask_path'])
        anns_results = dict(pts_semantic_mask_path=pts_semantic_mask_path)
        return anns_results