import os
import tempfile
from os import path as osp
import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from ..core.bbox import Box3DMode, points_cam2img
from .builder import DATASETS
from .kitti_dataset import KittiDataset

@DATASETS.register_module()
class WaymoDataset(KittiDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list(float), optional): The range of point cloud used
            to filter invalid predicted boxes.
            Default: [-85, -85, -5, 85, 85, 5].
    """
    CLASSES = ('Car', 'Cyclist', 'Pedestrian')

    def __init__(self, data_root, ann_file, split, pts_prefix='velodyne', pipeline=None, classes=None, modality=None, box_type_3d='LiDAR', filter_empty_gt=True, test_mode=False, load_interval=1, pcd_limit_range=[-85, -85, -5, 85, 85, 5], **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(data_root=data_root, ann_file=ann_file, split=split, pts_prefix=pts_prefix, pipeline=pipeline, classes=classes, modality=modality, box_type_3d=box_type_3d, filter_empty_gt=filter_empty_gt, test_mode=test_mode, pcd_limit_range=pcd_limit_range, **kwargs)
        self.data_infos = self.data_infos[::load_interval]
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]

    def _get_pts_filename(self, idx):
        if False:
            for i in range(10):
                print('nop')
        pts_filename = osp.join(self.root_split, self.pts_prefix, f'{idx:07d}.bin')
        return pts_filename

    def get_data_info(self, index):
        if False:
            i = 10
            return i + 15
        'Get data info according to the given index.\n\n        Args:\n            index (int): Index of the sample data to get.\n\n        Returns:\n            dict: Standard input_dict consists of the\n                data information.\n\n                - sample_idx (str): sample index\n                - pts_filename (str): filename of point clouds\n                - img_prefix (str): prefix of image files\n                - img_info (dict): image info\n                - lidar2img (list[np.ndarray], optional): transformations from\n                    lidar to different cameras\n                - ann_info (dict): annotation info\n        '
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']
        img_filename = os.path.join(self.data_root, info['image']['image_path'])
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        lidar2img = P0 @ rect @ Trv2c
        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(sample_idx=sample_idx, pts_filename=pts_filename, img_prefix=None, img_info=dict(filename=img_filename), lidar2img=lidar2img)
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
        return input_dict

    def format_results(self, outputs, pklfile_prefix=None, submission_prefix=None, data_format='waymo'):
        if False:
            print('Hello World!')
        'Format the results to pkl file.\n\n        Args:\n            outputs (list[dict]): Testing results of the dataset.\n            pklfile_prefix (str): The prefix of pkl files. It includes\n                the file path and the prefix of filename, e.g., "a/b/prefix".\n                If not specified, a temp file will be created. Default: None.\n            submission_prefix (str): The prefix of submitted files. It\n                includes the file path and the prefix of filename, e.g.,\n                "a/b/prefix". If not specified, a temp file will be created.\n                Default: None.\n            data_format (str, optional): Output data format.\n                Default: \'waymo\'. Another supported choice is \'kitti\'.\n\n        Returns:\n            tuple: (result_files, tmp_dir), result_files is a dict containing\n                the json filepaths, tmp_dir is the temporal directory created\n                for saving json files when jsonfile_prefix is not specified.\n        '
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        assert 'waymo' in data_format or 'kitti' in data_format, f'invalid data_format {data_format}'
        if not isinstance(outputs[0], dict) or 'img_bbox' in outputs[0]:
            raise TypeError('Not supported type for reformat results.')
        elif 'pts_bbox' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = f'{submission_prefix}_{name}'
                else:
                    submission_prefix_ = None
                result_files_ = self.bbox2result_kitti(results_, self.CLASSES, pklfile_prefix_, submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES, pklfile_prefix, submission_prefix)
        if 'waymo' in data_format:
            from ..core.evaluation.waymo_utils.prediction_kitti_to_waymo import KITTI2Waymo
            waymo_root = osp.join(self.data_root.split('kitti_format')[0], 'waymo_format')
            if self.split == 'training':
                waymo_tfrecords_dir = osp.join(waymo_root, 'validation')
                prefix = '1'
            elif self.split == 'testing':
                waymo_tfrecords_dir = osp.join(waymo_root, 'testing')
                prefix = '2'
            else:
                raise ValueError('Not supported split value.')
            save_tmp_dir = tempfile.TemporaryDirectory()
            waymo_results_save_dir = save_tmp_dir.name
            waymo_results_final_path = f'{pklfile_prefix}.bin'
            if 'pts_bbox' in result_files:
                converter = KITTI2Waymo(result_files['pts_bbox'], waymo_tfrecords_dir, waymo_results_save_dir, waymo_results_final_path, prefix)
            else:
                converter = KITTI2Waymo(result_files, waymo_tfrecords_dir, waymo_results_save_dir, waymo_results_final_path, prefix)
            converter.convert()
            save_tmp_dir.cleanup()
        return (result_files, tmp_dir)

    def evaluate(self, results, metric='waymo', logger=None, pklfile_prefix=None, submission_prefix=None, show=False, out_dir=None, pipeline=None):
        if False:
            print('Hello World!')
        'Evaluation in KITTI protocol.\n\n        Args:\n            results (list[dict]): Testing results of the dataset.\n            metric (str | list[str], optional): Metrics to be evaluated.\n                Default: \'waymo\'. Another supported metric is \'kitti\'.\n            logger (logging.Logger | str, optional): Logger used for printing\n                related information during evaluation. Default: None.\n            pklfile_prefix (str, optional): The prefix of pkl files including\n                the file path and the prefix of filename, e.g., "a/b/prefix".\n                If not specified, a temp file will be created. Default: None.\n            submission_prefix (str, optional): The prefix of submission data.\n                If not specified, the submission data will not be generated.\n            show (bool, optional): Whether to visualize.\n                Default: False.\n            out_dir (str, optional): Path to save the visualization results.\n                Default: None.\n            pipeline (list[dict], optional): raw data loading for showing.\n                Default: None.\n\n        Returns:\n            dict[str: float]: results of each evaluation metric\n        '
        assert 'waymo' in metric or 'kitti' in metric, f'invalid metric {metric}'
        if 'kitti' in metric:
            (result_files, tmp_dir) = self.format_results(results, pklfile_prefix, submission_prefix, data_format='kitti')
            from mmdet3d.core.evaluation import kitti_eval
            gt_annos = [info['annos'] for info in self.data_infos]
            if isinstance(result_files, dict):
                ap_dict = dict()
                for (name, result_files_) in result_files.items():
                    eval_types = ['bev', '3d']
                    (ap_result_str, ap_dict_) = kitti_eval(gt_annos, result_files_, self.CLASSES, eval_types=eval_types)
                    for (ap_type, ap) in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))
                    print_log(f'Results of {name}:\n' + ap_result_str, logger=logger)
            else:
                (ap_result_str, ap_dict) = kitti_eval(gt_annos, result_files, self.CLASSES, eval_types=['bev', '3d'])
                print_log('\n' + ap_result_str, logger=logger)
        if 'waymo' in metric:
            waymo_root = osp.join(self.data_root.split('kitti_format')[0], 'waymo_format')
            if pklfile_prefix is None:
                eval_tmp_dir = tempfile.TemporaryDirectory()
                pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
            else:
                eval_tmp_dir = None
            (result_files, tmp_dir) = self.format_results(results, pklfile_prefix, submission_prefix, data_format='waymo')
            import subprocess
            ret_bytes = subprocess.check_output('mmdet3d/core/evaluation/waymo_utils/' + f'compute_detection_metrics_main {pklfile_prefix}.bin ' + f'{waymo_root}/gt.bin', shell=True)
            ret_texts = ret_bytes.decode('utf-8')
            print_log(ret_texts)
            ap_dict = {'Vehicle/L1 mAP': 0, 'Vehicle/L1 mAPH': 0, 'Vehicle/L2 mAP': 0, 'Vehicle/L2 mAPH': 0, 'Pedestrian/L1 mAP': 0, 'Pedestrian/L1 mAPH': 0, 'Pedestrian/L2 mAP': 0, 'Pedestrian/L2 mAPH': 0, 'Sign/L1 mAP': 0, 'Sign/L1 mAPH': 0, 'Sign/L2 mAP': 0, 'Sign/L2 mAPH': 0, 'Cyclist/L1 mAP': 0, 'Cyclist/L1 mAPH': 0, 'Cyclist/L2 mAP': 0, 'Cyclist/L2 mAPH': 0, 'Overall/L1 mAP': 0, 'Overall/L1 mAPH': 0, 'Overall/L2 mAP': 0, 'Overall/L2 mAPH': 0}
            mAP_splits = ret_texts.split('mAP ')
            mAPH_splits = ret_texts.split('mAPH ')
            for (idx, key) in enumerate(ap_dict.keys()):
                split_idx = int(idx / 2) + 1
                if idx % 2 == 0:
                    ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
                else:
                    ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
            ap_dict['Overall/L1 mAP'] = (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] + ap_dict['Cyclist/L1 mAP']) / 3
            ap_dict['Overall/L1 mAPH'] = (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] + ap_dict['Cyclist/L1 mAPH']) / 3
            ap_dict['Overall/L2 mAP'] = (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] + ap_dict['Cyclist/L2 mAP']) / 3
            ap_dict['Overall/L2 mAPH'] = (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] + ap_dict['Cyclist/L2 mAPH']) / 3
            if eval_tmp_dir is not None:
                eval_tmp_dir.cleanup()
        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict

    def bbox2result_kitti(self, net_outputs, class_names, pklfile_prefix=None, submission_prefix=None):
        if False:
            for i in range(10):
                print('nop')
        'Convert results to kitti format for evaluation and test submission.\n\n        Args:\n            net_outputs (List[np.ndarray]): list of array storing the\n                bbox and score\n            class_nanes (List[String]): A list of class names\n            pklfile_prefix (str): The prefix of pkl file.\n            submission_prefix (str): The prefix of submission file.\n\n        Returns:\n            List[dict]: A list of dict have the kitti 3d format\n        '
        assert len(net_outputs) == len(self.data_infos), 'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)
        det_annos = []
        print('\nConverting prediction to KITTI format')
        for (idx, pred_dicts) in enumerate(mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['image_shape'][:2]
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']
                anno = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [], 'location': [], 'rotation_y': [], 'score': []}
                for (box, box_lidar, bbox, score, label) in zip(box_preds, box_preds_lidar, box_2d_preds, scores, label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)
                anno = {k: np.stack(v) for (k, v) in anno.items()}
                annos.append(anno)
                if submission_prefix is not None:
                    curr_file = f'{submission_prefix}/{sample_idx:07d}.txt'
                    with open(curr_file, 'w') as f:
                        bbox = anno['bbox']
                        loc = anno['location']
                        dims = anno['dimensions']
                        for idx in range(len(bbox)):
                            print('{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(anno['name'][idx], anno['alpha'][idx], bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3], dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0], loc[idx][1], loc[idx][2], anno['rotation_y'][idx], anno['score'][idx]), file=f)
            else:
                annos.append({'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]), 'alpha': np.array([]), 'bbox': np.zeros([0, 4]), 'dimensions': np.zeros([0, 3]), 'location': np.zeros([0, 3]), 'rotation_y': np.array([]), 'score': np.array([])})
            annos[-1]['sample_idx'] = np.array([sample_idx] * len(annos[-1]['score']), dtype=np.int64)
            det_annos += annos
        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')
        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        if False:
            i = 10
            return i + 15
        'Convert the boxes into valid format.\n\n        Args:\n            box_dict (dict): Bounding boxes to be converted.\n\n                - boxes_3d (:obj:``LiDARInstance3DBoxes``): 3D bounding boxes.\n                - scores_3d (np.ndarray): Scores of predicted boxes.\n                - labels_3d (np.ndarray): Class labels of predicted boxes.\n            info (dict): Dataset information dictionary.\n\n        Returns:\n            dict: Valid boxes after conversion.\n\n                - bbox (np.ndarray): 2D bounding boxes (in camera 0).\n                - box3d_camera (np.ndarray): 3D boxes in camera coordinates.\n                - box3d_lidar (np.ndarray): 3D boxes in lidar coordinates.\n                - scores (np.ndarray): Scores of predicted boxes.\n                - label_preds (np.ndarray): Class labels of predicted boxes.\n                - sample_idx (np.ndarray): Sample index.\n        '
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)
        if len(box_preds) == 0:
            return dict(bbox=np.zeros([0, 4]), box3d_camera=np.zeros([0, 7]), box3d_lidar=np.zeros([0, 7]), scores=np.zeros([0]), label_preds=np.zeros([0, 4]), sample_idx=sample_idx)
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        P0 = box_preds.tensor.new_tensor(P0)
        box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)
        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P0)
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = (box_preds.center > limit_range[:3]) & (box_preds.center < limit_range[3:])
        valid_inds = valid_pcd_inds.all(-1)
        if valid_inds.sum() > 0:
            return dict(bbox=box_2d_preds[valid_inds, :].numpy(), box3d_camera=box_preds_camera[valid_inds].tensor.numpy(), box3d_lidar=box_preds[valid_inds].tensor.numpy(), scores=scores[valid_inds].numpy(), label_preds=labels[valid_inds].numpy(), sample_idx=sample_idx)
        else:
            return dict(bbox=np.zeros([0, 4]), box3d_camera=np.zeros([0, 7]), box3d_lidar=np.zeros([0, 7]), scores=np.zeros([0]), label_preds=np.zeros([0, 4]), sample_idx=sample_idx)