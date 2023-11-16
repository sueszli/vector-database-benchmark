import copy
import tempfile
from os import path as osp
import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from ..core.bbox import Box3DMode, CameraInstance3DBoxes, points_cam2img
from .builder import DATASETS
from .nuscenes_mono_dataset import NuScenesMonoDataset

@DATASETS.register_module()
class KittiMonoDataset(NuScenesMonoDataset):
    """Monocular 3D detection on KITTI Dataset.

    Args:
        data_root (str): Path of dataset root.
        info_file (str): Path of info file.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to False.
        eval_version (str, optional): Configuration version of evaluation.
            Defaults to None.
        version (str, optional): Dataset version. Defaults to None.
        kwargs (dict): Other arguments are the same of NuScenesMonoDataset.
    """
    CLASSES = ('Pedestrian', 'Cyclist', 'Car')

    def __init__(self, data_root, info_file, ann_file, pipeline, load_interval=1, with_velocity=False, eval_version=None, version=None, **kwargs):
        if False:
            return 10
        super().__init__(data_root=data_root, ann_file=ann_file, pipeline=pipeline, load_interval=load_interval, with_velocity=with_velocity, eval_version=eval_version, version=version, **kwargs)
        self.anno_infos = mmcv.load(info_file)
        self.bbox_code_size = 7

    def _parse_ann_info(self, img_info, ann_info):
        if False:
            while True:
                i = 10
        'Parse bbox and mask annotation.\n\n        Args:\n            ann_info (list[dict]): Annotation info of an image.\n            with_mask (bool): Whether to parse mask annotations.\n\n        Returns:\n            dict: A dict containing the following keys: bboxes, bboxes_ignore,\n                labels, masks, seg_map. "masks" are raw annotations and not\n                decoded into binary masks.\n        '
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
        for (i, ann) in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            (x1, y1, w, h) = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(-1)
                gt_bboxes_cam3d.append(bbox_cam3d)
                center2d = ann['center2d'][:2]
                depth = ann['center2d'][2]
                centers2d.append(center2d)
                depths.append(depth)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size), dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros(0, dtype=np.float32)
        gt_bboxes_cam3d = CameraInstance3DBoxes(gt_bboxes_cam3d, box_dim=gt_bboxes_cam3d.shape[-1], origin=(0.5, 0.5, 0.5))
        gt_labels_3d = copy.deepcopy(gt_labels)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        seg_map = img_info['filename'].replace('jpg', 'png')
        ann = dict(bboxes=gt_bboxes, labels=gt_labels, gt_bboxes_3d=gt_bboxes_cam3d, gt_labels_3d=gt_labels_3d, centers2d=centers2d, depths=depths, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann, seg_map=seg_map)
        return ann

    def format_results(self, outputs, pklfile_prefix=None, submission_prefix=None):
        if False:
            for i in range(10):
                print('nop')
        'Format the results to pkl file.\n\n        Args:\n            outputs (list[dict]): Testing results of the dataset.\n            pklfile_prefix (str): The prefix of pkl files. It includes\n                the file path and the prefix of filename, e.g., "a/b/prefix".\n                If not specified, a temp file will be created. Default: None.\n            submission_prefix (str): The prefix of submitted files. It\n                includes the file path and the prefix of filename, e.g.,\n                "a/b/prefix". If not specified, a temp file will be created.\n                Default: None.\n\n        Returns:\n            tuple: (result_files, tmp_dir), result_files is a dict containing\n                the json filepaths, tmp_dir is the temporal directory created\n                for saving json files when jsonfile_prefix is not specified.\n        '
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        if not isinstance(outputs[0], dict):
            result_files = self.bbox2result_kitti2d(outputs, self.CLASSES, pklfile_prefix, submission_prefix)
        elif 'pts_bbox' in outputs[0] or 'img_bbox' in outputs[0] or 'img_bbox2d' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = submission_prefix + name
                else:
                    submission_prefix_ = None
                if '2d' in name:
                    result_files_ = self.bbox2result_kitti2d(results_, self.CLASSES, pklfile_prefix_, submission_prefix_)
                else:
                    result_files_ = self.bbox2result_kitti(results_, self.CLASSES, pklfile_prefix_, submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES, pklfile_prefix, submission_prefix)
        return (result_files, tmp_dir)

    def evaluate(self, results, metric=None, logger=None, pklfile_prefix=None, submission_prefix=None, show=False, out_dir=None, pipeline=None):
        if False:
            return 10
        'Evaluation in KITTI protocol.\n\n        Args:\n            results (list[dict]): Testing results of the dataset.\n            metric (str | list[str], optional): Metrics to be evaluated.\n                Defaults to None.\n            logger (logging.Logger | str, optional): Logger used for printing\n                related information during evaluation. Default: None.\n            pklfile_prefix (str, optional): The prefix of pkl files, including\n                the file path and the prefix of filename, e.g., "a/b/prefix".\n                If not specified, a temp file will be created. Default: None.\n            submission_prefix (str, optional): The prefix of submission data.\n                If not specified, the submission data will not be generated.\n            show (bool, optional): Whether to visualize.\n                Default: False.\n            out_dir (str, optional): Path to save the visualization results.\n                Default: None.\n            pipeline (list[dict], optional): raw data loading for showing.\n                Default: None.\n\n        Returns:\n            dict[str, float]: Results of each evaluation metric.\n        '
        (result_files, tmp_dir) = self.format_results(results, pklfile_prefix)
        from mmdet3d.core.evaluation import kitti_eval
        gt_annos = [info['annos'] for info in self.anno_infos]
        if isinstance(result_files, dict):
            ap_dict = dict()
            for (name, result_files_) in result_files.items():
                eval_types = ['bbox', 'bev', '3d']
                if '2d' in name:
                    eval_types = ['bbox']
                (ap_result_str, ap_dict_) = kitti_eval(gt_annos, result_files_, self.CLASSES, eval_types=eval_types)
                for (ap_type, ap) in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))
                print_log(f'Results of {name}:\n' + ap_result_str, logger=logger)
        else:
            if metric == 'img_bbox2d':
                (ap_result_str, ap_dict) = kitti_eval(gt_annos, result_files, self.CLASSES, eval_types=['bbox'])
            else:
                (ap_result_str, ap_dict) = kitti_eval(gt_annos, result_files, self.CLASSES)
            print_log('\n' + ap_result_str, logger=logger)
        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict

    def bbox2result_kitti(self, net_outputs, class_names, pklfile_prefix=None, submission_prefix=None):
        if False:
            for i in range(10):
                print('nop')
        'Convert 3D detection results to kitti format for evaluation and test\n        submission.\n\n        Args:\n            net_outputs (list[np.ndarray]): List of array storing the\n                inferenced bounding boxes and scores.\n            class_names (list[String]): A list of class names.\n            pklfile_prefix (str): The prefix of pkl file.\n            submission_prefix (str): The prefix of submission file.\n\n        Returns:\n            list[dict]: A list of dictionaries with the kitti format.\n        '
        assert len(net_outputs) == len(self.anno_infos)
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)
        det_annos = []
        print('\nConverting prediction to KITTI format')
        for (idx, pred_dicts) in enumerate(mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.anno_infos[idx]
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['image_shape'][:2]
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [], 'location': [], 'rotation_y': [], 'score': []}
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']
                for (box, box_lidar, bbox, score, label) in zip(box_preds, box_preds_lidar, box_2d_preds, scores, label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-np.arctan2(box[0], box[2]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)
                anno = {k: np.stack(v) for (k, v) in anno.items()}
                annos.append(anno)
            else:
                anno = {'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]), 'alpha': np.array([]), 'bbox': np.zeros([0, 4]), 'dimensions': np.zeros([0, 3]), 'location': np.zeros([0, 3]), 'rotation_y': np.array([]), 'score': np.array([])}
                annos.append(anno)
            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']
                    for idx in range(len(bbox)):
                        print('{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(anno['name'][idx], anno['alpha'][idx], bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3], dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0], loc[idx][1], loc[idx][2], anno['rotation_y'][idx], anno['score'][idx]), file=f)
            annos[-1]['sample_idx'] = np.array([sample_idx] * len(annos[-1]['score']), dtype=np.int64)
            det_annos += annos
        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print('Result is saved to %s' % out)
        return det_annos

    def bbox2result_kitti2d(self, net_outputs, class_names, pklfile_prefix=None, submission_prefix=None):
        if False:
            print('Hello World!')
        'Convert 2D detection results to kitti format for evaluation and test\n        submission.\n\n        Args:\n            net_outputs (list[np.ndarray]): List of array storing the\n                inferenced bounding boxes and scores.\n            class_names (list[String]): A list of class names.\n            pklfile_prefix (str): The prefix of pkl file.\n            submission_prefix (str): The prefix of submission file.\n\n        Returns:\n            list[dict]: A list of dictionaries have the kitti format\n        '
        assert len(net_outputs) == len(self.anno_infos)
        det_annos = []
        print('\nConverting prediction to KITTI format')
        for (i, bboxes_per_sample) in enumerate(mmcv.track_iter_progress(net_outputs)):
            annos = []
            anno = dict(name=[], truncated=[], occluded=[], alpha=[], bbox=[], dimensions=[], location=[], rotation_y=[], score=[])
            sample_idx = self.anno_infos[i]['image']['image_idx']
            num_example = 0
            for label in range(len(bboxes_per_sample)):
                bbox = bboxes_per_sample[label]
                for i in range(bbox.shape[0]):
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-10)
                    anno['bbox'].append(bbox[i, :4])
                    anno['dimensions'].append(np.zeros(shape=[3], dtype=np.float32))
                    anno['location'].append(np.ones(shape=[3], dtype=np.float32) * -1000.0)
                    anno['rotation_y'].append(0.0)
                    anno['score'].append(bbox[i, 4])
                    num_example += 1
            if num_example == 0:
                annos.append(dict(name=np.array([]), truncated=np.array([]), occluded=np.array([]), alpha=np.array([]), bbox=np.zeros([0, 4]), dimensions=np.zeros([0, 3]), location=np.zeros([0, 3]), rotation_y=np.array([]), score=np.array([])))
            else:
                anno = {k: np.stack(v) for (k, v) in anno.items()}
                annos.append(anno)
            annos[-1]['sample_idx'] = np.array([sample_idx] * num_example, dtype=np.int64)
            det_annos += annos
        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print('Result is saved to %s' % out)
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)
            print(f'Saving KITTI submission to {submission_prefix}')
            for (i, anno) in enumerate(det_annos):
                sample_idx = self.anno_infos[i]['image']['image_idx']
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions'][::-1]
                    for idx in range(len(bbox)):
                        print('{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(anno['name'][idx], anno['alpha'][idx], *bbox[idx], *dims[idx], *loc[idx], anno['rotation_y'][idx], anno['score'][idx]), file=f)
            print(f'Result is saved to {submission_prefix}')
        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        if False:
            print('Hello World!')
        'Convert the predicted boxes into valid ones.\n\n        Args:\n            box_dict (dict): Box dictionaries to be converted.\n                - boxes_3d (:obj:`CameraInstance3DBoxes`): 3D bounding boxes.\n                - scores_3d (torch.Tensor): Scores of boxes.\n                - labels_3d (torch.Tensor): Class labels of boxes.\n            info (dict): Data info.\n\n        Returns:\n            dict: Valid predicted boxes.\n                - bbox (np.ndarray): 2D bounding boxes.\n                - box3d_camera (np.ndarray): 3D bounding boxes in\n                    camera coordinate.\n                - scores (np.ndarray): Scores of boxes.\n                - label_preds (np.ndarray): Class label predictions.\n                - sample_idx (int): Sample index.\n        '
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        if len(box_preds) == 0:
            return dict(bbox=np.zeros([0, 4]), box3d_camera=np.zeros([0, 7]), scores=np.zeros([0]), label_preds=np.zeros([0, 4]), sample_idx=sample_idx)
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        img_shape = info['image']['image_shape']
        P2 = box_preds.tensor.new_tensor(P2)
        box_preds_camera = box_preds
        box_preds_lidar = box_preds.convert_to(Box3DMode.LIDAR, np.linalg.inv(rect @ Trv2c))
        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P2)
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = (box_2d_preds[:, 0] < image_shape[1]) & (box_2d_preds[:, 1] < image_shape[0]) & (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0)
        valid_inds = valid_cam_inds
        if valid_inds.sum() > 0:
            return dict(bbox=box_2d_preds[valid_inds, :].numpy(), box3d_camera=box_preds_camera[valid_inds].tensor.numpy(), box3d_lidar=box_preds_lidar[valid_inds].tensor.numpy(), scores=scores[valid_inds].numpy(), label_preds=labels[valid_inds].numpy(), sample_idx=sample_idx)
        else:
            return dict(bbox=np.zeros([0, 4]), box3d_camera=np.zeros([0, 7]), box3d_lidar=np.zeros([0, 7]), scores=np.zeros([0]), label_preds=np.zeros([0, 4]), sample_idx=sample_idx)