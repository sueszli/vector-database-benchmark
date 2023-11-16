import copy
import tempfile
import warnings
from os import path as osp
import mmcv
import numpy as np
import pyquaternion
import torch
from nuscenes.utils.data_classes import Box as NuScenesBox
from mmdet3d.core import bbox3d2result, box3d_multiclass_nms, xywhr2xyxyr
from mmdet.datasets import CocoDataset
from ..core import show_multi_modality_result
from ..core.bbox import CameraInstance3DBoxes, get_box_type
from .builder import DATASETS
from .pipelines import Compose
from .utils import extract_result_dict, get_loading_pipeline

@DATASETS.register_module()
class NuScenesMonoDataset(CocoDataset):
    """Monocular 3D detection on NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        data_root (str): Path of dataset root.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'Camera' in this class. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        eval_version (str, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        version (str, optional): Dataset version. Defaults to 'v1.0-trainval'.
    """
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier')
    DefaultAttribute = {'car': 'vehicle.parked', 'pedestrian': 'pedestrian.moving', 'trailer': 'vehicle.parked', 'truck': 'vehicle.parked', 'bus': 'vehicle.moving', 'motorcycle': 'cycle.without_rider', 'construction_vehicle': 'vehicle.parked', 'bicycle': 'cycle.without_rider', 'barrier': '', 'traffic_cone': ''}
    ErrNameMapping = {'trans_err': 'mATE', 'scale_err': 'mASE', 'orient_err': 'mAOE', 'vel_err': 'mAVE', 'attr_err': 'mAAE'}

    def __init__(self, data_root, ann_file, pipeline, load_interval=1, with_velocity=True, modality=None, box_type_3d='Camera', eval_version='detection_cvpr_2019', use_valid_flag=False, version='v1.0-trainval', classes=None, img_prefix='', seg_prefix=None, proposal_file=None, test_mode=False, filter_empty_gt=True, file_client_args=dict(backend='disk')):
        if False:
            print('Hello World!')
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        self.file_client = mmcv.FileClient(**file_client_args)
        with self.file_client.get_local_path(self.ann_file) as local_path:
            self.data_infos = self.load_annotations(local_path)
        if self.proposal_file is not None:
            with self.file_client.get_local_path(self.proposal_file) as local_path:
                self.proposals = self.load_proposals(local_path)
        else:
            self.proposals = None
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            self._set_group_flag()
        self.pipeline = Compose(pipeline)
        self.load_interval = load_interval
        self.with_velocity = with_velocity
        self.modality = modality
        (self.box_type_3d, self.box_mode_3d) = get_box_type(box_type_3d)
        self.eval_version = eval_version
        self.use_valid_flag = use_valid_flag
        self.bbox_code_size = 9
        self.version = version
        if self.eval_version is not None:
            from nuscenes.eval.detection.config import config_factory
            self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(use_camera=True, use_lidar=False, use_radar=False, use_map=False, use_external=False)

    def pre_pipeline(self, results):
        if False:
            return 10
        'Initialization before data preparation.\n\n        Args:\n            results (dict): Dict before data preprocessing.\n\n                - img_fields (list): Image fields.\n                - bbox3d_fields (list): 3D bounding boxes fields.\n                - pts_mask_fields (list): Mask fields of points.\n                - pts_seg_fields (list): Mask fields of point segments.\n                - bbox_fields (list): Fields of bounding boxes.\n                - mask_fields (list): Fields of masks.\n                - seg_fields (list): Segment fields.\n                - box_type_3d (str): 3D box type.\n                - box_mode_3d (str): 3D box mode.\n        '
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def _parse_ann_info(self, img_info, ann_info):
        if False:
            while True:
                i = 10
        'Parse bbox annotation.\n\n        Args:\n            img_info (list[dict]): Image info.\n            ann_info (list[dict]): Annotation info of an image.\n\n        Returns:\n            dict: A dict containing the following keys: bboxes, labels,\n                gt_bboxes_3d, gt_labels_3d, attr_labels, centers2d,\n                depths, bboxes_ignore, masks, seg_map\n        '
        gt_bboxes = []
        gt_labels = []
        attr_labels = []
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
                attr_labels.append(ann['attribute_id'])
                gt_masks_ann.append(ann.get('segmentation', None))
                bbox_cam3d = np.array(ann['bbox_cam3d']).reshape(1, -1)
                velo_cam3d = np.array(ann['velo_cam3d']).reshape(1, 2)
                nan_mask = np.isnan(velo_cam3d[:, 0])
                velo_cam3d[nan_mask] = [0.0, 0.0]
                bbox_cam3d = np.concatenate([bbox_cam3d, velo_cam3d], axis=-1)
                gt_bboxes_cam3d.append(bbox_cam3d.squeeze())
                center2d = ann['center2d'][:2]
                depth = ann['center2d'][2]
                centers2d.append(center2d)
                depths.append(depth)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            attr_labels = np.array(attr_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            attr_labels = np.array([], dtype=np.int64)
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
        ann = dict(bboxes=gt_bboxes, labels=gt_labels, gt_bboxes_3d=gt_bboxes_cam3d, gt_labels_3d=gt_labels_3d, attr_labels=attr_labels, centers2d=centers2d, depths=depths, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann, seg_map=seg_map)
        return ann

    def get_attr_name(self, attr_idx, label_name):
        if False:
            print('Hello World!')
        'Get attribute from predicted index.\n\n        This is a workaround to predict attribute when the predicted velocity\n        is not reliable. We map the predicted attribute index to the one\n        in the attribute set. If it is consistent with the category, we will\n        keep it. Otherwise, we will use the default attribute.\n\n        Args:\n            attr_idx (int): Attribute index.\n            label_name (str): Predicted category name.\n\n        Returns:\n            str: Predicted attribute name.\n        '
        AttrMapping_rev2 = ['cycle.with_rider', 'cycle.without_rider', 'pedestrian.moving', 'pedestrian.standing', 'pedestrian.sitting_lying_down', 'vehicle.moving', 'vehicle.parked', 'vehicle.stopped', 'None']
        if label_name == 'car' or label_name == 'bus' or label_name == 'truck' or (label_name == 'trailer') or (label_name == 'construction_vehicle'):
            if AttrMapping_rev2[attr_idx] == 'vehicle.moving' or AttrMapping_rev2[attr_idx] == 'vehicle.parked' or AttrMapping_rev2[attr_idx] == 'vehicle.stopped':
                return AttrMapping_rev2[attr_idx]
            else:
                return NuScenesMonoDataset.DefaultAttribute[label_name]
        elif label_name == 'pedestrian':
            if AttrMapping_rev2[attr_idx] == 'pedestrian.moving' or AttrMapping_rev2[attr_idx] == 'pedestrian.standing' or AttrMapping_rev2[attr_idx] == 'pedestrian.sitting_lying_down':
                return AttrMapping_rev2[attr_idx]
            else:
                return NuScenesMonoDataset.DefaultAttribute[label_name]
        elif label_name == 'bicycle' or label_name == 'motorcycle':
            if AttrMapping_rev2[attr_idx] == 'cycle.with_rider' or AttrMapping_rev2[attr_idx] == 'cycle.without_rider':
                return AttrMapping_rev2[attr_idx]
            else:
                return NuScenesMonoDataset.DefaultAttribute[label_name]
        else:
            return NuScenesMonoDataset.DefaultAttribute[label_name]

    def _format_bbox(self, results, jsonfile_prefix=None):
        if False:
            i = 10
            return i + 15
        'Convert the results to the standard format.\n\n        Args:\n            results (list[dict]): Testing results of the dataset.\n            jsonfile_prefix (str): The prefix of the output jsonfile.\n                You can specify the output directory/filename by\n                modifying the jsonfile_prefix. Default: None.\n\n        Returns:\n            str: Path of the output json file.\n        '
        nusc_annos = {}
        mapped_class_names = self.CLASSES
        print('Start to convert detection format...')
        CAM_NUM = 6
        for (sample_id, det) in enumerate(mmcv.track_iter_progress(results)):
            if sample_id % CAM_NUM == 0:
                boxes_per_frame = []
                attrs_per_frame = []
            annos = []
            (boxes, attrs) = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]['token']
            (boxes, attrs) = cam_nusc_box_to_global(self.data_infos[sample_id], boxes, attrs, mapped_class_names, self.eval_detection_configs, self.eval_version)
            boxes_per_frame.extend(boxes)
            attrs_per_frame.extend(attrs)
            if (sample_id + 1) % CAM_NUM != 0:
                continue
            boxes = global_nusc_box_to_cam(self.data_infos[sample_id + 1 - CAM_NUM], boxes_per_frame, mapped_class_names, self.eval_detection_configs, self.eval_version)
            (cam_boxes3d, scores, labels) = nusc_box_to_cam_box3d(boxes)
            nms_cfg = dict(use_rotate_nms=True, nms_across_levels=False, nms_pre=4096, nms_thr=0.05, score_thr=0.01, min_bbox_size=0, max_per_frame=500)
            from mmcv import Config
            nms_cfg = Config(nms_cfg)
            cam_boxes3d_for_nms = xywhr2xyxyr(cam_boxes3d.bev)
            boxes3d = cam_boxes3d.tensor
            attrs = labels.new_tensor([attr for attr in attrs_per_frame])
            (boxes3d, scores, labels, attrs) = box3d_multiclass_nms(boxes3d, cam_boxes3d_for_nms, scores, nms_cfg.score_thr, nms_cfg.max_per_frame, nms_cfg, mlvl_attr_scores=attrs)
            cam_boxes3d = CameraInstance3DBoxes(boxes3d, box_dim=9)
            det = bbox3d2result(cam_boxes3d, scores, labels, attrs)
            (boxes, attrs) = output_to_nusc_box(det)
            (boxes, attrs) = cam_nusc_box_to_global(self.data_infos[sample_id + 1 - CAM_NUM], boxes, attrs, mapped_class_names, self.eval_detection_configs, self.eval_version)
            for (i, box) in enumerate(boxes):
                name = mapped_class_names[box.label]
                attr = self.get_attr_name(attrs[i], name)
                nusc_anno = dict(sample_token=sample_token, translation=box.center.tolist(), size=box.wlh.tolist(), rotation=box.orientation.elements.tolist(), velocity=box.velocity[:2].tolist(), detection_name=name, detection_score=box.score, attribute_name=attr)
                annos.append(nusc_anno)
            if sample_token in nusc_annos:
                nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {'meta': self.modality, 'results': nusc_annos}
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self, result_path, logger=None, metric='bbox', result_name='img_bbox'):
        if False:
            return 10
        "Evaluation for a single model in nuScenes protocol.\n\n        Args:\n            result_path (str): Path of the result file.\n            logger (logging.Logger | str, optional): Logger used for printing\n                related information during evaluation. Default: None.\n            metric (str, optional): Metric name used for evaluation.\n                Default: 'bbox'.\n            result_name (str, optional): Result name in the metric prefix.\n                Default: 'img_bbox'.\n\n        Returns:\n            dict: Dictionary of evaluation details.\n        "
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval
        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {'v1.0-mini': 'mini_val', 'v1.0-trainval': 'val'}
        nusc_eval = NuScenesEval(nusc, config=self.eval_detection_configs, result_path=result_path, eval_set=eval_set_map[self.version], output_dir=output_dir, verbose=False)
        nusc_eval.main(render_curves=True)
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for (k, v) in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for (k, v) in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for (k, v) in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix, self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        if False:
            print('Hello World!')
        'Format the results to json (standard format for COCO evaluation).\n\n        Args:\n            results (list[tuple | numpy.ndarray]): Testing results of the\n                dataset.\n            jsonfile_prefix (str): The prefix of json files. It includes\n                the file path and the prefix of filename, e.g., "a/b/prefix".\n                If not specified, a temp file will be created. Default: None.\n\n        Returns:\n            tuple: (result_files, tmp_dir), result_files is a dict containing\n                the json filepaths, tmp_dir is the temporal directory created\n                for saving json files when jsonfile_prefix is not specified.\n        '
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), 'The length of results is not equal to the dataset len: {} != {}'.format(len(results), len(self))
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = dict()
            for name in results[0]:
                if '2d' in name:
                    continue
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update({name: self._format_bbox(results_, tmp_file_)})
        return (result_files, tmp_dir)

    def evaluate(self, results, metric='bbox', logger=None, jsonfile_prefix=None, result_names=['img_bbox'], show=False, out_dir=None, pipeline=None):
        if False:
            while True:
                i = 10
        'Evaluation in nuScenes protocol.\n\n        Args:\n            results (list[dict]): Testing results of the dataset.\n            metric (str | list[str], optional): Metrics to be evaluated.\n                Default: \'bbox\'.\n            logger (logging.Logger | str, optional): Logger used for printing\n                related information during evaluation. Default: None.\n            jsonfile_prefix (str): The prefix of json files. It includes\n                the file path and the prefix of filename, e.g., "a/b/prefix".\n                If not specified, a temp file will be created. Default: None.\n            result_names (list[str], optional): Result names in the\n                metric prefix. Default: [\'img_bbox\'].\n            show (bool, optional): Whether to visualize.\n                Default: False.\n            out_dir (str, optional): Path to save the visualization results.\n                Default: None.\n            pipeline (list[dict], optional): raw data loading for showing.\n                Default: None.\n\n        Returns:\n            dict[str, float]: Results of each evaluation metric.\n        '
        (result_files, tmp_dir) = self.format_results(results, jsonfile_prefix)
        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)
        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def _extract_data(self, index, pipeline, key, load_annos=False):
        if False:
            while True:
                i = 10
        'Load data using input pipeline and extract data according to key.\n\n        Args:\n            index (int): Index for accessing the target data.\n            pipeline (:obj:`Compose`): Composed data loading pipeline.\n            key (str | list[str]): One single or a list of data key.\n            load_annos (bool): Whether to load data annotations.\n                If True, need to set self.test_mode as False before loading.\n\n        Returns:\n            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:\n                A single or a list of loaded data.\n        '
        assert pipeline is not None, 'data loading pipeline is not provided'
        img_info = self.data_infos[index]
        input_dict = dict(img_info=img_info)
        if load_annos:
            ann_info = self.get_ann_info(index)
            input_dict.update(dict(ann_info=ann_info))
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        return data

    def _get_pipeline(self, pipeline):
        if False:
            print('Hello World!')
        'Get data loading pipeline in self.show/evaluate function.\n\n        Args:\n            pipeline (list[dict]): Input pipeline. If None is given,\n                get from self.pipeline.\n        '
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn('Use default pipeline for data loading, this may cause errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)

    def _build_default_pipeline(self):
        if False:
            for i in range(10):
                print('nop')
        'Build the default pipeline for this dataset.'
        pipeline = [dict(type='LoadImageFromFileMono3D'), dict(type='DefaultFormatBundle3D', class_names=self.CLASSES, with_label=False), dict(type='Collect3D', keys=['img'])]
        return Compose(pipeline)

    def show(self, results, out_dir, show=False, pipeline=None):
        if False:
            return 10
        'Results visualization.\n\n        Args:\n            results (list[dict]): List of bounding boxes results.\n            out_dir (str): Output directory of visualization result.\n            show (bool): Whether to visualize the results online.\n                Default: False.\n            pipeline (list[dict], optional): raw data loading for showing.\n                Default: None.\n        '
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for (i, result) in enumerate(results):
            if 'img_bbox' in result.keys():
                result = result['img_bbox']
            data_info = self.data_infos[i]
            img_path = data_info['file_name']
            file_name = osp.split(img_path)[-1].split('.')[0]
            (img, img_metas) = self._extract_data(i, pipeline, ['img', 'img_metas'])
            img = img.numpy().transpose(1, 2, 0)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            pred_bboxes = result['boxes_3d']
            show_multi_modality_result(img, gt_bboxes, pred_bboxes, img_metas['cam2img'], out_dir, file_name, box_mode='camera', show=show)

def output_to_nusc_box(detection):
    if False:
        while True:
            i = 10
    'Convert the output to the box class in the nuScenes.\n\n    Args:\n        detection (dict): Detection results.\n\n            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.\n            - scores_3d (torch.Tensor): Detection scores.\n            - labels_3d (torch.Tensor): Predicted box labels.\n            - attrs_3d (torch.Tensor, optional): Predicted attributes.\n\n    Returns:\n        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.\n    '
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    attrs = None
    if 'attrs_3d' in detection:
        attrs = detection['attrs_3d'].numpy()
    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    box_dims[:, [0, 1, 2]] = box_dims[:, [2, 0, 1]]
    box_yaw = -box_yaw
    box_list = []
    for i in range(len(box3d)):
        q1 = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
        quat = q2 * q1
        velocity = (box3d.tensor[i, 7], 0.0, box3d.tensor[i, 8])
        box = NuScenesBox(box_gravity_center[i], box_dims[i], quat, label=labels[i], score=scores[i], velocity=velocity)
        box_list.append(box)
    return (box_list, attrs)

def cam_nusc_box_to_global(info, boxes, attrs, classes, eval_configs, eval_version='detection_cvpr_2019'):
    if False:
        while True:
            i = 10
    "Convert the box from camera to global coordinate.\n\n    Args:\n        info (dict): Info for a specific sample data, including the\n            calibration information.\n        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.\n        classes (list[str]): Mapped classes in the evaluation.\n        eval_configs (object): Evaluation configuration object.\n        eval_version (str, optional): Evaluation version.\n            Default: 'detection_cvpr_2019'\n\n    Returns:\n        list: List of standard NuScenesBoxes in the global\n            coordinate.\n    "
    box_list = []
    attr_list = []
    for (box, attr) in zip(boxes, attrs):
        box.rotate(pyquaternion.Quaternion(info['cam2ego_rotation']))
        box.translate(np.array(info['cam2ego_translation']))
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
        attr_list.append(attr)
    return (box_list, attr_list)

def global_nusc_box_to_cam(info, boxes, classes, eval_configs, eval_version='detection_cvpr_2019'):
    if False:
        for i in range(10):
            print('nop')
    "Convert the box from global to camera coordinate.\n\n    Args:\n        info (dict): Info for a specific sample data, including the\n            calibration information.\n        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.\n        classes (list[str]): Mapped classes in the evaluation.\n        eval_configs (object): Evaluation configuration object.\n        eval_version (str, optional): Evaluation version.\n            Default: 'detection_cvpr_2019'\n\n    Returns:\n        list: List of standard NuScenesBoxes in the global\n            coordinate.\n    "
    box_list = []
    for box in boxes:
        box.translate(-np.array(info['ego2global_translation']))
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']).inverse)
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        box.translate(-np.array(info['cam2ego_translation']))
        box.rotate(pyquaternion.Quaternion(info['cam2ego_rotation']).inverse)
        box_list.append(box)
    return box_list

def nusc_box_to_cam_box3d(boxes):
    if False:
        return 10
    'Convert boxes from :obj:`NuScenesBox` to :obj:`CameraInstance3DBoxes`.\n\n    Args:\n        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.\n\n    Returns:\n        tuple (:obj:`CameraInstance3DBoxes` | torch.Tensor | torch.Tensor):\n            Converted 3D bounding boxes, scores and labels.\n    '
    locs = torch.Tensor([b.center for b in boxes]).view(-1, 3)
    dims = torch.Tensor([b.wlh for b in boxes]).view(-1, 3)
    rots = torch.Tensor([b.orientation.yaw_pitch_roll[0] for b in boxes]).view(-1, 1)
    velocity = torch.Tensor([b.velocity[0::2] for b in boxes]).view(-1, 2)
    dims[:, [0, 1, 2]] = dims[:, [1, 2, 0]]
    rots = -rots
    boxes_3d = torch.cat([locs, dims, rots, velocity], dim=1).cuda()
    cam_boxes3d = CameraInstance3DBoxes(boxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
    scores = torch.Tensor([b.score for b in boxes]).cuda()
    labels = torch.LongTensor([b.label for b in boxes]).cuda()
    nms_scores = scores.new_zeros(scores.shape[0], 10 + 1)
    indices = labels.new_tensor(list(range(scores.shape[0])))
    nms_scores[indices, labels] = scores
    return (cam_boxes3d, nms_scores, labels)