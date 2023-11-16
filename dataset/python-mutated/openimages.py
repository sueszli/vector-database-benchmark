"""
Open Images-style detection evaluation.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import defaultdict
from copy import deepcopy
import numpy as np
import fiftyone.core.plots as fop
import fiftyone.utils.iou as foui
from .detection import DetectionEvaluation, DetectionEvaluationConfig, DetectionResults

class OpenImagesEvaluationConfig(DetectionEvaluationConfig):
    """Open Images-style evaluation config.

    Args:
        pred_field: the name of the field containing the predicted
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polylines`, or
            :class:`fiftyone.core.labels.Keypoints`
        gt_field: the name of the field containing the ground truth
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polylines`, or
            :class:`fiftyone.core.labels.Keypoints`
        iou (None): the IoU threshold to use to determine matches
        classwise (None): whether to only match objects with the same class
            label (True) or allow matches between classes (False)
        iscrowd ("IsGroupOf"): the name of the crowd attribute
        use_masks (False): whether to compute IoUs using the instances masks in
            the ``mask`` attribute of the provided objects, which must be
            :class:`fiftyone.core.labels.Detection` instances
        use_boxes (False): whether to compute IoUs using the bounding boxes
            of the provided :class:`fiftyone.core.labels.Polyline` instances
            rather than using their actual geometries
        tolerance (None): a tolerance, in pixels, when generating approximate
            polylines for instance masks. Typical values are 1-3 pixels
        max_preds (None): the maximum number of predicted objects to evaluate
            when computing mAP and PR curves
        error_level (1): the error level to use when manipulating instance
            masks or polylines. Valid values are:

            -   0: raise geometric errors that are encountered
            -   1: log warnings if geometric errors are encountered
            -   2: ignore geometric errors

            If ``error_level > 0``, any calculation that raises a geometric
            error will default to an IoU of 0
        hierarchy (None): an optional dict containing a hierachy of classes for
            evaluation following the structure
            ``{"LabelName": label, "Subcategory": [{...}, ...]}``
        pos_label_field (None): the name of a field containing image-level
            :class:`fiftyone.core.labels.Classifications` that specify which
            classes should be evaluated in the image
        neg_label_field (None): the name of a field containing image-level
            :class:`fiftyone.core.labels.Classifications` that specify which
            classes should not be evaluated in the image
        expand_gt_hierarchy (True): whether to expand ground truth objects and
            labels according to the provided ``hierarchy``
        expand_pred_hierarchy (False): whether to expand predicted objects and
            labels according to the provided ``hierarchy``
    """

    def __init__(self, pred_field, gt_field, iou=None, classwise=None, iscrowd='IsGroupOf', use_masks=False, use_boxes=False, tolerance=None, max_preds=None, error_level=1, hierarchy=None, pos_label_field=None, neg_label_field=None, expand_gt_hierarchy=True, expand_pred_hierarchy=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(pred_field, gt_field, iou=iou, classwise=classwise, **kwargs)
        self.iscrowd = iscrowd
        self.use_masks = use_masks
        self.use_boxes = use_boxes
        self.tolerance = tolerance
        self.max_preds = max_preds
        self.error_level = error_level
        self.hierarchy = hierarchy
        self.pos_label_field = pos_label_field
        self.neg_label_field = neg_label_field
        self.expand_gt_hierarchy = expand_gt_hierarchy
        self.expand_pred_hierarchy = expand_pred_hierarchy
        if expand_pred_hierarchy:
            if not hierarchy:
                self.expand_pred_hierarchy = False
            self.expand_gt_hierarchy = self.expand_pred_hierarchy
        if expand_gt_hierarchy and (not hierarchy):
            self.expand_gt_hierarchy = False
        if self.expand_gt_hierarchy or self.expand_pred_hierarchy:
            (self._hierarchy_keyed_parent, self._hierarchy_keyed_child, _) = _build_plain_hierarchy(self.hierarchy, skip_root=True)

    @property
    def method(self):
        if False:
            while True:
                i = 10
        return 'open-images'

    @property
    def requires_additional_fields(self):
        if False:
            for i in range(10):
                print('nop')
        return True

class OpenImagesEvaluation(DetectionEvaluation):
    """Open Images-style evaluation.

    Args:
        config: a :class:`OpenImagesEvaluationConfig`
    """

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__(config)
        if config.iou is None:
            raise ValueError('You must specify an `iou` threshold in order to run Open Images evaluation')
        if config.classwise is None:
            raise ValueError('You must specify a `classwise` value in order to run Open Images evaluation')

    def evaluate(self, sample_or_frame, eval_key=None):
        if False:
            return 10
        'Performs Open Images-style evaluation on the given image.\n\n        Predicted objects are matched to ground truth objects in descending\n        order of confidence, with matches requiring a minimum IoU of\n        ``self.config.iou``.\n\n        The ``self.config.classwise`` parameter controls whether to only match\n        objects with the same class label (True) or allow matches between\n        classes (False).\n\n        If a ground truth object has its ``self.config.iscrowd`` attribute set,\n        then the object can have multiple true positive predictions matched to\n        it.\n\n        Args:\n            sample_or_frame: a :class:`fiftyone.core.sample.Sample` or\n                :class:`fiftyone.core.frame.Frame`\n            eval_key (None): the evaluation key for this evaluation\n\n        Returns:\n            a list of matched\n            ``(gt_label, pred_label, iou, pred_confidence, gt_id, pred_id)``\n            tuples\n        '
        gts = sample_or_frame[self.gt_field]
        preds = sample_or_frame[self.pred_field]
        pos_labs = None
        if self.config.pos_label_field:
            pos_labs = sample_or_frame[self.config.pos_label_field]
            if pos_labs is None:
                pos_labs = []
            else:
                pos_labs = [c.label for c in pos_labs.classifications]
                if self.config.expand_gt_hierarchy:
                    pos_labs = _expand_label_hierarchy(pos_labs, self.config)
        neg_labs = None
        if self.config.neg_label_field:
            neg_labs = sample_or_frame[self.config.neg_label_field]
            if neg_labs is None:
                neg_labs = []
            else:
                neg_labs = [c.label for c in neg_labs.classifications]
                if self.config.expand_gt_hierarchy:
                    neg_labs = _expand_label_hierarchy(neg_labs, self.config, expand_child=False)
        if eval_key is None:
            eval_key = 'eval'
            gts = _copy_labels(gts)
            preds = _copy_labels(preds)
        return _open_images_evaluation_single_iou(gts, preds, eval_key, self.config, pos_labs, neg_labs)

    def generate_results(self, samples, matches, eval_key=None, classes=None, missing=None):
        if False:
            i = 10
            return i + 15
        'Generates aggregate evaluation results for the samples.\n\n        This method generates precision and recall curves for the configured\n        IoU at ``self.config.iou``.\n\n        Args:\n            samples: a :class:`fiftyone.core.collections.SampleCollection`\n            matches: a list of\n                ``(gt_label, pred_label, iou, pred_confidence, gt_id, pred_id)``\n                matches. Either label can be ``None`` to indicate an unmatched\n                object\n            eval_key (None): the evaluation key for this evaluation\n            classes (None): the list of possible classes. If not provided, the\n                observed ground truth/predicted labels are used for results\n                purposes\n            missing (None): a missing label string. Any unmatched objects are\n                given this label for results purposes\n\n        Returns:\n            a :class:`OpenImagesDetectionResults`\n        '
        (precision, recall, thresholds, classes) = _compute_pr_curves(matches, classes=classes)
        return OpenImagesDetectionResults(samples, self.config, eval_key, matches, precision, recall, classes, thresholds=thresholds, missing=missing, backend=self)

class OpenImagesDetectionResults(DetectionResults):
    """Class that stores the results of an Open Images detection evaluation.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`OpenImagesEvaluationConfig` used
        eval_key: the evaluation key
        matches: a list of
            ``(gt_label, pred_label, iou, pred_confidence, gt_id, pred_id)``
            matches. Either label can be ``None`` to indicate an unmatched
            object
        precision: a dict of per-class precision values
        recall: a dict of per-class recall values
        classes: the list of possible classes
        thresholds (None): an optional dict of per-class decision thresholds
        missing (None): a missing label string. Any unmatched objects are
            given this label for evaluation purposes
        backend (None): a :class:`OpenImagesEvaluation` backend
    """

    def __init__(self, samples, config, eval_key, matches, precision, recall, classes, thresholds=None, missing=None, backend=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(samples, config, eval_key, matches, classes=classes, missing=missing, backend=backend)
        self.precision = precision
        self.recall = recall
        self.thresholds = thresholds
        self._classwise_AP = {}
        for c in classes:
            if c in precision and c in recall:
                ap = _compute_AP(precision[c], recall[c])
            else:
                ap = -1
            self._classwise_AP[c] = ap

    def plot_pr_curves(self, classes=None, num_points=101, backend='plotly', **kwargs):
        if False:
            while True:
                i = 10
        'Plots precision-recall (PR) curves for the detection results.\n\n        Args:\n            classes (None): a list of classes to generate curves for. By\n                default, the top 3 AP classes will be plotted\n            num_points (101): the number of linearly spaced recall values to\n                plot\n            backend ("plotly"): the plotting backend to use. Supported values\n                are ``("plotly", "matplotlib")``\n            **kwargs: keyword arguments for the backend plotting method:\n\n                -   "plotly" backend: :meth:`fiftyone.core.plots.plotly.plot_pr_curves`\n                -   "matplotlib" backend: :meth:`fiftyone.core.plots.matplotlib.plot_pr_curves`\n\n        Returns:\n            one of the following:\n\n            -   a :class:`fiftyone.core.plots.plotly.PlotlyNotebookPlot`, if\n                you are working in a notebook context and the plotly backend is\n                used\n            -   a plotly or matplotlib figure, otherwise\n        '
        if classes is not None:
            self._validate_classes(classes)
        else:
            classwise_AP = [(ap, c) for (c, ap) in self._classwise_AP.items()]
            classes = [c for (ap, c) in sorted(classwise_AP)[-3:]]
        precisions = []
        recall = None
        has_thresholds = self.thresholds is not None
        thresholds = [] if has_thresholds else None
        for c in classes:
            p = self.precision.get(c, None)
            r = self.recall.get(c, None)
            if has_thresholds:
                t = self.thresholds.get(c, None)
            else:
                t = None
            (pre, rec, thr) = _interpolate_pr(p, r, thresholds=t, num_points=num_points)
            precisions.append(pre)
            if recall is None:
                recall = rec
            if has_thresholds:
                thresholds.append(thr)
        return fop.plot_pr_curves(precisions, recall, classes, thresholds=thresholds, backend=backend, **kwargs)

    def mAP(self, classes=None):
        if False:
            while True:
                i = 10
        'Computes Open Images-style mean average precision (mAP) for the\n        specified classes.\n\n        See `this page <https://storage.googleapis.com/openimages/web/evaluation.html>`_\n        for more details about Open Images-style mAP.\n\n        Args:\n            classes (None): a list of classes for which to compute mAP\n\n        Returns:\n            the mAP in ``[0, 1]``\n        '
        if classes is not None:
            self._validate_classes(classes)
            classwise_AP = [self._classwise_AP[c] for c in classes]
        else:
            classwise_AP = list(self._classwise_AP.values())
        classwise_AP = np.array(classwise_AP)
        classwise_AP = classwise_AP[classwise_AP > -1]
        if classwise_AP.size == 0:
            return -1
        return np.mean(classwise_AP)

    def _validate_classes(self, classes):
        if False:
            while True:
                i = 10
        missing_classes = set(classes) - set(self.classes)
        if missing_classes:
            raise ValueError('Classes %s not found' % missing_classes)

    @classmethod
    def _from_dict(cls, d, samples, config, eval_key, **kwargs):
        if False:
            return 10
        precision = d['precision']
        recall = d['recall']
        thresholds = d.get('thresholds', None)
        return super()._from_dict(d, samples, config, eval_key, precision=precision, recall=recall, thresholds=thresholds, **kwargs)
_NO_MATCH_ID = ''
_NO_MATCH_IOU = None

def _expand_label_hierarchy(labels, config, expand_child=True):
    if False:
        i = 10
        return i + 15
    keyed_nodes = config._hierarchy_keyed_parent
    if expand_child:
        keyed_nodes = config._hierarchy_keyed_child
    additional_labs = []
    for lab in labels:
        if lab in keyed_nodes:
            additional_labs += list(keyed_nodes[lab])
    return list(set(labels + additional_labs))

def _expand_detection_hierarchy(cats, obj, config, label_type):
    if False:
        for i in range(10):
            print('nop')
    keyed_children = config._hierarchy_keyed_child
    for parent in keyed_children[obj.label]:
        new_obj = obj.copy()
        new_obj.id = obj.id
        new_obj.label = parent
        cats[parent][label_type].append(new_obj)

def _open_images_evaluation_single_iou(gts, preds, eval_key, config, pos_labs, neg_labs):
    if False:
        for i in range(10):
            print('nop')
    iou_thresh = min(config.iou, 1 - 1e-10)
    id_key = '%s_id' % eval_key
    iou_key = '%s_iou' % eval_key
    (cats, pred_ious, iscrowd) = _open_images_evaluation_setup(gts, preds, id_key, iou_key, config, pos_labs, neg_labs, max_preds=config.max_preds)
    matches = _compute_matches(cats, pred_ious, iou_thresh, iscrowd, eval_key=eval_key, id_key=id_key, iou_key=iou_key)
    return matches

def _open_images_evaluation_setup(gts, preds, id_key, iou_key, config, pos_labs, neg_labs, max_preds=None):
    if False:
        for i in range(10):
            print('nop')
    if pos_labs is None:
        relevant_labs = neg_labs
    elif neg_labs is None:
        relevant_labs = pos_labs
    else:
        relevant_labs = list(set(pos_labs + neg_labs))
    iscrowd = lambda l: bool(l.get_attribute_value(config.iscrowd, False))
    classwise = config.classwise
    iou_kwargs = dict(iscrowd=iscrowd, error_level=config.error_level)
    if config.use_masks:
        iou_kwargs.update(use_masks=True, tolerance=config.tolerance)
    if config.use_boxes:
        iou_kwargs.update(use_boxes=True)
    cats = defaultdict(lambda : defaultdict(list))
    if gts is not None:
        for obj in gts[gts._LABEL_LIST_FIELD]:
            if relevant_labs is None or obj.label in relevant_labs:
                obj[iou_key] = _NO_MATCH_IOU
                obj[id_key] = _NO_MATCH_ID
                label = obj.label if classwise else 'all'
                cats[label]['gts'].append(obj)
                if config.expand_gt_hierarchy and label != 'all':
                    _expand_detection_hierarchy(cats, obj, config, 'gts')
    if preds is not None:
        for obj in preds[preds._LABEL_LIST_FIELD]:
            if relevant_labs is None or obj.label in relevant_labs:
                obj[iou_key] = _NO_MATCH_IOU
                obj[id_key] = _NO_MATCH_ID
                label = obj.label if classwise else 'all'
                cats[label]['preds'].append(obj)
                if config.expand_pred_hierarchy and label != 'all':
                    _expand_detection_hierarchy(cats, obj, config, 'preds')
    pred_ious = {}
    for objects in cats.values():
        gts = objects['gts']
        preds = objects['preds']
        preds = sorted(preds, key=lambda p: p.confidence or -1, reverse=True)
        if max_preds is not None:
            preds = preds[:max_preds]
        objects['preds'] = preds
        gts = sorted(gts, key=iscrowd)
        ious = foui.compute_ious(preds, gts, **iou_kwargs)
        gt_ids = [g.id for g in gts]
        for (pred, gt_ious) in zip(preds, ious):
            pred_ious[pred.id] = list(zip(gt_ids, gt_ious))
    return (cats, pred_ious, iscrowd)

def _compute_matches(cats, pred_ious, iou_thresh, iscrowd, eval_key, id_key, iou_key):
    if False:
        i = 10
        return i + 15
    matches = []
    p_round = 10 ** 10
    for (cat, objects) in cats.items():
        gt_map = {gt.id: gt for gt in objects['gts']}
        for pred in objects['preds']:
            if pred.id in pred_ious:
                best_match = None
                best_match_iou = iou_thresh
                highest_already_matched_iou = iou_thresh
                for (gt_id, iou) in pred_ious[pred.id]:
                    iou = int(iou * p_round + 0.5) / p_round
                    gt = gt_map[gt_id]
                    gt_iscrowd = iscrowd(gt)
                    if gt[id_key] != _NO_MATCH_ID and (not gt_iscrowd):
                        if iou > highest_already_matched_iou:
                            highest_already_matched_iou = iou
                            if iou > best_match_iou:
                                best_match = None
                                best_match_iou = iou_thresh
                        continue
                    if gt_iscrowd and gt.label != pred.label:
                        continue
                    if best_match and (not iscrowd(gt_map[best_match])) and gt_iscrowd:
                        break
                    if best_match_iou == 1:
                        break
                    if iou < best_match_iou:
                        continue
                    best_match_iou = iou
                    best_match = gt_id
                if highest_already_matched_iou > best_match_iou:
                    if best_match is not None and (not iscrowd(gt_map[best_match])):
                        best_match = None
                if best_match:
                    gt = gt_map[best_match]
                    if gt[id_key] == _NO_MATCH_ID:
                        record_match = True
                        gt[eval_key] = 'tp' if gt.label == pred.label else 'fn'
                        gt[id_key] = pred.id
                        gt[iou_key] = best_match_iou
                    else:
                        record_match = False
                    pred[eval_key] = 'tp' if gt.label == pred.label else 'fp'
                    pred[id_key] = best_match
                    pred[iou_key] = best_match_iou
                    if record_match:
                        matches.append((gt.label, pred.label, best_match_iou, pred.confidence, gt.id, pred.id))
                else:
                    pred[eval_key] = 'fp'
                    matches.append((None, pred.label, None, pred.confidence, None, pred.id))
            elif pred.label == cat:
                pred[eval_key] = 'fp'
                matches.append((None, pred.label, None, pred.confidence, None, pred.id))
        for gt in objects['gts']:
            if gt[id_key] == _NO_MATCH_ID:
                gt[eval_key] = 'fn'
                matches.append((gt.label, None, None, None, gt.id, None))
    return matches

def _compute_pr_curves(matches, classes=None):
    if False:
        for i in range(10):
            print('nop')
    if classes is None:
        _classes = set()
    class_matches = {}
    counted_gts = []
    for m in matches:
        if classes is None:
            _classes.add(m[0])
            _classes.add(m[1])
        c = m[0] if m[0] is not None else m[1]
        if c not in class_matches:
            class_matches[c] = {'tp': [], 'fp': [], 'num_gt': 0}
        if m[0] == m[1]:
            class_matches[c]['tp'].append(m)
        elif m[1]:
            class_matches[c]['fp'].append(m)
        if m[0] and m[4] not in counted_gts:
            class_matches[c]['num_gt'] += 1
            counted_gts.append(m[4])
    if classes is None:
        _classes.discard(None)
        classes = sorted(_classes)
    precision = {}
    recall = {}
    thresholds = {}
    for c in class_matches.keys():
        tp = class_matches[c]['tp']
        fp = class_matches[c]['fp']
        num_gt = class_matches[c]['num_gt']
        if num_gt == 0:
            continue
        tp_fp = np.array([1] * len(tp) + [0] * len(fp))
        confs = np.array([p[3] for p in tp] + [p[3] for p in fp])
        inds = np.argsort(-confs)
        tp_fp = tp_fp[inds]
        confs = confs[inds]
        tp_sum = np.cumsum(tp_fp).astype(dtype=float)
        total = np.arange(1, len(tp_fp) + 1).astype(dtype=float)
        pre = tp_sum / total
        rec = tp_sum / num_gt
        pre0 = pre[0] if pre.size > 0 else 1
        conf0 = max(1, confs[0]) if confs.size > 0 else 1
        pre = np.concatenate([[pre0], pre, [0]])
        confs = np.concatenate([[conf0], confs, [0]])
        rec = np.concatenate([[0], rec, [1]])
        for i in range(len(pre) - 1, 0, -1):
            if pre[i] > pre[i - 1]:
                pre[i - 1] = pre[i]
        precision[c] = pre
        recall[c] = rec
        thresholds[c] = confs
    return (precision, recall, thresholds, classes)

def _compute_AP(precision, recall):
    if False:
        i = 10
        return i + 15
    recall = np.asarray(recall)
    precision = np.asarray(precision)
    inds = np.where(recall[1:] != recall[:-1])[0] + 1
    return np.sum((recall[inds] - recall[inds - 1]) * precision[inds])

def _interpolate_pr(precision, recall, thresholds=None, num_points=101):
    if False:
        print('Hello World!')
    has_thresholds = thresholds is not None
    pre = np.zeros(num_points)
    thr = np.zeros(num_points) if has_thresholds else None
    rec = np.linspace(0, 1, num_points)
    if precision is None or recall is None:
        return (pre, rec, thr)
    precision = precision.copy()
    for i in range(len(precision) - 1, 0, -1):
        if precision[i] > precision[i - 1]:
            precision[i - 1] = precision[i]
    inds = np.searchsorted(recall, rec, side='left')
    try:
        for (ri, pi) in enumerate(inds):
            pre[ri] = precision[pi]
            if has_thresholds:
                thr[ri] = thresholds[pi]
    except:
        pass
    return (pre, rec, thr)

def _copy_labels(labels):
    if False:
        i = 10
        return i + 15
    if labels is None:
        return None
    field = labels._LABEL_LIST_FIELD
    _labels = labels.copy()
    for (_label, label) in zip(_labels[field], labels[field]):
        _label.id = label.id
    return _labels

def _update_dict(initial_dict, update):
    if False:
        while True:
            i = 10
    'Updates dictionary with update content\n\n    Args:\n        initial_dict: initial dictionary\n        update: updated dictionary\n    '
    for (key, value_list) in update.items():
        if key in initial_dict:
            initial_dict[key].update(value_list)
        else:
            initial_dict[key] = set(value_list)

def _build_plain_hierarchy(hierarchy, skip_root=False):
    if False:
        i = 10
        return i + 15
    'Expands tree hierarchy representation to parent-child dictionary.\n\n    Args:\n        hierarchy: labels hierarchy\n        skip_root (False): if true skips root from the processing (done for the case when all\n            classes under hierarchy are collected under virtual node)\n\n    Returns:\n        a tuple of:\n\n        -   ``keyed_parent``: dictionary of parent - all its children nodes\n        -   ``keyed_child``: dictionary of children - all its parent node\n        -   ``children``: all children of the current node\n    '
    all_children = set([])
    all_keyed_parent = {}
    all_keyed_child = {}
    if 'Subcategory' in hierarchy:
        for node in hierarchy['Subcategory']:
            (keyed_parent, keyed_child, children) = _build_plain_hierarchy(node)
            _update_dict(all_keyed_parent, keyed_parent)
            _update_dict(all_keyed_child, keyed_child)
            all_children.update(children)
    if not skip_root:
        all_keyed_parent[hierarchy['LabelName']] = deepcopy(all_children)
        all_children.add(hierarchy['LabelName'])
        for (child, _) in all_keyed_child.items():
            all_keyed_child[child].add(hierarchy['LabelName'])
        all_keyed_child[hierarchy['LabelName']] = set([])
    return (all_keyed_parent, all_keyed_child, all_children)