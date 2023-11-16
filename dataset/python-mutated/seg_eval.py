import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

def fast_hist(preds, labels, num_classes):
    if False:
        return 10
    'Compute the confusion matrix for every batch.\n\n    Args:\n        preds (np.ndarray):  Prediction labels of points with shape of\n        (num_points, ).\n        labels (np.ndarray): Ground truth labels of points with shape of\n        (num_points, ).\n        num_classes (int): number of classes\n\n    Returns:\n        np.ndarray: Calculated confusion matrix.\n    '
    k = (labels >= 0) & (labels < num_classes)
    bin_count = np.bincount(num_classes * labels[k].astype(int) + preds[k], minlength=num_classes ** 2)
    return bin_count[:num_classes ** 2].reshape(num_classes, num_classes)

def per_class_iou(hist):
    if False:
        while True:
            i = 10
    'Compute the per class iou.\n\n    Args:\n        hist(np.ndarray):  Overall confusion martix\n        (num_classes, num_classes ).\n\n    Returns:\n        np.ndarray: Calculated per class iou\n    '
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def get_acc(hist):
    if False:
        for i in range(10):
            print('nop')
    'Compute the overall accuracy.\n\n    Args:\n        hist(np.ndarray):  Overall confusion martix\n        (num_classes, num_classes ).\n\n    Returns:\n        float: Calculated overall acc\n    '
    return np.diag(hist).sum() / hist.sum()

def get_acc_cls(hist):
    if False:
        for i in range(10):
            print('nop')
    'Compute the class average accuracy.\n\n    Args:\n        hist(np.ndarray):  Overall confusion martix\n        (num_classes, num_classes ).\n\n    Returns:\n        float: Calculated class average acc\n    '
    return np.nanmean(np.diag(hist) / hist.sum(axis=1))

def seg_eval(gt_labels, seg_preds, label2cat, ignore_index, logger=None):
    if False:
        return 10
    'Semantic Segmentation  Evaluation.\n\n    Evaluate the result of the Semantic Segmentation.\n\n    Args:\n        gt_labels (list[torch.Tensor]): Ground truth labels.\n        seg_preds  (list[torch.Tensor]): Predictions.\n        label2cat (dict): Map from label to category name.\n        ignore_index (int): Index that will be ignored in evaluation.\n        logger (logging.Logger | str, optional): The way to print the mAP\n            summary. See `mmdet.utils.print_log()` for details. Default: None.\n\n    Returns:\n        dict[str, float]: Dict of results.\n    '
    assert len(seg_preds) == len(gt_labels)
    num_classes = len(label2cat)
    hist_list = []
    for i in range(len(gt_labels)):
        gt_seg = gt_labels[i].clone().numpy().astype(np.int)
        pred_seg = seg_preds[i].clone().numpy().astype(np.int)
        pred_seg[gt_seg == ignore_index] = -1
        gt_seg[gt_seg == ignore_index] = -1
        hist_list.append(fast_hist(pred_seg, gt_seg, num_classes))
    iou = per_class_iou(sum(hist_list))
    miou = np.nanmean(iou)
    acc = get_acc(sum(hist_list))
    acc_cls = get_acc_cls(sum(hist_list))
    header = ['classes']
    for i in range(len(label2cat)):
        header.append(label2cat[i])
    header.extend(['miou', 'acc', 'acc_cls'])
    ret_dict = dict()
    table_columns = [['results']]
    for i in range(len(label2cat)):
        ret_dict[label2cat[i]] = float(iou[i])
        table_columns.append([f'{iou[i]:.4f}'])
    ret_dict['miou'] = float(miou)
    ret_dict['acc'] = float(acc)
    ret_dict['acc_cls'] = float(acc_cls)
    table_columns.append([f'{miou:.4f}'])
    table_columns.append([f'{acc:.4f}'])
    table_columns.append([f'{acc_cls:.4f}'])
    table_data = [header]
    table_rows = list(zip(*table_columns))
    table_data += table_rows
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print_log('\n' + table.table, logger=logger)
    return ret_dict