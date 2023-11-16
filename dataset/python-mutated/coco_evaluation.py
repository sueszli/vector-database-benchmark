import itertools
import json
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List, Union
import fire
import numpy as np
from terminaltables import AsciiTable

def _cocoeval_summarize(cocoeval, ap=1, iouThr=None, catIdx=None, areaRng='all', maxDets=100, catName='', nameStrLen=None):
    if False:
        i = 10
        return i + 15
    p = cocoeval.params
    if catName:
        iStr = ' {:<18} {} {:<{nameStrLen}} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        nameStr = catName
    else:
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '{:0.2f}'.format(iouThr)
    aind = [i for (i, aRng) in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for (i, mDet) in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        s = cocoeval.eval['precision']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        if catIdx is not None:
            s = s[:, :, catIdx, aind, mind]
        else:
            s = s[:, :, :, aind, mind]
    else:
        s = cocoeval.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        if catIdx is not None:
            s = s[:, catIdx, aind, mind]
        else:
            s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    if catName:
        print(iStr.format(titleStr, typeStr, nameStr, iouStr, areaRng, maxDets, mean_s, nameStrLen=nameStrLen))
    else:
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s

def evaluate_core(dataset_path, result_path, metric: str='bbox', classwise: bool=False, max_detections: int=500, iou_thrs=None, metric_items=None, out_dir: str=None, areas: List[int]=[1024, 9216, 10000000000], COCO=None, COCOeval=None):
    if False:
        print('Hello World!')
    "Evaluation in COCO protocol.\n    Args:\n        dataset_path (str): COCO dataset json path.\n        result_path (str): COCO result json path.\n        metric (str | list[str]): Metrics to be evaluated. Options are\n            'bbox', 'segm', 'proposal'.\n        classwise (bool): Whether to evaluating the AP for each class.\n        max_detections (int): Maximum number of detections to consider for AP\n            calculation.\n            Default: 500\n        iou_thrs (List[float], optional): IoU threshold used for\n            evaluating recalls/mAPs. If set to a list, the average of all\n            IoUs will also be computed. If not specified, [0.50, 0.55,\n            0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.\n            Default: None.\n        metric_items (list[str] | str, optional): Metric items that will\n            be returned. If not specified, ``['AR@10', 'AR@100',\n            'AR@500', 'AR_s@500', 'AR_m@500', 'AR_l@500' ]`` will be\n            used when ``metric=='proposal'``, ``['mAP', 'mAP50', 'mAP75',\n            'mAP_s', 'mAP_m', 'mAP_l', 'mAP50_s', 'mAP50_m', 'mAP50_l']``\n            will be used when ``metric=='bbox' or metric=='segm'``.\n        out_dir (str): Directory to save evaluation result json.\n        areas (List[int]): area regions for coco evaluation calculations\n    Returns:\n        dict:\n            eval_results (dict[str, float]): COCO style evaluation metric.\n            export_path (str): Path for the exported eval result json.\n\n    "
    metrics = metric if isinstance(metric, list) else [metric]
    allowed_metrics = ['bbox', 'segm']
    for metric in metrics:
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
    if iou_thrs is None:
        iou_thrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
    if metric_items is not None:
        if not isinstance(metric_items, list):
            metric_items = [metric_items]
    if areas is not None:
        if len(areas) != 3:
            raise ValueError('3 integers should be specified as areas, representing 3 area regions')
    eval_results = OrderedDict()
    cocoGt = COCO(dataset_path)
    cat_ids = list(cocoGt.cats.keys())
    for metric in metrics:
        msg = f'Evaluating {metric}...'
        msg = '\n' + msg
        print(msg)
        iou_type = metric
        with open(result_path) as json_file:
            results = json.load(json_file)
        try:
            cocoDt = cocoGt.loadRes(results)
        except IndexError:
            print('The testing results of the whole dataset is empty.')
            break
        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        if areas is not None:
            cocoEval.params.areaRng = [[0 ** 2, areas[2]], [0 ** 2, areas[0]], [areas[0], areas[1]], [areas[1], areas[2]]]
        cocoEval.params.catIds = cat_ids
        cocoEval.params.maxDets = [max_detections]
        cocoEval.params.iouThrs = [iou_thrs] if not isinstance(iou_thrs, list) and (not isinstance(iou_thrs, np.ndarray)) else iou_thrs
        coco_metric_names = {'mAP': 0, 'mAP75': 1, 'mAP50': 2, 'mAP_s': 3, 'mAP_m': 4, 'mAP_l': 5, 'mAP50_s': 6, 'mAP50_m': 7, 'mAP50_l': 8, 'AR_s': 9, 'AR_m': 10, 'AR_l': 11}
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(f'metric item {metric_item} is not supported')
        cocoEval.evaluate()
        cocoEval.accumulate()
        mAP = _cocoeval_summarize(cocoEval, ap=1, iouThr=None, areaRng='all', maxDets=max_detections)
        mAP50 = _cocoeval_summarize(cocoEval, ap=1, iouThr=0.5, areaRng='all', maxDets=max_detections)
        mAP75 = _cocoeval_summarize(cocoEval, ap=1, iouThr=0.75, areaRng='all', maxDets=max_detections)
        mAP50_s = _cocoeval_summarize(cocoEval, ap=1, iouThr=0.5, areaRng='small', maxDets=max_detections)
        mAP50_m = _cocoeval_summarize(cocoEval, ap=1, iouThr=0.5, areaRng='medium', maxDets=max_detections)
        mAP50_l = _cocoeval_summarize(cocoEval, ap=1, iouThr=0.5, areaRng='large', maxDets=max_detections)
        mAP_s = _cocoeval_summarize(cocoEval, ap=1, iouThr=None, areaRng='small', maxDets=max_detections)
        mAP_m = _cocoeval_summarize(cocoEval, ap=1, iouThr=None, areaRng='medium', maxDets=max_detections)
        mAP_l = _cocoeval_summarize(cocoEval, ap=1, iouThr=None, areaRng='large', maxDets=max_detections)
        AR_s = _cocoeval_summarize(cocoEval, ap=0, iouThr=None, areaRng='small', maxDets=max_detections)
        AR_m = _cocoeval_summarize(cocoEval, ap=0, iouThr=None, areaRng='medium', maxDets=max_detections)
        AR_l = _cocoeval_summarize(cocoEval, ap=0, iouThr=None, areaRng='large', maxDets=max_detections)
        cocoEval.stats = np.append([mAP, mAP75, mAP50, mAP_s, mAP_m, mAP_l, mAP50_s, mAP50_m, mAP50_l, AR_s, AR_m, AR_l], 0)
        if classwise:
            precisions = cocoEval.eval['precision']
            if len(cat_ids) != precisions.shape[2]:
                raise ValueError(f'The number of categories {len(cat_ids)} is not equal to the number of precisions {precisions.shape[2]}')
            max_cat_name_len = 0
            for (idx, catId) in enumerate(cat_ids):
                nm = cocoGt.loadCats(catId)[0]
                cat_name_len = len(nm['name'])
                max_cat_name_len = cat_name_len if cat_name_len > max_cat_name_len else max_cat_name_len
            results_per_category = []
            for (idx, catId) in enumerate(cat_ids):
                image_ids = cocoGt.getImgIds(catIds=[catId])
                if len(image_ids) == 0:
                    continue
                nm = cocoGt.loadCats(catId)[0]
                ap = _cocoeval_summarize(cocoEval, ap=1, catIdx=idx, areaRng='all', maxDets=max_detections, catName=nm['name'], nameStrLen=max_cat_name_len)
                ap_s = _cocoeval_summarize(cocoEval, ap=1, catIdx=idx, areaRng='small', maxDets=max_detections, catName=nm['name'], nameStrLen=max_cat_name_len)
                ap_m = _cocoeval_summarize(cocoEval, ap=1, catIdx=idx, areaRng='medium', maxDets=max_detections, catName=nm['name'], nameStrLen=max_cat_name_len)
                ap_l = _cocoeval_summarize(cocoEval, ap=1, catIdx=idx, areaRng='large', maxDets=max_detections, catName=nm['name'], nameStrLen=max_cat_name_len)
                ap50 = _cocoeval_summarize(cocoEval, ap=1, iouThr=0.5, catIdx=idx, areaRng='all', maxDets=max_detections, catName=nm['name'], nameStrLen=max_cat_name_len)
                ap50_s = _cocoeval_summarize(cocoEval, ap=1, iouThr=0.5, catIdx=idx, areaRng='small', maxDets=max_detections, catName=nm['name'], nameStrLen=max_cat_name_len)
                ap50_m = _cocoeval_summarize(cocoEval, ap=1, iouThr=0.5, catIdx=idx, areaRng='medium', maxDets=max_detections, catName=nm['name'], nameStrLen=max_cat_name_len)
                ap50_l = _cocoeval_summarize(cocoEval, ap=1, iouThr=0.5, catIdx=idx, areaRng='large', maxDets=max_detections, catName=nm['name'], nameStrLen=max_cat_name_len)
                results_per_category.append((f"{metric}_{nm['name']}_mAP", f'{float(ap):0.3f}'))
                results_per_category.append((f"{metric}_{nm['name']}_mAP_s", f'{float(ap_s):0.3f}'))
                results_per_category.append((f"{metric}_{nm['name']}_mAP_m", f'{float(ap_m):0.3f}'))
                results_per_category.append((f"{metric}_{nm['name']}_mAP_l", f'{float(ap_l):0.3f}'))
                results_per_category.append((f"{metric}_{nm['name']}_mAP50", f'{float(ap50):0.3f}'))
                results_per_category.append((f"{metric}_{nm['name']}_mAP50_s", f'{float(ap50_s):0.3f}'))
                results_per_category.append((f"{metric}_{nm['name']}_mAP50_m", f'{float(ap50_m):0.3f}'))
                results_per_category.append((f"{metric}_{nm['name']}_mAP50_l", f'{float(ap50_l):0.3f}'))
            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (num_columns // 2)
            results_2d = itertools.zip_longest(*[results_flatten[i::num_columns] for i in range(num_columns)])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            print('\n' + table.table)
        if metric_items is None:
            metric_items = ['mAP', 'mAP50', 'mAP75', 'mAP_s', 'mAP_m', 'mAP_l', 'mAP50_s', 'mAP50_m', 'mAP50_l']
        for metric_item in metric_items:
            key = f'{metric}_{metric_item}'
            val = float(f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}')
            eval_results[key] = val
        ap = cocoEval.stats
        eval_results[f'{metric}_mAP_copypaste'] = f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f} {ap[6]:.3f} {ap[7]:.3f} {ap[8]:.3f}'
        if classwise:
            eval_results['results_per_category'] = {key: value for (key, value) in results_per_category}
    if not out_dir:
        out_dir = Path(result_path).parent
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    export_path = str(Path(out_dir) / 'eval.json')
    with open(export_path, 'w', encoding='utf-8') as outfile:
        json.dump(eval_results, outfile, indent=4, separators=(',', ':'))
    print(f'COCO evaluation results are successfully exported to {export_path}')
    return {'eval_results': eval_results, 'export_path': export_path}

def evaluate(dataset_json_path: str, result_json_path: str, out_dir: str=None, type: str='bbox', classwise: bool=False, max_detections: int=500, iou_thrs: Union[List[float], float]=None, areas: List[int]=[1024, 9216, 10000000000], return_dict: bool=False):
    if False:
        return 10
    "\n    Args:\n        dataset_json_path (str): file path for the coco dataset json file\n        result_json_path (str): file path for the coco result json file\n        out_dir (str): dir to save eval result\n        type (bool): 'bbox' or 'segm'\n        classwise (bool): whether to evaluate the AP for each class\n        max_detections (int): Maximum number of detections to consider for AP alculation. Default: 500\n        iou_thrs (float): IoU threshold used for evaluating recalls/mAPs\n        areas (List[int]): area regions for coco evaluation calculations\n        return_dict (bool): If True, returns a dict with 'eval_results' 'export_path' fields.\n    "
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ModuleNotFoundError:
        raise ModuleNotFoundError('Please run "pip install -U pycocotools" to install pycocotools first for coco evaluation.')
    result = evaluate_core(dataset_json_path, result_json_path, type, classwise, max_detections, iou_thrs, out_dir=out_dir, areas=areas, COCO=COCO, COCOeval=COCOeval)
    if return_dict:
        return result
if __name__ == '__main__':
    fire.Fire(evaluate)