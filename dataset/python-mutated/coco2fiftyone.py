import time
from pathlib import Path
from typing import List
import fire
from sahi.utils.file import load_json

def main(image_dir: str, dataset_json_path: str, *result_json_paths, iou_thresh: float=0.5):
    if False:
        print('Hello World!')
    '\n    Args:\n        image_dir (str): directory for coco images\n        dataset_json_path (str): file path for the coco dataset json file\n        result_json_paths (str): one or more paths for the coco result json file\n        iou_thresh (float): iou threshold for coco evaluation\n    '
    from fiftyone.utils.coco import add_coco_labels
    from sahi.utils.fiftyone import create_fiftyone_dataset_from_coco_file, fo
    coco_result_list = []
    result_name_list = []
    if result_json_paths:
        for result_json_path in result_json_paths:
            coco_result = load_json(result_json_path)
            coco_result_list.append(coco_result)
            result_name_temp = Path(result_json_path).stem
            result_name = result_name_temp
            name_increment = 2
            while result_name in result_name_list:
                result_name = result_name_temp + '_' + str(name_increment)
                name_increment += 1
            result_name_list.append(result_name)
    dataset = create_fiftyone_dataset_from_coco_file(image_dir, dataset_json_path)
    if result_json_paths:
        for (result_name, coco_result) in zip(result_name_list, coco_result_list):
            add_coco_labels(dataset, result_name, coco_result, coco_id_field='gt_coco_id')
    session = fo.launch_app()
    session.dataset = dataset
    if result_json_paths:
        first_coco_result_name = result_name_list[0]
        _ = dataset.evaluate_detections(first_coco_result_name, gt_field='gt_detections', eval_key=f'{first_coco_result_name}_eval', iou=iou_thresh, compute_mAP=False)
        eval_view = dataset.load_evaluation_view(f'{first_coco_result_name}_eval')
        session.view = eval_view.sort_by(f'{first_coco_result_name}_eval_fp', reverse=True)
        print(f'SAHI has successfully launched a Fiftyone app at http://localhost:{fo.config.default_app_port}')
    while 1:
        time.sleep(3)
if __name__ == '__main__':
    fire.Fire(main)