import fire
from sahi.utils.coco import Coco
from sahi.utils.file import Path, increment_path

def main(image_dir: str, dataset_json_path: str, train_split: str=0.9, project: str='runs/coco2yolov5', name: str='exp', seed: str=1, disable_symlink=False):
    if False:
        return 10
    '\n    Args:\n        images_dir (str): directory for coco images\n        dataset_json_path (str): file path for the coco json file to be converted\n        train_split (str): set the training split ratio\n        project (str): save results to project/name\n        name (str): save results to project/name"\n        seed (int): fix the seed for reproducibility\n        disable_symlink (bool): required in google colab env\n    '
    save_dir = Path(increment_path(Path(project) / name, exist_ok=False))
    coco = Coco.from_coco_dict_or_path(coco_dict_or_path=dataset_json_path, image_dir=image_dir)
    coco.export_as_yolov5(output_dir=str(save_dir), train_split_rate=train_split, numpy_seed=seed, disable_symlink=disable_symlink)
    print(f'COCO to YOLOv5 conversion results are successfully exported to {save_dir}')
if __name__ == '__main__':
    fire.Fire(main)