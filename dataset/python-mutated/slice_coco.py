import os
import fire
from sahi.slicing import slice_coco
from sahi.utils.file import Path, save_json

def slice(image_dir: str, dataset_json_path: str, slice_size: int=512, overlap_ratio: float=0.2, ignore_negative_samples: bool=False, output_dir: str='runs/slice_coco', min_area_ratio: float=0.1):
    if False:
        while True:
            i = 10
    '\n    Args:\n        image_dir (str): directory for coco images\n        dataset_json_path (str): file path for the coco dataset json file\n        slice_size (int)\n        overlap_ratio (float): slice overlap ratio\n        ignore_negative_samples (bool): ignore images without annotation\n        output_dir (str): output export dir\n        min_area_ratio (float): If the cropped annotation area to original\n            annotation ratio is smaller than this value, the annotation\n            is filtered out. Default 0.1.\n    '
    slice_size_list = slice_size
    if isinstance(slice_size_list, (int, float)):
        slice_size_list = [slice_size_list]
    print('Slicing step is starting...')
    for slice_size in slice_size_list:
        output_images_folder_name = Path(dataset_json_path).stem + f"_images_{str(slice_size)}_{str(overlap_ratio).replace('.', '')}"
        output_images_dir = str(Path(output_dir) / output_images_folder_name)
        sliced_coco_name = Path(dataset_json_path).name.replace('.json', f"_{str(slice_size)}_{str(overlap_ratio).replace('.', '')}")
        (coco_dict, coco_path) = slice_coco(coco_annotation_file_path=dataset_json_path, image_dir=image_dir, output_coco_annotation_file_name='', output_dir=output_images_dir, ignore_negative_samples=ignore_negative_samples, slice_height=slice_size, slice_width=slice_size, min_area_ratio=min_area_ratio, overlap_height_ratio=overlap_ratio, overlap_width_ratio=overlap_ratio, out_ext='.jpg', verbose=False)
        output_coco_annotation_file_path = os.path.join(output_dir, sliced_coco_name + '.json')
        save_json(coco_dict, output_coco_annotation_file_path)
        print(f"Sliced dataset for 'slice_size: {slice_size}' is exported to {output_dir}")
if __name__ == '__main__':
    fire.Fire(slice)