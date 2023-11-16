import os
import mmcv
import numpy as np
from tools.data_converter.s3dis_data_utils import S3DISData, S3DISSegData
from tools.data_converter.scannet_data_utils import ScanNetData, ScanNetSegData
from tools.data_converter.sunrgbd_data_utils import SUNRGBDData

def create_indoor_info_file(data_path, pkl_prefix='sunrgbd', save_path=None, workers=4, **kwargs):
    if False:
        while True:
            i = 10
    "Create indoor information file.\n\n    Get information of the raw data and save it to the pkl file.\n\n    Args:\n        data_path (str): Path of the data.\n        pkl_prefix (str, optional): Prefix of the pkl to be saved.\n            Default: 'sunrgbd'.\n        save_path (str, optional): Path of the pkl to be saved. Default: None.\n        workers (int, optional): Number of threads to be used. Default: 4.\n        kwargs (dict): Additional parameters for dataset-specific Data class.\n            May include `use_v1` for SUN RGB-D and `num_points`.\n    "
    assert os.path.exists(data_path)
    assert pkl_prefix in ['sunrgbd', 'scannet', 's3dis'], f'unsupported indoor dataset {pkl_prefix}'
    save_path = data_path if save_path is None else save_path
    assert os.path.exists(save_path)
    if pkl_prefix in ['sunrgbd', 'scannet']:
        train_filename = os.path.join(save_path, f'{pkl_prefix}_infos_train.pkl')
        val_filename = os.path.join(save_path, f'{pkl_prefix}_infos_val.pkl')
        if pkl_prefix == 'sunrgbd':
            num_points = kwargs.get('num_points', -1)
            use_v1 = kwargs.get('use_v1', False)
            train_dataset = SUNRGBDData(root_path=data_path, split='train', use_v1=use_v1, num_points=num_points)
            val_dataset = SUNRGBDData(root_path=data_path, split='val', use_v1=use_v1, num_points=num_points)
        else:
            train_dataset = ScanNetData(root_path=data_path, split='train')
            val_dataset = ScanNetData(root_path=data_path, split='val')
            test_dataset = ScanNetData(root_path=data_path, split='test')
            test_filename = os.path.join(save_path, f'{pkl_prefix}_infos_test.pkl')
        infos_train = train_dataset.get_infos(num_workers=workers, has_label=True)
        mmcv.dump(infos_train, train_filename, 'pkl')
        print(f'{pkl_prefix} info train file is saved to {train_filename}')
        infos_val = val_dataset.get_infos(num_workers=workers, has_label=True)
        mmcv.dump(infos_val, val_filename, 'pkl')
        print(f'{pkl_prefix} info val file is saved to {val_filename}')
    if pkl_prefix == 'scannet':
        infos_test = test_dataset.get_infos(num_workers=workers, has_label=False)
        mmcv.dump(infos_test, test_filename, 'pkl')
        print(f'{pkl_prefix} info test file is saved to {test_filename}')
    if pkl_prefix == 'scannet':
        num_points = kwargs.get('num_points', 8192)
        train_dataset = ScanNetSegData(data_root=data_path, ann_file=train_filename, split='train', num_points=num_points, label_weight_func=lambda x: 1.0 / np.log(1.2 + x))
        val_dataset = ScanNetSegData(data_root=data_path, ann_file=val_filename, split='val', num_points=num_points, label_weight_func=lambda x: 1.0 / np.log(1.2 + x))
        train_dataset.get_seg_infos()
        val_dataset.get_seg_infos()
    elif pkl_prefix == 's3dis':
        splits = [f'Area_{i}' for i in [1, 2, 3, 4, 5, 6]]
        for split in splits:
            dataset = S3DISData(root_path=data_path, split=split)
            info = dataset.get_infos(num_workers=workers, has_label=True)
            filename = os.path.join(save_path, f'{pkl_prefix}_infos_{split}.pkl')
            mmcv.dump(info, filename, 'pkl')
            print(f'{pkl_prefix} info {split} file is saved to {filename}')
            num_points = kwargs.get('num_points', 4096)
            seg_dataset = S3DISSegData(data_root=data_path, ann_file=filename, split=split, num_points=num_points, label_weight_func=lambda x: 1.0 / np.log(1.2 + x))
            seg_dataset.get_seg_infos()