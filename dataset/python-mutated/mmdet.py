import shutil
import sys
import urllib.request
from importlib import import_module
from os import path
from pathlib import Path
from typing import Optional
from sahi.utils.file import download_from_url

def mmdet_version_as_integer():
    if False:
        return 10
    import mmdet
    return int(mmdet.__version__.replace('.', ''))

class MmdetTestConstants:
    MMDET_CASCADEMASKRCNN_MODEL_URL = 'http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
    MMDET_CASCADEMASKRCNN_MODEL_PATH = 'tests/data/models/mmdet/cascade_mask_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth'
    MMDET_RETINANET_MODEL_URL = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
    MMDET_RETINANET_MODEL_PATH = 'tests/data/models/mmdet/retinanet/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
    MMDET_YOLOX_TINY_MODEL_URL = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
    MMDET_YOLOX_TINY_MODEL_PATH = 'tests/data/models/mmdet/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
    MMDET_CASCADEMASKRCNN_CONFIG_PATH = 'tests/data/models/mmdet/cascade_mask_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py'
    MMDET_RETINANET_CONFIG_PATH = 'tests/data/models/mmdet/retinanet/retinanet_r50_fpn_1x_coco.py'
    MMDET_YOLOX_TINY_CONFIG_PATH = 'tests/data/models/mmdet/yolox/yolox_tiny_8xb8-300e_coco.py'

def download_mmdet_cascade_mask_rcnn_model(destination_path: Optional[str]=None):
    if False:
        i = 10
        return i + 15
    if destination_path is None:
        destination_path = MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH
    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
    download_from_url(MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_URL, destination_path)

def download_mmdet_retinanet_model(destination_path: Optional[str]=None):
    if False:
        while True:
            i = 10
    if destination_path is None:
        destination_path = MmdetTestConstants.MMDET_RETINANET_MODEL_PATH
    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
    download_from_url(MmdetTestConstants.MMDET_RETINANET_MODEL_URL, destination_path)

def download_mmdet_yolox_tiny_model(destination_path: Optional[str]=None):
    if False:
        while True:
            i = 10
    if destination_path is None:
        destination_path = MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_PATH
    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
    download_from_url(MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_URL, destination_path)

def download_mmdet_config(model_name: str='cascade_rcnn', config_file_name: str='cascade_mask_rcnn_r50_fpn_1x_coco.py', verbose: bool=True) -> str:
    if False:
        print('Hello World!')
    '\n    Merges config files starting from given main config file name. Saves as single file.\n\n    Args:\n        model_name (str): mmdet model name. check https://github.com/open-mmlab/mmdetection/tree/master/configs.\n        config_file_name (str): mdmet config file name.\n        verbose (bool): if True, print save path.\n\n    Returns:\n        (str) abs path of the downloaded config file.\n    '
    from mmdet import __version__
    mmdet_ver = 'v' + __version__
    base_config_url = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/' + mmdet_ver + '/configs/' + model_name + '/'
    main_config_url = base_config_url + config_file_name
    configs_dir = Path('mmdet_configs') / mmdet_ver
    model_config_dir = configs_dir / model_name
    configs_dir.mkdir(parents=True, exist_ok=True)
    model_config_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(main_config_url).name
    final_config_path = str(model_config_dir / filename)
    if not Path(final_config_path).exists():
        temp_configs_dir = Path('temp_mmdet_configs')
        main_config_dir = temp_configs_dir / model_name
        temp_configs_dir.mkdir(parents=True, exist_ok=True)
        main_config_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(main_config_url).name
        main_config_path = str(main_config_dir / filename)
        urllib.request.urlretrieve(main_config_url, main_config_path)
        sys.path.insert(0, str(main_config_dir))
        temp_module_name = path.splitext(filename)[0]
        mod = import_module(temp_module_name)
        sys.path.pop(0)
        config_dict = {name: value for (name, value) in mod.__dict__.items() if not name.startswith('__')}
        if not isinstance(config_dict['_base_'], list):
            config_dict['_base_'] = [config_dict['_base_']]
        for secondary_config_file_path in config_dict['_base_']:
            config_url = base_config_url + secondary_config_file_path
            config_path = main_config_dir / secondary_config_file_path
            config_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(config_url, str(config_path))
            secondary_config_dir = config_path.parent
            sys.path.insert(0, str(secondary_config_dir))
            temp_module_name = path.splitext(Path(config_path).name)[0]
            mod = import_module(temp_module_name)
            sys.path.pop(0)
            secondary_config_dict = {name: value for (name, value) in mod.__dict__.items() if not name.startswith('__')}
            if secondary_config_dict.get('_base_') is not None:
                if not isinstance(secondary_config_dict['_base_'], list):
                    secondary_config_dict['_base_'] = [secondary_config_dict['_base_']]
                for third_config_file_path in secondary_config_dict['_base_']:
                    config_url = base_config_url + third_config_file_path
                    config_path = main_config_dir / third_config_file_path
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(config_url, str(config_path))
        from mmengine import Config
        config = Config.fromfile(main_config_path)
        config.dump(final_config_path)
        if verbose:
            print(f'mmdet config file has been downloaded to {path.abspath(final_config_path)}')
        shutil.rmtree(temp_configs_dir)
    return path.abspath(final_config_path)