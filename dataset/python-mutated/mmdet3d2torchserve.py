from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import mmcv
try:
    from model_archiver.model_packaging import package_model
    from model_archiver.model_packaging_utils import ModelExportUtils
except ImportError:
    package_model = None

def mmdet3d2torchserve(config_file: str, checkpoint_file: str, output_folder: str, model_name: str, model_version: str='1.0', force: bool=False):
    if False:
        while True:
            i = 10
    "Converts MMDetection3D model (config + checkpoint) to TorchServe `.mar`.\n\n    Args:\n        config_file (str):\n            In MMDetection3D config format.\n            The contents vary for each task repository.\n        checkpoint_file (str):\n            In MMDetection3D checkpoint format.\n            The contents vary for each task repository.\n        output_folder (str):\n            Folder where `{model_name}.mar` will be created.\n            The file created will be in TorchServe archive format.\n        model_name (str):\n            If not None, used for naming the `{model_name}.mar` file\n            that will be created under `output_folder`.\n            If None, `{Path(checkpoint_file).stem}` will be used.\n        model_version (str, optional):\n            Model's version. Default: '1.0'.\n        force (bool, optional):\n            If True, if there is an existing `{model_name}.mar`\n            file under `output_folder` it will be overwritten.\n            Default: False.\n    "
    mmcv.mkdir_or_exist(output_folder)
    config = mmcv.Config.fromfile(config_file)
    with TemporaryDirectory() as tmpdir:
        config.dump(f'{tmpdir}/config.py')
        args = Namespace(**{'model_file': f'{tmpdir}/config.py', 'serialized_file': checkpoint_file, 'handler': f'{Path(__file__).parent}/mmdet3d_handler.py', 'model_name': model_name or Path(checkpoint_file).stem, 'version': model_version, 'export_path': output_folder, 'force': force, 'requirements_file': None, 'extra_files': None, 'runtime': 'python', 'archive_format': 'default'})
        manifest = ModelExportUtils.generate_manifest_json(args)
        package_model(args, manifest)

def parse_args():
    if False:
        print('Hello World!')
    parser = ArgumentParser(description='Convert MMDetection models to TorchServe `.mar` format.')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument('--output-folder', type=str, required=True, help='Folder where `{model_name}.mar` will be created.')
    parser.add_argument('--model-name', type=str, default=None, help='If not None, used for naming the `{model_name}.mar`file that will be created under `output_folder`.If None, `{Path(checkpoint_file).stem}` will be used.')
    parser.add_argument('--model-version', type=str, default='1.0', help='Number used for versioning.')
    parser.add_argument('-f', '--force', action='store_true', help='overwrite the existing `{model_name}.mar`')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    if package_model is None:
        raise ImportError('`torch-model-archiver` is required.Try: pip install torch-model-archiver')
    mmdet3d2torchserve(args.config, args.checkpoint, args.output_folder, args.model_name, args.model_version, args.force)