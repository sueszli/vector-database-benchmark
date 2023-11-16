import datetime
import importlib
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict
import fsspec
import torch

def to_cuda(x: torch.Tensor) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.contiguous()
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
    return x

def get_cuda():
    if False:
        for i in range(10):
            print('nop')
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return (use_cuda, device)

def get_git_branch():
    if False:
        i = 10
        return i + 15
    try:
        out = subprocess.check_output(['git', 'branch']).decode('utf8')
        current = next((line for line in out.split('\n') if line.startswith('*')))
        current.replace('* ', '')
    except subprocess.CalledProcessError:
        current = 'inside_docker'
    except FileNotFoundError:
        current = 'unknown'
    except StopIteration:
        current = 'unknown'
    return current

def get_commit_hash():
    if False:
        i = 10
        return i + 15
    'https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script'
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = '0000000'
    return commit

def get_experiment_folder_path(root_path, model_name):
    if False:
        print('Hello World!')
    'Get an experiment folder path with the current date and time'
    date_str = datetime.datetime.now().strftime('%B-%d-%Y_%I+%M%p')
    commit_hash = get_commit_hash()
    output_folder = os.path.join(root_path, model_name + '-' + date_str + '-' + commit_hash)
    return output_folder

def remove_experiment_folder(experiment_path):
    if False:
        print('Hello World!')
    'Check folder if there is a checkpoint, otherwise remove the folder'
    fs = fsspec.get_mapper(experiment_path).fs
    checkpoint_files = fs.glob(experiment_path + '/*.pth')
    if not checkpoint_files:
        if fs.exists(experiment_path):
            fs.rm(experiment_path, recursive=True)
            print(' ! Run is removed from {}'.format(experiment_path))
    else:
        print(' ! Run is kept in {}'.format(experiment_path))

def count_parameters(model):
    if False:
        return 10
    'Count number of trainable parameters in a network'
    return sum((p.numel() for p in model.parameters() if p.requires_grad))

def to_camel(text):
    if False:
        for i in range(10):
            print('nop')
    text = text.capitalize()
    text = re.sub('(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), text)
    text = text.replace('Tts', 'TTS')
    text = text.replace('vc', 'VC')
    return text

def find_module(module_path: str, module_name: str) -> object:
    if False:
        print('Hello World!')
    module_name = module_name.lower()
    module = importlib.import_module(module_path + '.' + module_name)
    class_name = to_camel(module_name)
    return getattr(module, class_name)

def import_class(module_path: str) -> object:
    if False:
        while True:
            i = 10
    'Import a class from a module path.\n\n    Args:\n        module_path (str): The module path of the class.\n\n    Returns:\n        object: The imported class.\n    '
    class_name = module_path.split('.')[-1]
    module_path = '.'.join(module_path.split('.')[:-1])
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_import_path(obj: object) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get the import path of a class.\n\n    Args:\n        obj (object): The class object.\n\n    Returns:\n        str: The import path of the class.\n    '
    return '.'.join([type(obj).__module__, type(obj).__name__])

def get_user_data_dir(appname):
    if False:
        while True:
            i = 10
    TTS_HOME = os.environ.get('TTS_HOME')
    XDG_DATA_HOME = os.environ.get('XDG_DATA_HOME')
    if TTS_HOME is not None:
        ans = Path(TTS_HOME).expanduser().resolve(strict=False)
    elif XDG_DATA_HOME is not None:
        ans = Path(XDG_DATA_HOME).expanduser().resolve(strict=False)
    elif sys.platform == 'win32':
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders')
        (dir_, _) = winreg.QueryValueEx(key, 'Local AppData')
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == 'darwin':
        ans = Path('~/Library/Application Support/').expanduser()
    else:
        ans = Path.home().joinpath('.local/share')
    return ans.joinpath(appname)

def set_init_dict(model_dict, checkpoint_state, c):
    if False:
        while True:
            i = 10
    for (k, v) in checkpoint_state.items():
        if k not in model_dict:
            print(' | > Layer missing in the model definition: {}'.format(k))
    pretrained_dict = {k: v for (k, v) in checkpoint_state.items() if k in model_dict}
    pretrained_dict = {k: v for (k, v) in pretrained_dict.items() if v.numel() == model_dict[k].numel()}
    if c.has('reinit_layers') and c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {k: v for (k, v) in pretrained_dict.items() if reinit_layer_name not in k}
    model_dict.update(pretrained_dict)
    print(' | > {} / {} layers are restored.'.format(len(pretrained_dict), len(model_dict)))
    return model_dict

def format_aux_input(def_args: Dict, kwargs: Dict) -> Dict:
    if False:
        print('Hello World!')
    'Format kwargs to hande auxilary inputs to models.\n\n    Args:\n        def_args (Dict): A dictionary of argument names and their default values if not defined in `kwargs`.\n        kwargs (Dict): A `dict` or `kwargs` that includes auxilary inputs to the model.\n\n    Returns:\n        Dict: arguments with formatted auxilary inputs.\n    '
    kwargs = kwargs.copy()
    for name in def_args:
        if name not in kwargs or kwargs[name] is None:
            kwargs[name] = def_args[name]
    return kwargs

class KeepAverage:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.avg_values = {}
        self.iters = {}

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return self.avg_values[key]

    def items(self):
        if False:
            print('Hello World!')
        return self.avg_values.items()

    def add_value(self, name, init_val=0, init_iter=0):
        if False:
            for i in range(10):
                print('nop')
        self.avg_values[name] = init_val
        self.iters[name] = init_iter

    def update_value(self, name, value, weighted_avg=False):
        if False:
            return 10
        if name not in self.avg_values:
            self.add_value(name, init_val=value)
        elif weighted_avg:
            self.avg_values[name] = 0.99 * self.avg_values[name] + 0.01 * value
            self.iters[name] += 1
        else:
            self.avg_values[name] = self.avg_values[name] * self.iters[name] + value
            self.iters[name] += 1
            self.avg_values[name] /= self.iters[name]

    def add_values(self, name_dict):
        if False:
            print('Hello World!')
        for (key, value) in name_dict.items():
            self.add_value(key, init_val=value)

    def update_values(self, value_dict):
        if False:
            print('Hello World!')
        for (key, value) in value_dict.items():
            self.update_value(key, value)

def get_timestamp():
    if False:
        while True:
            i = 10
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    if False:
        return 10
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)