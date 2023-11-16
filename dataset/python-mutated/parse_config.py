import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json

class ConfigParser:

    def __init__(self, config, resume=None, modification=None, run_id=None):
        if False:
            i = 10
            return i + 15
        '\n        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving\n        and logging module.\n        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.\n        :param resume: String, path to the checkpoint being loaded.\n        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.\n        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default\n        '
        self._config = _update_config(config, modification)
        self.resume = resume
        save_dir = Path(self.config['recon_trainer']['save_dir'])
        exper_name = self.config['name']
        if run_id is None:
            run_id = datetime.now().strftime('%m%d_%H%M%S')
        self._save_dir = save_dir / 'recon_models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
        write_json(self.config, self.save_dir / 'config_R.json')
        setup_logging(self.log_dir)
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    @classmethod
    def from_args(cls, args, options=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize this class from some cli arguments. Used in train, test.\n        '
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()
        if args.device is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        config = read_json(cfg_fname)
        if args.config and resume:
            config.update(read_json(args.config))
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        if False:
            print('Hello World!')
        "\n        Finds a function handle with the name given as 'type' in config, and returns the\n        instance initialized with corresponding arguments given.\n\n        `object = config.init_obj('name', module, a, b=1)`\n        is equivalent to\n        `object = module.name(a, b=1)`\n        "
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        if False:
            return 10
        "\n        Finds a function handle with the name given as 'type' in config, and returns the\n        function with given arguments fixed with functools.partial.\n\n        `function = config.init_ftn('name', module, a, b=1)`\n        is equivalent to\n        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.\n        "
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        if False:
            while True:
                i = 10
        'Access items like ordinary dict.'
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        if False:
            return 10
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    @property
    def config(self):
        if False:
            while True:
                i = 10
        return self._config

    @property
    def save_dir(self):
        if False:
            while True:
                i = 10
        return self._save_dir

    @property
    def log_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return self._log_dir

def _update_config(config, modification):
    if False:
        for i in range(10):
            print('nop')
    if modification is None:
        return config
    for (k, v) in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    if False:
        i = 10
        return i + 15
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    if False:
        for i in range(10):
            print('nop')
    'Set a value in a nested object in tree by sequence of keys.'
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    if False:
        for i in range(10):
            print('nop')
    'Access a nested object in tree by sequence of keys.'
    return reduce(getitem, keys, tree)