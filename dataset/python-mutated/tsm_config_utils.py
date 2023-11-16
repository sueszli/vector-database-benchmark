import logging
logger = logging.getLogger(__name__)
CONFIG_SECS = ['train', 'valid', 'test', 'infer']

class AttrDict(dict):

    def __getattr__(self, key):
        if False:
            print('Hello World!')
        return self[key]

    def __setattr__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

def parse_config(cfg_file):
    if False:
        for i in range(10):
            print('nop')
    'Load a config file into AttrDict'
    import yaml
    with open(cfg_file, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.Loader))
    create_attr_dict(yaml_config)
    return yaml_config

def create_attr_dict(yaml_config):
    if False:
        for i in range(10):
            print('nop')
    from ast import literal_eval
    for (key, value) in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value

def merge_configs(cfg, sec, args_dict):
    if False:
        i = 10
        return i + 15
    assert sec in CONFIG_SECS, f'invalid config section {sec}'
    sec_dict = getattr(cfg, sec.upper())
    for (k, v) in args_dict.items():
        if v is None:
            continue
        try:
            if hasattr(sec_dict, k):
                setattr(sec_dict, k, v)
        except:
            pass
    return cfg

def print_configs(cfg, mode):
    if False:
        while True:
            i = 10
    logger.info(f'---------------- {mode:>5} Arguments ----------------')
    for (sec, sec_items) in cfg.items():
        logger.info(f'{sec}:')
        for (k, v) in sec_items.items():
            logger.info(f'    {k}:{v}')
    logger.info('-------------------------------------------------')