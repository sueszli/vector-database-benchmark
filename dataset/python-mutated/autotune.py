import json
import warnings
import paddle
from paddle.base import core
__all__ = ['set_config']

def set_config(config=None):
    if False:
        i = 10
        return i + 15
    '\n    Set the configuration for kernel, layout and dataloader auto-tuning.\n\n    1. kernel: When it is enabled, exhaustive search method will be used to select\n    and cache the best algorithm for the operator in the tuning iteration. Tuning\n    parameters are as follows:\n\n    - enable(bool): Whether to enable kernel tuning.\n    - tuning_range(list): Start and end iteration for auto-tuning. Default: [1, 10].\n\n    2. layout: When it is enabled, the best data layout such as NCHW or NHWC will be\n    determined based on the device and data type. When the origin layout setting is\n    not best, layout transformation will be automaticly performed to improve model\n    performance. Layout auto-tuning only supports dygraph mode currently. Tuning\n    parameters are as follows:\n\n    - enable(bool): Whether to enable layout tuning.\n\n    3. dataloader: When it is enabled, the best num_workers will be selected to replace\n    the origin dataloader setting. Tuning parameters are as follows:\n\n    - enable(bool): Whether to enable dataloader tuning.\n\n    Args:\n        config (dict|str|None, optional): Configuration for auto-tuning. If it is a\n            dictionary, the key is the tuning type, and the value is a dictionary\n            of the corresponding tuning parameters. If it is a string, the path of\n            a json file will be specified and the tuning configuration will be set\n            by the json file. Default: None, auto-tuning for kernel, layout and\n            dataloader will be enabled.\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n            >>> import json\n\n            >>> # config is a dict.\n            >>> config = {\n            ...     "kernel": {\n            ...         "enable": True,\n            ...         "tuning_range": [1, 5],\n            ...     },\n            ...     "layout": {\n            ...         "enable": True,\n            ...     },\n            ...     "dataloader": {\n            ...         "enable": True,\n            ...     }\n            >>> }\n            >>> paddle.incubate.autotune.set_config(config)\n\n            >>> # config is the path of json file.\n            >>> config_json = json.dumps(config)\n            >>> with open(\'config.json\', \'w\') as json_file:\n            ...     json_file.write(config_json)\n            >>> paddle.incubate.autotune.set_config(\'config.json\')\n\n    '
    if config is None:
        core.enable_autotune()
        core.enable_layout_autotune()
        paddle.io.reader.set_autotune_config(use_autotune=True)
        return
    config_dict = {}
    if isinstance(config, dict):
        config_dict = config
    elif isinstance(config, str):
        try:
            with open(config, 'r') as filehandle:
                config_dict = json.load(filehandle)
        except Exception as e:
            print(f'Load config error: {e}')
            warnings.warn('Use default configuration for auto-tuning.')
    if 'kernel' in config_dict:
        kernel_config = config_dict['kernel']
        if 'enable' in kernel_config:
            if isinstance(kernel_config['enable'], bool):
                if kernel_config['enable']:
                    core.enable_autotune()
                else:
                    core.disable_autotune()
            else:
                warnings.warn('The auto-tuning configuration of the kernel is incorrect.The `enable` should be bool. Use default parameter instead.')
        if 'tuning_range' in kernel_config:
            if isinstance(kernel_config['tuning_range'], list):
                tuning_range = kernel_config['tuning_range']
                assert len(tuning_range) == 2
                core.set_autotune_range(tuning_range[0], tuning_range[1])
            else:
                warnings.warn('The auto-tuning configuration of the kernel is incorrect.The `tuning_range` should be list. Use default parameter instead.')
    if 'layout' in config_dict:
        layout_config = config_dict['layout']
        if 'enable' in layout_config:
            if isinstance(layout_config['enable'], bool):
                if layout_config['enable']:
                    core.enable_layout_autotune()
                else:
                    core.disable_layout_autotune()
            else:
                warnings.warn('The auto-tuning configuration of the layout is incorrect.The `enable` should be bool. Use default parameter instead.')
    if 'dataloader' in config_dict:
        dataloader_config = config_dict['dataloader']
        use_autoune = False
        if 'enable' in dataloader_config:
            if isinstance(dataloader_config['enable'], bool):
                use_autoune = dataloader_config['enable']
            else:
                warnings.warn('The auto-tuning configuration of the dataloader is incorrect.The `enable` should be bool. Use default parameter instead.')
        if 'tuning_steps' in dataloader_config:
            if isinstance(dataloader_config['tuning_steps'], int):
                paddle.io.reader.set_autotune_config(use_autoune, dataloader_config['tuning_steps'])
            else:
                warnings.warn('The auto-tuning configuration of the dataloader is incorrect.The `tuning_steps` should be int. Use default parameter instead.')
                paddle.io.reader.set_autotune_config(use_autoune)