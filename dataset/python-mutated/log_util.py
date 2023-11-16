import logging
from paddle.distributed.utils.log_utils import get_logger
logger = get_logger('INFO', __name__)

def set_log_level(level):
    if False:
        print('Hello World!')
    '\n    Set log level\n\n    Args:\n        level (str|int): a specified level\n\n    Example 1:\n        import paddle\n        import paddle.distributed.fleet as fleet\n        fleet.init()\n        fleet.setLogLevel("DEBUG")\n\n    Example 2:\n        import paddle\n        import paddle.distributed.fleet as fleet\n        fleet.init()\n        fleet.setLogLevel(1)\n\n    '
    assert isinstance(level, (str, int)), "level's type must be str or int"
    if isinstance(level, int):
        logger.setLevel(level)
    else:
        logger.setLevel(level.upper())

def get_log_level_code():
    if False:
        while True:
            i = 10
    '\n    Return current log level code\n    '
    return logger.getEffectiveLevel()

def get_log_level_name():
    if False:
        return 10
    '\n    Return current log level name\n    '
    return logging.getLevelName(get_log_level_code())

def layer_to_str(base, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    name = base + '('
    if args:
        name += ', '.join((str(arg) for arg in args))
        if kwargs:
            name += ', '
    if kwargs:
        name += ', '.join((f'{key}={str(value)}' for (key, value) in kwargs.items()))
    name += ')'
    return name