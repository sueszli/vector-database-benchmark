"""
Utilies for beacons
"""
import copy

def remove_hidden_options(config, whitelist):
    if False:
        while True:
            i = 10
    '\n    Remove any hidden options not whitelisted\n    '
    for entry in copy.copy(config):
        for func in entry:
            if func.startswith('_') and func not in whitelist:
                config.remove(entry)
    return config

def list_to_dict(config):
    if False:
        print('Hello World!')
    '\n    Convert list based beacon configuration\n    into a dictionary.\n    '
    _config = {}
    list(map(_config.update, config))
    return _config