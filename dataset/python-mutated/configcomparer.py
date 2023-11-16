"""
Utilities for comparing and updating configurations while keeping track of
changes in a way that can be easily reported in a state.
"""

def compare_and_update_config(config, update_config, changes, namespace=''):
    if False:
        for i in range(10):
            print('nop')
    '\n    Recursively compare two configs, writing any needed changes to the\n    update_config and capturing changes in the changes dict.\n    '
    if isinstance(config, dict):
        if not update_config:
            if config:
                changes[namespace] = {'new': config, 'old': update_config}
            return config
        elif not isinstance(update_config, dict):
            changes[namespace] = {'new': config, 'old': update_config}
            return config
        else:
            for (key, value) in config.items():
                _namespace = key
                if namespace:
                    _namespace = '{}.{}'.format(namespace, _namespace)
                update_config[key] = compare_and_update_config(value, update_config.get(key, None), changes, namespace=_namespace)
            return update_config
    elif isinstance(config, list):
        if not update_config:
            if config:
                changes[namespace] = {'new': config, 'old': update_config}
            return config
        elif not isinstance(update_config, list):
            changes[namespace] = {'new': config, 'old': update_config}
            return config
        else:
            for (idx, item) in enumerate(config):
                _namespace = '[{}]'.format(idx)
                if namespace:
                    _namespace = '{}{}'.format(namespace, _namespace)
                _update = None
                if len(update_config) > idx:
                    _update = update_config[idx]
                if _update:
                    update_config[idx] = compare_and_update_config(config[idx], _update, changes, namespace=_namespace)
                else:
                    changes[_namespace] = {'new': config[idx], 'old': _update}
                    update_config.append(config[idx])
            if len(update_config) > len(config):
                for (idx, old_item) in enumerate(update_config):
                    if idx < len(config):
                        continue
                    _namespace = '[{}]'.format(idx)
                    if namespace:
                        _namespace = '{}{}'.format(namespace, _namespace)
                    changes[_namespace] = {'new': None, 'old': old_item}
                del update_config[len(config):]
            return update_config
    else:
        if config != update_config:
            changes[namespace] = {'new': config, 'old': update_config}
        return config