"""
A beacon to execute salt execution module functions. This beacon will fire only if the return data is "truthy".
The function return, function name and args and/or kwargs, will be passed as data in the event.

The configuration can accept a list of salt functions to execute every interval.
Make sure to allot enough time via 'interval' key to allow all salt functions to execute.
The salt functions will be executed sequentially.

The elements in list of functions can be either a simple string (with no arguments) or a dictionary with a single
key being the salt execution module and sub keys indicating args and / or kwargs.

See example config below.

.. code-block:: yaml

    beacons:
      salt_monitor:
        - salt_fun:
        - slsutil.renderer:
            args:
              - salt://states/apache.sls
            kwargs:
              - default_renderer: jinja
        - test.ping
        - interval: 3600 # seconds
"""
import salt.utils.beacons

def _parse_args(args_kwargs_dict):
    if False:
        print('Hello World!')
    args = args_kwargs_dict.get('args', [])
    kwargs = args_kwargs_dict.get('kwargs', {})
    if kwargs:
        _kwargs = {}
        list(map(_kwargs.update, kwargs))
        kwargs = _kwargs
    return (args, kwargs)

def validate(config):
    if False:
        i = 10
        return i + 15
    config = salt.utils.beacons.list_to_dict(config)
    if isinstance(config['salt_fun'], str):
        fun = config['salt_fun']
        if fun not in __salt__:
            return (False, '{} not in __salt__'.format(fun))
    else:
        for entry in config['salt_fun']:
            if isinstance(entry, dict):
                (fun, args_kwargs_dict) = next(iter(entry.items()))
                for key in args_kwargs_dict:
                    if key == 'args':
                        if not isinstance(args_kwargs_dict[key], list):
                            return (False, 'args key for fun {} must be list'.format(fun))
                    elif key == 'kwargs':
                        if not isinstance(args_kwargs_dict[key], list):
                            return (False, 'kwargs key for fun {} must be list of key value pairs'.format(fun))
                        for key_value in args_kwargs_dict[key]:
                            if not isinstance(key_value, dict):
                                return (False, '{} is not a key / value pair'.format(key_value))
                    else:
                        return (False, 'key {} not allowed under fun {}'.format(key, fun))
            else:
                fun = entry
            if fun not in __salt__:
                return (False, '{} not in __salt__'.format(fun))
    return (True, 'valid config')

def beacon(config):
    if False:
        i = 10
        return i + 15
    events = []
    config = salt.utils.beacons.list_to_dict(config)
    if isinstance(config['salt_fun'], str):
        fun = config['salt_fun']
        ret = __salt__[fun]()
        return [{'salt_fun': fun, 'ret': ret}]
    for entry in config['salt_fun']:
        if isinstance(entry, dict):
            (fun, args_kwargs_dict) = list(entry.items())[0]
            (args, kwargs) = _parse_args(args_kwargs_dict)
        else:
            fun = entry
            args = ()
            kwargs = {}
        ret = __salt__[fun](*args, **kwargs)
        if ret:
            _ret = {'salt_fun': fun, 'ret': ret}
            if args:
                _ret['args'] = args
            if kwargs:
                _ret['kwargs'] = kwargs
            events.append(_ret)
    return events