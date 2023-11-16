"""
Return/control aspects of the grains data

Grains set or altered with this module are stored in the 'grains'
file on the minions. By default, this file is located at: ``/etc/salt/grains``

.. Note::

   This does **NOT** override any grains set in the minion config file.
"""
import collections
import logging
import math
import operator
import os
from collections.abc import Mapping
from functools import reduce
import salt.utils.compat
import salt.utils.data
import salt.utils.files
import salt.utils.json
import salt.utils.platform
import salt.utils.yaml
from salt.defaults import DEFAULT_TARGET_DELIM
from salt.exceptions import SaltException
__proxyenabled__ = ['*']
__grains__ = {}
__outputter__ = {'items': 'nested', 'item': 'nested', 'setval': 'nested'}
_infinitedict = lambda : collections.defaultdict(_infinitedict)
_non_existent_key = 'NonExistentValueMagicNumberSpK3hnufdHfeBUXCfqVK'
log = logging.getLogger(__name__)

def _serial_sanitizer(instr):
    if False:
        for i in range(10):
            print('nop')
    "Replaces the last 1/4 of a string with X's"
    length = len(instr)
    index = int(math.floor(length * 0.75))
    return '{}{}'.format(instr[:index], 'X' * (length - index))
_FQDN_SANITIZER = lambda x: 'MINION.DOMAINNAME'
_HOSTNAME_SANITIZER = lambda x: 'MINION'
_DOMAINNAME_SANITIZER = lambda x: 'DOMAINNAME'
_SANITIZERS = {'serialnumber': _serial_sanitizer, 'domain': _DOMAINNAME_SANITIZER, 'fqdn': _FQDN_SANITIZER, 'id': _FQDN_SANITIZER, 'host': _HOSTNAME_SANITIZER, 'localhost': _HOSTNAME_SANITIZER, 'nodename': _HOSTNAME_SANITIZER}

def get(key, default='', delimiter=DEFAULT_TARGET_DELIM, ordered=True):
    if False:
        while True:
            i = 10
    '\n    Attempt to retrieve the named value from grains, if the named value is not\n    available return the passed default. The default return is an empty string.\n\n    The value can also represent a value in a nested dict using a ":" delimiter\n    for the dict. This means that if a dict in grains looks like this::\n\n        {\'pkg\': {\'apache\': \'httpd\'}}\n\n    To retrieve the value associated with the apache key in the pkg dict this\n    key can be passed::\n\n        pkg:apache\n\n\n    :param delimiter:\n        Specify an alternate delimiter to use when traversing a nested dict.\n        This is useful for when the desired key contains a colon. See CLI\n        example below for usage.\n\n        .. versionadded:: 2014.7.0\n\n    :param ordered:\n        Outputs an ordered dict if applicable (default: True)\n\n        .. versionadded:: 2016.11.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' grains.get pkg:apache\n        salt \'*\' grains.get abc::def|ghi delimiter=\'|\'\n    '
    if ordered is True:
        grains = __grains__
    else:
        grains = salt.utils.json.loads(salt.utils.json.dumps(__grains__))
    return salt.utils.data.traverse_dict_and_list(grains, key, default, delimiter)

def has_value(key):
    if False:
        i = 10
        return i + 15
    "\n    Determine whether a key exists in the grains dictionary.\n\n    Given a grains dictionary that contains the following structure::\n\n        {'pkg': {'apache': 'httpd'}}\n\n    One would determine if the apache key in the pkg dict exists by::\n\n        pkg:apache\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grains.has_value pkg:apache\n    "
    return salt.utils.data.traverse_dict_and_list(__grains__, key, KeyError) is not KeyError

def items(sanitize=False):
    if False:
        return 10
    "\n    Return all of the minion's grains\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grains.items\n\n    Sanitized CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grains.items sanitize=True\n    "
    if salt.utils.data.is_true(sanitize):
        out = dict(__grains__)
        for (key, func) in _SANITIZERS.items():
            if key in out:
                out[key] = func(out[key])
        return out
    else:
        return dict(__grains__)

def item(*args, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Return one or more grains\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grains.item os\n        salt '*' grains.item os osrelease oscodename\n\n    Sanitized CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grains.item host sanitize=True\n    "
    ret = {}
    default = kwargs.get('default', '')
    delimiter = kwargs.get('delimiter', DEFAULT_TARGET_DELIM)
    try:
        for arg in args:
            ret[arg] = salt.utils.data.traverse_dict_and_list(__grains__, arg, default, delimiter)
    except KeyError:
        pass
    if salt.utils.data.is_true(kwargs.get('sanitize')):
        for (arg, func) in _SANITIZERS.items():
            if arg in ret:
                ret[arg] = func(ret[arg])
    return ret

def setvals(grains, destructive=False, refresh_pillar=True):
    if False:
        print('Hello World!')
    '\n    Set new grains values in the grains config file\n\n    destructive\n        If an operation results in a key being removed, delete the key, too.\n        Defaults to False.\n\n    refresh_pillar\n        Whether pillar will be refreshed.\n        Defaults to True.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' grains.setvals "{\'key1\': \'val1\', \'key2\': \'val2\'}"\n    '
    new_grains = grains
    if not isinstance(new_grains, Mapping):
        raise SaltException('setvals grains must be a dictionary.')
    grains = {}
    if os.path.isfile(__opts__['conf_file']):
        if salt.utils.platform.is_proxy():
            gfn = os.path.join(os.path.dirname(__opts__['conf_file']), 'proxy.d', __opts__['id'], 'grains')
        else:
            gfn = os.path.join(os.path.dirname(__opts__['conf_file']), 'grains')
    elif os.path.isdir(__opts__['conf_file']):
        if salt.utils.platform.is_proxy():
            gfn = os.path.join(__opts__['conf_file'], 'proxy.d', __opts__['id'], 'grains')
        else:
            gfn = os.path.join(__opts__['conf_file'], 'grains')
    elif salt.utils.platform.is_proxy():
        gfn = os.path.join(os.path.dirname(__opts__['conf_file']), 'proxy.d', __opts__['id'], 'grains')
    else:
        gfn = os.path.join(os.path.dirname(__opts__['conf_file']), 'grains')
    if os.path.isfile(gfn):
        with salt.utils.files.fopen(gfn, 'rb') as fp_:
            try:
                grains = salt.utils.yaml.safe_load(fp_)
            except salt.utils.yaml.YAMLError as exc:
                return 'Unable to read existing grains file: {}'.format(exc)
        if not isinstance(grains, dict):
            grains = {}
    for (key, val) in new_grains.items():
        if val is None and destructive is True:
            if key in grains:
                del grains[key]
            if key in __grains__:
                del __grains__[key]
        else:
            grains[key] = val
            __grains__[key] = val
    try:
        with salt.utils.files.fopen(gfn, 'w+', encoding='utf-8') as fp_:
            salt.utils.yaml.safe_dump(grains, fp_, default_flow_style=False)
    except OSError:
        log.error('Unable to write to grains file at %s. Check permissions.', gfn)
    fn_ = os.path.join(__opts__['cachedir'], 'module_refresh')
    try:
        with salt.utils.files.flopen(fn_, 'w+'):
            pass
    except OSError:
        log.error('Unable to write to cache file %s. Check permissions.', fn_)
    if not __opts__.get('local', False):
        __salt__['saltutil.refresh_grains'](refresh_pillar=refresh_pillar)
    return new_grains

def setval(key, val, destructive=False, refresh_pillar=True):
    if False:
        i = 10
        return i + 15
    '\n    Set a grains value in the grains config file\n\n    key\n        The grain key to be set.\n\n    val\n        The value to set the grain key to.\n\n    destructive\n        If an operation results in a key being removed, delete the key, too.\n        Defaults to False.\n\n    refresh_pillar\n        Whether pillar will be refreshed.\n        Defaults to True.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' grains.setval key val\n        salt \'*\' grains.setval key "{\'sub-key\': \'val\', \'sub-key2\': \'val2\'}"\n    '
    return setvals({key: val}, destructive, refresh_pillar=refresh_pillar)

def append(key, val, convert=False, delimiter=DEFAULT_TARGET_DELIM):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 0.17.0\n\n    Append a value to a list in the grains config file. If the grain doesn't\n    exist, the grain key is added and the value is appended to the new grain\n    as a list item.\n\n    key\n        The grain key to be appended to\n\n    val\n        The value to append to the grain key\n\n    convert\n        If convert is True, convert non-list contents into a list.\n        If convert is False and the grain contains non-list contents, an error\n        is given. Defaults to False.\n\n    delimiter\n        The key can be a nested dict key. Use this parameter to\n        specify the delimiter you use, instead of the default ``:``.\n        You can now append values to a list in nested dictionary grains. If the\n        list doesn't exist at this level, it will be created.\n\n        .. versionadded:: 2014.7.6\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grains.append key val\n    "
    grains = get(key, [], delimiter)
    if convert:
        if not isinstance(grains, list):
            grains = [] if grains is None else [grains]
    if not isinstance(grains, list):
        return 'The key {} is not a valid list'.format(key)
    if val in grains:
        return 'The val {} was already in the list {}'.format(val, key)
    if isinstance(val, list):
        for item in val:
            grains.append(item)
    else:
        grains.append(val)
    while delimiter in key:
        (key, rest) = key.rsplit(delimiter, 1)
        _grain = get(key, _infinitedict(), delimiter)
        if isinstance(_grain, dict):
            _grain.update({rest: grains})
        grains = _grain
    return setval(key, grains)

def remove(key, val, delimiter=DEFAULT_TARGET_DELIM):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 0.17.0\n\n    Remove a value from a list in the grains config file\n\n    key\n        The grain key to remove.\n\n    val\n        The value to remove.\n\n    delimiter\n        The key can be a nested dict key. Use this parameter to\n        specify the delimiter you use, instead of the default ``:``.\n        You can now append values to a list in nested dictionary grains. If the\n        list doesn't exist at this level, it will be created.\n\n        .. versionadded:: 2015.8.2\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grains.remove key val\n    "
    grains = get(key, [], delimiter)
    if not isinstance(grains, list):
        return 'The key {} is not a valid list'.format(key)
    if val not in grains:
        return 'The val {} was not in the list {}'.format(val, key)
    grains.remove(val)
    while delimiter in key:
        (key, rest) = key.rsplit(delimiter, 1)
        _grain = get(key, None, delimiter)
        if isinstance(_grain, dict):
            _grain.update({rest: grains})
        grains = _grain
    return setval(key, grains)

def delkey(key, force=False):
    if False:
        return 10
    "\n    .. versionadded:: 2017.7.0\n\n    Remove a grain completely from the grain system, this will remove the\n    grain key and value\n\n    key\n        The grain key from which to delete the value.\n\n    force\n        Force remove the grain even when it is a mapped value.\n        Defaults to False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grains.delkey key\n    "
    return delval(key, destructive=True, force=force)

def delval(key, destructive=False, force=False):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 0.17.0\n\n    Delete a grain value from the grains config file. This will just set the\n    grain value to ``None``. To completely remove the grain, run ``grains.delkey``\n    or pass ``destructive=True`` to ``grains.delval``.\n\n    key\n        The grain key from which to delete the value.\n\n    destructive\n        Delete the key, too. Defaults to False.\n\n    force\n        Force remove the grain even when it is a mapped value.\n        Defaults to False\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grains.delval key\n    "
    return set(key, None, destructive=destructive, force=force)

def ls():
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of all available grains\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grains.ls\n    "
    return sorted(__grains__)

def filter_by(lookup_dict, grain='os_family', merge=None, default='default', base=None):
    if False:
        while True:
            i = 10
    '\n    .. versionadded:: 0.17.0\n\n    Look up the given grain in a given dictionary for the current OS and return\n    the result\n\n    Although this may occasionally be useful at the CLI, the primary intent of\n    this function is for use in Jinja to make short work of creating lookup\n    tables for OS-specific data. For example:\n\n    .. code-block:: jinja\n\n        {% set apache = salt[\'grains.filter_by\']({\n            \'Debian\': {\'pkg\': \'apache2\', \'srv\': \'apache2\'},\n            \'RedHat\': {\'pkg\': \'httpd\', \'srv\': \'httpd\'},\n        }, default=\'Debian\') %}\n\n        myapache:\n          pkg.installed:\n            - name: {{ apache.pkg }}\n          service.running:\n            - name: {{ apache.srv }}\n\n    Values in the lookup table may be overridden by values in Pillar. An\n    example Pillar to override values in the example above could be as follows:\n\n    .. code-block:: yaml\n\n        apache:\n          lookup:\n            pkg: apache_13\n            srv: apache\n\n    The call to ``filter_by()`` would be modified as follows to reference those\n    Pillar values:\n\n    .. code-block:: jinja\n\n        {% set apache = salt[\'grains.filter_by\']({\n            ...\n        }, merge=salt[\'pillar.get\'](\'apache:lookup\')) %}\n\n\n    :param lookup_dict: A dictionary, keyed by a grain, containing a value or\n        values relevant to systems matching that grain. For example, a key\n        could be the grain for an OS and the value could the name of a package\n        on that particular OS.\n\n        .. versionchanged:: 2016.11.0\n\n            The dictionary key could be a globbing pattern. The function will\n            return the corresponding ``lookup_dict`` value where grain value\n            matches the pattern. For example:\n\n            .. code-block:: bash\n\n                # this will render \'got some salt\' if Minion ID begins from \'salt\'\n                salt \'*\' grains.filter_by \'{salt*: got some salt, default: salt is not here}\' id\n\n    :param grain: The name of a grain to match with the current system\'s\n        grains. For example, the value of the "os_family" grain for the current\n        system could be used to pull values from the ``lookup_dict``\n        dictionary.\n\n        .. versionchanged:: 2016.11.0\n\n            The grain value could be a list. The function will return the\n            ``lookup_dict`` value for a first found item in the list matching\n            one of the ``lookup_dict`` keys.\n\n    :param merge: A dictionary to merge with the results of the grain selection\n        from ``lookup_dict``. This allows Pillar to override the values in the\n        ``lookup_dict``. This could be useful, for example, to override the\n        values for non-standard package names such as when using a different\n        Python version from the default Python version provided by the OS\n        (e.g., ``python26-mysql`` instead of ``python-mysql``).\n\n    :param default: default lookup_dict\'s key used if the grain does not exists\n        or if the grain value has no match on lookup_dict.  If unspecified\n        the value is "default".\n\n        .. versionadded:: 2014.1.0\n\n    :param base: A lookup_dict key to use for a base dictionary.  The\n        grain-selected ``lookup_dict`` is merged over this and then finally\n        the ``merge`` dictionary is merged.  This allows common values for\n        each case to be collected in the base and overridden by the grain\n        selection dictionary and the merge dictionary.  Default is unset.\n\n        .. versionadded:: 2015.5.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' grains.filter_by \'{Debian: Debheads rule, RedHat: I love my hat}\'\n        # this one will render {D: {E: I, G: H}, J: K}\n        salt \'*\' grains.filter_by \'{A: B, C: {D: {E: F, G: H}}}\' \'xxx\' \'{D: {E: I}, J: K}\' \'C\'\n        # next one renders {A: {B: G}, D: J}\n        salt \'*\' grains.filter_by \'{default: {A: {B: C}, D: E}, F: {A: {B: G}}, H: {D: I}}\' \'xxx\' \'{D: J}\' \'F\' \'default\'\n        # next same as above when default=\'H\' instead of \'F\' renders {A: {B: C}, D: J}\n    '
    return salt.utils.data.filter_by(lookup_dict=lookup_dict, lookup=grain, traverse=__grains__, merge=merge, default=default, base=base)

def _dict_from_path(path, val, delimiter=DEFAULT_TARGET_DELIM):
    if False:
        i = 10
        return i + 15
    '\n    Given a lookup string in the form of \'foo:bar:baz" return a nested\n    dictionary of the appropriate depth with the final segment as a value.\n\n    >>> _dict_from_path(\'foo:bar:baz\', \'somevalue\')\n    {"foo": {"bar": {"baz": "somevalue"}}\n    '
    nested_dict = _infinitedict()
    keys = path.rsplit(delimiter)
    lastplace = reduce(operator.getitem, keys[:-1], nested_dict)
    lastplace[keys[-1]] = val
    return nested_dict

def set(key, val='', force=False, destructive=False, delimiter=DEFAULT_TARGET_DELIM):
    if False:
        for i in range(10):
            print('nop')
    "\n    Set a key to an arbitrary value. It is used like setval but works\n    with nested keys.\n\n    This function is conservative. It will only overwrite an entry if\n    its value and the given one are not a list or a dict. The ``force``\n    parameter is used to allow overwriting in all cases.\n\n    .. versionadded:: 2015.8.0\n\n    :param force: Force writing over existing entry if given or existing\n                  values are list or dict. Defaults to False.\n    :param destructive: If an operation results in a key being removed,\n                  delete the key, too. Defaults to False.\n    :param delimiter:\n        Specify an alternate delimiter to use when traversing a nested dict,\n        the default being ``:``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grains.set 'apps:myApp:port' 2209\n        salt '*' grains.set 'apps:myApp' '{port: 2209}'\n    "
    ret = {'comment': '', 'changes': {}, 'result': True}
    _new_value_type = 'simple'
    if isinstance(val, dict):
        _new_value_type = 'complex'
    elif isinstance(val, list):
        _new_value_type = 'complex'
    _non_existent = object()
    _existing_value = get(key, _non_existent, delimiter)
    _value = _existing_value
    _existing_value_type = 'simple'
    if _existing_value is _non_existent:
        _existing_value_type = None
    elif isinstance(_existing_value, dict):
        _existing_value_type = 'complex'
    elif isinstance(_existing_value, list):
        _existing_value_type = 'complex'
    if _existing_value_type is not None and _existing_value == val and (val is not None or destructive is not True):
        ret['comment'] = 'Grain is already set'
        return ret
    if _existing_value is not None and (not force):
        if _existing_value_type == 'complex':
            ret['comment'] = "The key '{}' exists but is a dict or a list. Use 'force=True' to overwrite.".format(key)
            ret['result'] = False
            return ret
        elif _new_value_type == 'complex' and _existing_value_type is not None:
            ret['comment'] = "The key '{}' exists and the given value is a dict or a list. Use 'force=True' to overwrite.".format(key)
            ret['result'] = False
            return ret
        else:
            _value = val
    else:
        _value = val
    while delimiter in key:
        (key, rest) = key.rsplit(delimiter, 1)
        _existing_value = get(key, {}, delimiter)
        if isinstance(_existing_value, dict):
            if _value is None and destructive:
                if rest in _existing_value.keys():
                    _existing_value.pop(rest)
            else:
                _existing_value.update({rest: _value})
        elif isinstance(_existing_value, list):
            _list_updated = False
            for (_index, _item) in enumerate(_existing_value):
                if _item == rest:
                    _existing_value[_index] = {rest: _value}
                    _list_updated = True
                elif isinstance(_item, dict) and rest in _item:
                    _item.update({rest: _value})
                    _list_updated = True
            if not _list_updated:
                _existing_value.append({rest: _value})
        elif _existing_value == rest or force:
            _existing_value = {rest: _value}
        else:
            ret['comment'] = "The key '{}' value is '{}', which is different from the provided key '{}'. Use 'force=True' to overwrite.".format(key, _existing_value, rest)
            ret['result'] = False
            return ret
        _value = _existing_value
    _setval_ret = setval(key, _value, destructive=destructive)
    if isinstance(_setval_ret, dict):
        ret['changes'] = _setval_ret
    else:
        ret['comment'] = _setval_ret
        ret['result'] = False
    return ret

def equals(key, value):
    if False:
        return 10
    "\n    Used to make sure the minion's grain key/value matches.\n\n    Returns ``True`` if matches otherwise ``False``.\n\n    .. versionadded:: 2017.7.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grains.equals fqdn <expected_fqdn>\n        salt '*' grains.equals systemd:version 219\n    "
    return str(value) == str(get(key))
fetch = get