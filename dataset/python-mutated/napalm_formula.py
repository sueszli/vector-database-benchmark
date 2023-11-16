"""
NAPALM Formula helpers
======================

.. versionadded:: 2019.2.0

This is an Execution Module providing helpers for various NAPALM formulas,
e.g., napalm-interfaces-formula, napalm-bgp-formula, napalm-ntp-formula etc.,
meant to provide various helper functions to make the templates more readable.
"""
import copy
import fnmatch
import logging
import salt.utils.dictupdate
import salt.utils.napalm
from salt.defaults import DEFAULT_TARGET_DELIM
from salt.utils.data import traverse_dict_and_list as _traverse_dict_and_list
__proxyenabled__ = ['*']
__virtualname__ = 'napalm_formula'
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Available only on NAPALM Minions.\n    '
    return salt.utils.napalm.virtual(__opts__, __virtualname__, __file__)

def _container_path(model, key=None, container=None, delim=DEFAULT_TARGET_DELIM):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate all the possible paths within an OpenConfig-like object.\n    This function returns a generator.\n    '
    if not key:
        key = ''
    if not container:
        container = 'config'
    for (model_key, model_value) in model.items():
        if key:
            key_depth = '{prev_key}{delim}{cur_key}'.format(prev_key=key, delim=delim, cur_key=model_key)
        else:
            key_depth = model_key
        if model_key == container:
            yield key_depth
        else:
            yield from _container_path(model_value, key=key_depth, container=container, delim=delim)

def container_path(model, key=None, container=None, delim=DEFAULT_TARGET_DELIM):
    if False:
        print('Hello World!')
    '\n    Return the list of all the possible paths in a container, down to the\n    ``config`` container.\n    This function can be used to verify that the ``model`` is a Python object\n    correctly structured and respecting the OpenConfig hierarchy.\n\n    model\n        The OpenConfig-structured object to inspect.\n\n    delim: ``:``\n        The key delimiter. In particular cases, it is indicated to use ``//``\n        as ``:`` might be already used in various cases, e.g., IPv6 addresses,\n        interface name (e.g., Juniper QFX series), etc.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' napalm_formula.container_path "{\'interfaces\': {\'interface\': {\'Ethernet1\': {\'config\': {\'name\': \'Ethernet1\'}}}}}"\n\n    The example above would return a list with the following element:\n    ``interfaces:interface:Ethernet1:config`` which is the only possible path\n    in that hierarchy.\n\n    Other output examples:\n\n    .. code-block:: text\n\n        - interfaces:interface:Ethernet1:config\n        - interfaces:interface:Ethernet1:subinterfaces:subinterface:0:config\n        - interfaces:interface:Ethernet2:config\n    '
    return list(_container_path(model))

def setval(key, val, dict_=None, delim=DEFAULT_TARGET_DELIM):
    if False:
        i = 10
        return i + 15
    "\n    Set a value under the dictionary hierarchy identified\n    under the key. The target 'foo/bar/baz' returns the\n    dictionary hierarchy {'foo': {'bar': {'baz': {}}}}.\n\n    .. note::\n\n        Currently this doesn't work with integers, i.e.\n        cannot build lists dynamically.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' formula.setval foo:baz:bar True\n    "
    if not dict_:
        dict_ = {}
    prev_hier = dict_
    dict_hier = key.split(delim)
    for each in dict_hier[:-1]:
        if each not in prev_hier:
            prev_hier[each] = {}
        prev_hier = prev_hier[each]
    prev_hier[dict_hier[-1]] = copy.deepcopy(val)
    return dict_

def traverse(data, key, default=None, delimiter=DEFAULT_TARGET_DELIM):
    if False:
        i = 10
        return i + 15
    '\n    Traverse a dict or list using a colon-delimited (or otherwise delimited,\n    using the ``delimiter`` param) target string. The target ``foo:bar:0`` will\n    return ``data[\'foo\'][\'bar\'][0]`` if this value exists, and will otherwise\n    return the dict in the default argument.\n    Function will automatically determine the target type.\n    The target ``foo:bar:0`` will return data[\'foo\'][\'bar\'][0] if data like\n    ``{\'foo\':{\'bar\':[\'baz\']}}`` , if data like ``{\'foo\':{\'bar\':{\'0\':\'baz\'}}}``\n    then ``return data[\'foo\'][\'bar\'][\'0\']``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' napalm_formula.traverse "{\'foo\': {\'bar\': {\'baz\': True}}}" foo:baz:bar\n    '
    return _traverse_dict_and_list(data, key, default=default, delimiter=delimiter)

def dictupdate(dest, upd, recursive_update=True, merge_lists=False):
    if False:
        i = 10
        return i + 15
    '\n    Recursive version of the default dict.update\n\n    Merges upd recursively into dest\n\n    If recursive_update=False, will use the classic dict.update, or fall back\n    on a manual merge (helpful for non-dict types like ``FunctionWrapper``).\n\n    If ``merge_lists=True``, will aggregate list object types instead of replace.\n    The list in ``upd`` is added to the list in ``dest``, so the resulting list\n    is ``dest[key] + upd[key]``. This behaviour is only activated when\n    ``recursive_update=True``. By default ``merge_lists=False``.\n    '
    return salt.utils.dictupdate.update(dest, upd, recursive_update=recursive_update, merge_lists=merge_lists)

def defaults(model, defaults_, delim='//', flipped_merge=False):
    if False:
        while True:
            i = 10
    '\n    Apply the defaults to a Python dictionary having the structure as described\n    in the OpenConfig standards.\n\n    model\n        The OpenConfig model to apply the defaults to.\n\n    defaults\n        The dictionary of defaults. This argument must equally be structured\n        with respect to the OpenConfig standards.\n\n        For ease of use, the keys of these support glob matching, therefore\n        we don\'t have to provide the defaults for each entity but only for\n        the entity type. See an example below.\n\n    delim: ``//``\n        The key delimiter to use. Generally, ``//`` should cover all the possible\n        cases, and you don\'t need to override this value.\n\n    flipped_merge: ``False``\n        Whether should merge the model into the defaults, or the defaults\n        into the model. Default: ``False`` (merge the model into the defaults,\n        i.e., any defaults would be overridden by the values from the ``model``).\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' napalm_formula.defaults "{\'interfaces\': {\'interface\': {\'Ethernet1\': {\'config\': {\'name\': \'Ethernet1\'}}}}}" "{\'interfaces\': {\'interface\': {\'*\': {\'config\': {\'enabled\': True}}}}}"\n\n    As one can notice in the example above, the ``*`` corresponds to the\n    interface name, therefore, the defaults will be applied on all the\n    interfaces.\n    '
    merged = {}
    log.debug('Applying the defaults:')
    log.debug(defaults_)
    log.debug('openconfig like dictionary:')
    log.debug(model)
    for model_path in _container_path(model, delim=delim):
        for default_path in _container_path(defaults_, delim=delim):
            log.debug('Comparing %s to %s', model_path, default_path)
            if not fnmatch.fnmatch(model_path, default_path) or not len(model_path.split(delim)) == len(default_path.split(delim)):
                continue
            log.debug('%s matches %s', model_path, default_path)
            devault_val = _traverse_dict_and_list(defaults_, default_path, delimiter=delim)
            merged = setval(model_path, devault_val, dict_=merged, delim=delim)
    log.debug('Complete default dictionary')
    log.debug(merged)
    log.debug('Merging with the model')
    log.debug(model)
    if flipped_merge:
        return salt.utils.dictupdate.update(model, merged)
    return salt.utils.dictupdate.update(merged, model)

def render_field(dictionary, field, prepend=None, append=None, quotes=False, **opts):
    if False:
        return 10
    '\n    Render a field found under the ``field`` level of the hierarchy in the\n    ``dictionary`` object.\n    This is useful to render a field in a Jinja template without worrying that\n    the hierarchy might not exist. For example if we do the following in Jinja:\n    ``{{ interfaces.interface.Ethernet5.config.description }}`` for the\n    following object:\n    ``{\'interfaces\': {\'interface\': {\'Ethernet1\': {\'config\': {\'enabled\': True}}}}}``\n    it would error, as the ``Ethernet5`` key does not exist.\n    With this helper, we can skip this and avoid existence checks. This must be\n    however used with care.\n\n    dictionary\n        The dictionary to traverse.\n\n    field\n        The key name or part to traverse in the ``dictionary``.\n\n    prepend: ``None``\n        The text to prepend in front of the text. Usually, we need to have the\n        name of the field too when generating the configuration.\n\n    append: ``None``\n        Text to append at the end.\n\n    quotes: ``False``\n        Whether should wrap the text around quotes.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' napalm_formula.render_field "{\'enabled\': True}" enabled\n        # This would return the value of the ``enabled`` leaf key\n        salt \'*\' napalm_formula.render_field "{\'enabled\': True}" description\n        # This would not error\n\n    Jinja usage example:\n\n    .. code-block:: jinja\n\n        {%- set config = {\'enabled\': True, \'description\': \'Interface description\'} %}\n        {{ salt.napalm_formula.render_field(config, \'description\', quotes=True) }}\n\n    The example above would be rendered on Arista / Cisco as:\n\n    .. code-block:: text\n\n        description "Interface description"\n\n    While on Junos (the semicolon is important to be added, otherwise the\n    configuration won\'t be accepted by Junos):\n\n    .. code-block:: text\n\n        description "Interface description";\n    '
    value = traverse(dictionary, field)
    if value is None:
        return ''
    if prepend is None:
        prepend = field.replace('_', '-')
    if append is None:
        if __grains__['os'] in ('junos',):
            append = ';'
        else:
            append = ''
    if quotes:
        value = '"{value}"'.format(value=value)
    return '{prepend} {value}{append}'.format(prepend=prepend, value=value, append=append)

def render_fields(dictionary, *fields, **opts):
    if False:
        return 10
    '\n    This function works similarly to\n    :mod:`render_field <salt.modules.napalm_formula.render_field>` but for a\n    list of fields from the same dictionary, rendering, indenting and\n    distributing them on separate lines.\n\n    dictionary\n        The dictionary to traverse.\n\n    fields\n        A list of field names or paths in the dictionary.\n\n    indent: ``0``\n        The indentation to use, prepended to the rendered field.\n\n    separator: ``\\n``\n        The separator to use between fields.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' napalm_formula.render_fields "{\'mtu\': 68, \'description\': \'Interface description\'}" mtu description\n\n    Jinja usage example:\n\n    .. code-block:: jinja\n\n        {%- set config={\'mtu\': 68, \'description\': \'Interface description\'} %}\n        {{ salt.napalm_formula.render_fields(config, \'mtu\', \'description\', quotes=True) }}\n\n    The Jinja example above would generate the following configuration:\n\n    .. code-block:: text\n\n        mtu "68"\n        description "Interface description"\n    '
    results = []
    for field in fields:
        res = render_field(dictionary, field, **opts)
        if res:
            results.append(res)
    if 'indent' not in opts:
        opts['indent'] = 0
    if 'separator' not in opts:
        opts['separator'] = '\n{ind}'.format(ind=' ' * opts['indent'])
    return opts['separator'].join(results)