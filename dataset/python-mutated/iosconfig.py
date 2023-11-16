"""
Cisco IOS configuration manipulation helpers

.. versionadded:: 2019.2.0

This module provides a collection of helper functions for Cisco IOS style
configuration manipulation. This module does not have external dependencies
and can be used from any Proxy or regular Minion.
"""
import difflib
import salt.utils.dictdiffer
import salt.utils.dictupdate
from salt.exceptions import SaltException
from salt.utils.odict import OrderedDict
__virtualname__ = 'iosconfig'
__proxyenabled__ = ['*']

def _attach_data_to_path(obj, ele, data):
    if False:
        i = 10
        return i + 15
    if ele not in obj:
        obj[ele] = OrderedDict()
        obj[ele] = data
    else:
        obj[ele].update(data)

def _attach_data_to_path_tags(obj, path, data, list_=False):
    if False:
        return 10
    if '#list' not in obj:
        obj['#list'] = []
    path = [path]
    obj_tmp = obj
    first = True
    while True:
        obj_tmp['#text'] = ' '.join(path)
        path_item = path.pop(0)
        if not path:
            break
        else:
            if path_item not in obj_tmp:
                obj_tmp[path_item] = OrderedDict()
            obj_tmp = obj_tmp[path_item]
            if first and list_:
                obj['#list'].append({path_item: obj_tmp})
                first = False
    if path_item in obj_tmp:
        obj_tmp[path_item].update(data)
    else:
        obj_tmp[path_item] = data
    obj_tmp[path_item]['#standalone'] = True

def _parse_text_config(config_lines, with_tags=False, current_indent=0, nested=False):
    if False:
        print('Hello World!')
    struct_cfg = OrderedDict()
    while config_lines:
        line = config_lines.pop(0)
        if not line.strip() or line.lstrip().startswith('!'):
            continue
        current_line = line.lstrip()
        leading_spaces = len(line) - len(current_line)
        if leading_spaces > current_indent:
            current_block = _parse_text_config(config_lines, current_indent=leading_spaces, with_tags=with_tags, nested=True)
            if with_tags:
                _attach_data_to_path_tags(struct_cfg, current_line, current_block, nested)
            else:
                _attach_data_to_path(struct_cfg, current_line, current_block)
        elif leading_spaces < current_indent:
            config_lines.insert(0, line)
            break
        elif not nested:
            current_block = _parse_text_config(config_lines, current_indent=leading_spaces, with_tags=with_tags, nested=True)
            if with_tags:
                _attach_data_to_path_tags(struct_cfg, current_line, current_block, nested)
            else:
                _attach_data_to_path(struct_cfg, current_line, current_block)
        else:
            config_lines.insert(0, line)
            break
    return struct_cfg

def _get_diff_text(old, new):
    if False:
        while True:
            i = 10
    '\n    Returns the diff of two text blobs.\n    '
    diff = difflib.unified_diff(old.splitlines(1), new.splitlines(1))
    return ''.join([x.replace('\r', '') for x in diff])

def _print_config_text(tree, indentation=0):
    if False:
        i = 10
        return i + 15
    '\n    Return the config as text from a config tree.\n    '
    config = ''
    for (key, value) in tree.items():
        config += '{indent}{line}\n'.format(indent=' ' * indentation, line=key)
        if value:
            config += _print_config_text(value, indentation=indentation + 1)
    return config

def tree(config=None, path=None, with_tags=False, saltenv='base'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Transform Cisco IOS style configuration to structured Python dictionary.\n    Depending on the value of the ``with_tags`` argument, this function may\n    provide different views, valuable in different situations.\n\n    config\n        The configuration sent as text. This argument is ignored when ``path``\n        is configured.\n\n    path\n        Absolute or remote path from where to load the configuration text. This\n        argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    with_tags: ``False``\n        Whether this function should return a detailed view, with tags.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``path`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' iosconfig.tree path=salt://path/to/my/config.txt\n        salt '*' iosconfig.tree path=https://bit.ly/2mAdq7z\n    "
    if path:
        config = __salt__['cp.get_file_str'](path, saltenv=saltenv)
        if config is False:
            raise SaltException('{} is not available'.format(path))
    config_lines = config.splitlines()
    return _parse_text_config(config_lines, with_tags=with_tags)

def clean(config=None, path=None, saltenv='base'):
    if False:
        while True:
            i = 10
    "\n    Return a clean version of the config, without any special signs (such as\n    ``!`` as an individual line) or empty lines, but just lines with significant\n    value in the configuration of the network device.\n\n    config\n        The configuration sent as text. This argument is ignored when ``path``\n        is configured.\n\n    path\n        Absolute or remote path from where to load the configuration text. This\n        argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``path`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' iosconfig.clean path=salt://path/to/my/config.txt\n        salt '*' iosconfig.clean path=https://bit.ly/2mAdq7z\n    "
    config_tree = tree(config=config, path=path, saltenv=saltenv)
    return _print_config_text(config_tree)

def merge_tree(initial_config=None, initial_path=None, merge_config=None, merge_path=None, saltenv='base'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the merge tree of the ``initial_config`` with the ``merge_config``,\n    as a Python dictionary.\n\n    initial_config\n        The initial configuration sent as text. This argument is ignored when\n        ``initial_path`` is set.\n\n    initial_path\n        Absolute or remote path from where to load the initial configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    merge_config\n        The config to be merged into the initial config, sent as text. This\n        argument is ignored when ``merge_path`` is set.\n\n    merge_path\n        Absolute or remote path from where to load the merge configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``initial_path`` or ``merge_path`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' iosconfig.merge_tree initial_path=salt://path/to/running.cfg merge_path=salt://path/to/merge.cfg\n    "
    merge_tree = tree(config=merge_config, path=merge_path, saltenv=saltenv)
    initial_tree = tree(config=initial_config, path=initial_path, saltenv=saltenv)
    return salt.utils.dictupdate.merge(initial_tree, merge_tree)

def merge_text(initial_config=None, initial_path=None, merge_config=None, merge_path=None, saltenv='base'):
    if False:
        print('Hello World!')
    "\n    Return the merge result of the ``initial_config`` with the ``merge_config``,\n    as plain text.\n\n    initial_config\n        The initial configuration sent as text. This argument is ignored when\n        ``initial_path`` is set.\n\n    initial_path\n        Absolute or remote path from where to load the initial configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    merge_config\n        The config to be merged into the initial config, sent as text. This\n        argument is ignored when ``merge_path`` is set.\n\n    merge_path\n        Absolute or remote path from where to load the merge configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``initial_path`` or ``merge_path`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' iosconfig.merge_text initial_path=salt://path/to/running.cfg merge_path=salt://path/to/merge.cfg\n    "
    candidate_tree = merge_tree(initial_config=initial_config, initial_path=initial_path, merge_config=merge_config, merge_path=merge_path, saltenv=saltenv)
    return _print_config_text(candidate_tree)

def merge_diff(initial_config=None, initial_path=None, merge_config=None, merge_path=None, saltenv='base'):
    if False:
        return 10
    "\n    Return the merge diff, as text, after merging the merge config into the\n    initial config.\n\n    initial_config\n        The initial configuration sent as text. This argument is ignored when\n        ``initial_path`` is set.\n\n    initial_path\n        Absolute or remote path from where to load the initial configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    merge_config\n        The config to be merged into the initial config, sent as text. This\n        argument is ignored when ``merge_path`` is set.\n\n    merge_path\n        Absolute or remote path from where to load the merge configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``initial_path`` or ``merge_path`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' iosconfig.merge_diff initial_path=salt://path/to/running.cfg merge_path=salt://path/to/merge.cfg\n    "
    if initial_path:
        initial_config = __salt__['cp.get_file_str'](initial_path, saltenv=saltenv)
    candidate_config = merge_text(initial_config=initial_config, merge_config=merge_config, merge_path=merge_path, saltenv=saltenv)
    clean_running_dict = tree(config=initial_config)
    clean_running = _print_config_text(clean_running_dict)
    return _get_diff_text(clean_running, candidate_config)

def diff_tree(candidate_config=None, candidate_path=None, running_config=None, running_path=None, saltenv='base'):
    if False:
        return 10
    "\n    Return the diff, as Python dictionary, between the candidate and the running\n    configuration.\n\n    candidate_config\n        The candidate configuration sent as text. This argument is ignored when\n        ``candidate_path`` is set.\n\n    candidate_path\n        Absolute or remote path from where to load the candidate configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    running_config\n        The running configuration sent as text. This argument is ignored when\n        ``running_path`` is set.\n\n    running_path\n        Absolute or remote path from where to load the running configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``candidate_path`` or ``running_path`` is not a\n        ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' iosconfig.diff_tree candidate_path=salt://path/to/candidate.cfg running_path=salt://path/to/running.cfg\n    "
    candidate_tree = tree(config=candidate_config, path=candidate_path, saltenv=saltenv)
    running_tree = tree(config=running_config, path=running_path, saltenv=saltenv)
    return salt.utils.dictdiffer.deep_diff(running_tree, candidate_tree)

def diff_text(candidate_config=None, candidate_path=None, running_config=None, running_path=None, saltenv='base'):
    if False:
        print('Hello World!')
    "\n    Return the diff, as text, between the candidate and the running config.\n\n    candidate_config\n        The candidate configuration sent as text. This argument is ignored when\n        ``candidate_path`` is set.\n\n    candidate_path\n        Absolute or remote path from where to load the candidate configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    running_config\n        The running configuration sent as text. This argument is ignored when\n        ``running_path`` is set.\n\n    running_path\n        Absolute or remote path from where to load the running configuration\n        text. This argument allows any URI supported by\n        :py:func:`cp.get_url <salt.modules.cp.get_url>`), e.g., ``salt://``,\n        ``https://``, ``s3://``, ``ftp:/``, etc.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file.\n        Ignored if ``candidate_path`` or ``running_path`` is not a\n        ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' iosconfig.diff_text candidate_path=salt://path/to/candidate.cfg running_path=salt://path/to/running.cfg\n    "
    candidate_text = clean(config=candidate_config, path=candidate_path, saltenv=saltenv)
    running_text = clean(config=running_config, path=running_path, saltenv=saltenv)
    return _get_diff_text(running_text, candidate_text)