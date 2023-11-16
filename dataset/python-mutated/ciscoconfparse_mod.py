"""
Execution module for `ciscoconfparse <http://www.pennington.net/py/ciscoconfparse/index.html>`_

.. versionadded:: 2019.2.0

This module can be used for basic configuration parsing, audit or validation
for a variety of network platforms having Cisco IOS style configuration (one
space indentation), including: Cisco IOS, Cisco Nexus, Cisco IOS-XR,
Cisco IOS-XR, Cisco ASA, Arista EOS, Brocade, HP Switches, Dell PowerConnect
Switches, or Extreme Networks devices. In newer versions, ``ciscoconfparse``
provides support for brace-delimited configuration style as well, for platforms
such as: Juniper Junos, Palo Alto, or F5 Networks.

See http://www.pennington.net/py/ciscoconfparse/index.html for further details.

:depends: ciscoconfparse

This module depends on the Python library with the same name,
``ciscoconfparse`` - to install execute: ``pip install ciscoconfparse``.
"""
from salt.exceptions import SaltException
try:
    import ciscoconfparse
    HAS_CISCOCONFPARSE = True
except ImportError:
    HAS_CISCOCONFPARSE = False
__virtualname__ = 'ciscoconfparse'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    if HAS_CISCOCONFPARSE:
        return HAS_CISCOCONFPARSE
    else:
        return (False, 'Missing dependency ciscoconfparse')

def _get_ccp(config=None, config_path=None, saltenv='base'):
    if False:
        while True:
            i = 10
    ' '
    if config_path:
        config = __salt__['cp.get_file_str'](config_path, saltenv=saltenv)
        if config is False:
            raise SaltException('{} is not available'.format(config_path))
    if isinstance(config, str):
        config = config.splitlines()
    ccp = ciscoconfparse.CiscoConfParse(config)
    return ccp

def find_objects(config=None, config_path=None, regex=None, saltenv='base'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return all the line objects that match the expression in the ``regex``\n    argument.\n\n    .. warning::\n        This function is mostly valuable when invoked from other Salt\n        components (i.e., execution modules, states, templates etc.). For CLI\n        usage, please consider using\n        :py:func:`ciscoconfparse.find_lines <salt.ciscoconfparse_mod.find_lines>`\n\n    config\n        The configuration sent as text.\n\n        .. note::\n            This argument is ignored when ``config_path`` is specified.\n\n    config_path\n        The absolute or remote path to the file with the configuration to be\n        parsed. This argument supports the usual Salt filesystem URIs, e.g.,\n        ``salt://``, ``https://``, ``ftp://``, ``s3://``, etc.\n\n    regex\n        The regular expression to match the lines against.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file. This\n        argument is ignored when ``config_path`` is not a ``salt://`` URL.\n\n    Usage example:\n\n    .. code-block:: python\n\n        objects = __salt__['ciscoconfparse.find_objects'](config_path='salt://path/to/config.txt',\n                                                          regex='Gigabit')\n        for obj in objects:\n            print(obj.text)\n    "
    ccp = _get_ccp(config=config, config_path=config_path, saltenv=saltenv)
    lines = ccp.find_objects(regex)
    return lines

def find_lines(config=None, config_path=None, regex=None, saltenv='base'):
    if False:
        print('Hello World!')
    "\n    Return all the lines (as text) that match the expression in the ``regex``\n    argument.\n\n    config\n        The configuration sent as text.\n\n        .. note::\n            This argument is ignored when ``config_path`` is specified.\n\n    config_path\n        The absolute or remote path to the file with the configuration to be\n        parsed. This argument supports the usual Salt filesystem URIs, e.g.,\n        ``salt://``, ``https://``, ``ftp://``, ``s3://``, etc.\n\n    regex\n        The regular expression to match the lines against.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file. This\n        argument is ignored when ``config_path`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ciscoconfparse.find_lines config_path=https://bit.ly/2mAdq7z regex='ip address'\n\n    Output example:\n\n    .. code-block:: text\n\n       cisco-ios-router:\n            -  ip address dhcp\n            -  ip address 172.20.0.1 255.255.255.0\n            -  no ip address\n    "
    lines = find_objects(config=config, config_path=config_path, regex=regex, saltenv=saltenv)
    return [line.text for line in lines]

def find_objects_w_child(config=None, config_path=None, parent_regex=None, child_regex=None, ignore_ws=False, saltenv='base'):
    if False:
        while True:
            i = 10
    "\n    Parse through the children of all parent lines matching ``parent_regex``,\n    and return a list of child objects, which matched the ``child_regex``.\n\n    .. warning::\n        This function is mostly valuable when invoked from other Salt\n        components (i.e., execution modules, states, templates etc.). For CLI\n        usage, please consider using\n        :py:func:`ciscoconfparse.find_lines_w_child <salt.ciscoconfparse_mod.find_lines_w_child>`\n\n    config\n        The configuration sent as text.\n\n        .. note::\n            This argument is ignored when ``config_path`` is specified.\n\n    config_path\n        The absolute or remote path to the file with the configuration to be\n        parsed. This argument supports the usual Salt filesystem URIs, e.g.,\n        ``salt://``, ``https://``, ``ftp://``, ``s3://``, etc.\n\n    parent_regex\n        The regular expression to match the parent lines against.\n\n    child_regex\n        The regular expression to match the child lines against.\n\n    ignore_ws: ``False``\n        Whether to ignore the white spaces.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file. This\n        argument is ignored when ``config_path`` is not a ``salt://`` URL.\n\n    Usage example:\n\n    .. code-block:: python\n\n        objects = __salt__['ciscoconfparse.find_objects_w_child'](config_path='https://bit.ly/2mAdq7z',\n                                                                  parent_regex='line con',\n                                                                  child_regex='stopbits')\n        for obj in objects:\n            print(obj.text)\n    "
    ccp = _get_ccp(config=config, config_path=config_path, saltenv=saltenv)
    lines = ccp.find_objects_w_child(parent_regex, child_regex, ignore_ws=ignore_ws)
    return lines

def find_lines_w_child(config=None, config_path=None, parent_regex=None, child_regex=None, ignore_ws=False, saltenv='base'):
    if False:
        while True:
            i = 10
    "\n    Return a list of parent lines (as text)  matching the regular expression\n    ``parent_regex`` that have children lines matching ``child_regex``.\n\n    config\n        The configuration sent as text.\n\n        .. note::\n            This argument is ignored when ``config_path`` is specified.\n\n    config_path\n        The absolute or remote path to the file with the configuration to be\n        parsed. This argument supports the usual Salt filesystem URIs, e.g.,\n        ``salt://``, ``https://``, ``ftp://``, ``s3://``, etc.\n\n    parent_regex\n        The regular expression to match the parent lines against.\n\n    child_regex\n        The regular expression to match the child lines against.\n\n    ignore_ws: ``False``\n        Whether to ignore the white spaces.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file. This\n        argument is ignored when ``config_path`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ciscoconfparse.find_lines_w_child config_path=https://bit.ly/2mAdq7z parent_line='line con' child_line='stopbits'\n        salt '*' ciscoconfparse.find_lines_w_child config_path=https://bit.ly/2uIRxau parent_regex='ge-(.*)' child_regex='unit \\d+'\n    "
    lines = find_objects_w_child(config=config, config_path=config_path, parent_regex=parent_regex, child_regex=child_regex, ignore_ws=ignore_ws, saltenv=saltenv)
    return [line.text for line in lines]

def find_objects_wo_child(config=None, config_path=None, parent_regex=None, child_regex=None, ignore_ws=False, saltenv='base'):
    if False:
        print('Hello World!')
    "\n    Return a list of parent ``ciscoconfparse.IOSCfgLine`` objects, which matched\n    the ``parent_regex`` and whose children did *not* match ``child_regex``.\n    Only the parent ``ciscoconfparse.IOSCfgLine`` objects will be returned. For\n    simplicity, this method only finds oldest ancestors without immediate\n    children that match.\n\n    .. warning::\n        This function is mostly valuable when invoked from other Salt\n        components (i.e., execution modules, states, templates etc.). For CLI\n        usage, please consider using\n        :py:func:`ciscoconfparse.find_lines_wo_child <salt.ciscoconfparse_mod.find_lines_wo_child>`\n\n    config\n        The configuration sent as text.\n\n        .. note::\n            This argument is ignored when ``config_path`` is specified.\n\n    config_path\n        The absolute or remote path to the file with the configuration to be\n        parsed. This argument supports the usual Salt filesystem URIs, e.g.,\n        ``salt://``, ``https://``, ``ftp://``, ``s3://``, etc.\n\n    parent_regex\n        The regular expression to match the parent lines against.\n\n    child_regex\n        The regular expression to match the child lines against.\n\n    ignore_ws: ``False``\n        Whether to ignore the white spaces.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file. This\n        argument is ignored when ``config_path`` is not a ``salt://`` URL.\n\n    Usage example:\n\n    .. code-block:: python\n\n        objects = __salt__['ciscoconfparse.find_objects_wo_child'](config_path='https://bit.ly/2mAdq7z',\n                                                                   parent_regex='line con',\n                                                                   child_regex='stopbits')\n        for obj in objects:\n            print(obj.text)\n    "
    ccp = _get_ccp(config=config, config_path=config_path, saltenv=saltenv)
    lines = ccp.find_objects_wo_child(parent_regex, child_regex, ignore_ws=ignore_ws)
    return lines

def find_lines_wo_child(config=None, config_path=None, parent_regex=None, child_regex=None, ignore_ws=False, saltenv='base'):
    if False:
        i = 10
        return i + 15
    "\n    Return a list of parent ``ciscoconfparse.IOSCfgLine`` lines as text, which\n    matched the ``parent_regex`` and whose children did *not* match ``child_regex``.\n    Only the parent ``ciscoconfparse.IOSCfgLine`` text lines  will be returned.\n    For simplicity, this method only finds oldest ancestors without immediate\n    children that match.\n\n    config\n        The configuration sent as text.\n\n        .. note::\n            This argument is ignored when ``config_path`` is specified.\n\n    config_path\n        The absolute or remote path to the file with the configuration to be\n        parsed. This argument supports the usual Salt filesystem URIs, e.g.,\n        ``salt://``, ``https://``, ``ftp://``, ``s3://``, etc.\n\n    parent_regex\n        The regular expression to match the parent lines against.\n\n    child_regex\n        The regular expression to match the child lines against.\n\n    ignore_ws: ``False``\n        Whether to ignore the white spaces.\n\n    saltenv: ``base``\n        Salt fileserver environment from which to retrieve the file. This\n        argument is ignored when ``config_path`` is not a ``salt://`` URL.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ciscoconfparse.find_lines_wo_child config_path=https://bit.ly/2mAdq7z parent_line='line con' child_line='stopbits'\n    "
    lines = find_objects_wo_child(config=config, config_path=config_path, parent_regex=parent_regex, child_regex=child_regex, ignore_ws=ignore_ws, saltenv=saltenv)
    return [line.text for line in lines]

def filter_lines(config=None, config_path=None, parent_regex=None, child_regex=None, saltenv='base'):
    if False:
        i = 10
        return i + 15
    "\n    Return a list of detailed matches, for the configuration blocks (parent-child\n    relationship) whose parent respects the regular expressions configured via\n    the ``parent_regex`` argument, and the child matches the ``child_regex``\n    regular expression. The result is a list of dictionaries with the following\n    keys:\n\n    - ``match``: a boolean value that tells whether ``child_regex`` matched any\n      children lines.\n    - ``parent``: the parent line (as text).\n    - ``child``: the child line (as text). If no child line matched, this field\n      will be ``None``.\n\n    Note that the return list contains the elements that matched the parent\n    condition, the ``parent_regex`` regular expression. Therefore, the ``parent``\n    field will always have a valid value, while ``match`` and ``child`` may\n    default to ``False`` and ``None`` respectively when there is not child match.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' ciscoconfparse.filter_lines config_path=https://bit.ly/2mAdq7z parent_regex='Gigabit' child_regex='shutdown'\n\n    Example output (for the example above):\n\n    .. code-block:: python\n\n        [\n            {\n                'parent': 'interface GigabitEthernet1',\n                'match': False,\n                'child': None\n            },\n            {\n                'parent': 'interface GigabitEthernet2',\n                'match': True,\n                'child': ' shutdown'\n            },\n            {\n                'parent': 'interface GigabitEthernet3',\n                'match': True,\n                'child': ' shutdown'\n            }\n        ]\n    "
    ret = []
    ccp = _get_ccp(config=config, config_path=config_path, saltenv=saltenv)
    parent_lines = ccp.find_objects(parent_regex)
    for parent_line in parent_lines:
        child_lines = parent_line.re_search_children(child_regex)
        if child_lines:
            for child_line in child_lines:
                ret.append({'match': True, 'parent': parent_line.text, 'child': child_line.text})
        else:
            ret.append({'match': False, 'parent': parent_line.text, 'child': None})
    return ret