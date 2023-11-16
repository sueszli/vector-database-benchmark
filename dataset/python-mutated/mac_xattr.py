"""
This module allows you to manage extended attributes on files or directories

.. code-block:: bash

    salt '*' xattr.list /path/to/file
"""
import logging
import salt.utils.args
import salt.utils.mac_utils
from salt.exceptions import CommandExecutionError
log = logging.getLogger(__name__)
__virtualname__ = 'xattr'
__func_alias__ = {'list_': 'list'}

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only work on Mac OS\n    '
    if __grains__['os'] in ['MacOS', 'Darwin']:
        return __virtualname__
    return (False, 'Only available on Mac OS systems')

def list_(path, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    List all of the extended attributes on the given file/directory\n\n    :param str path: The file(s) to get attributes from\n\n    :param bool hex: Return the values with forced hexadecimal values\n\n    :return: A dictionary containing extended attributes and values for the\n        given file\n    :rtype: dict\n\n    :raises: CommandExecutionError on file not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' xattr.list /path/to/file\n        salt '*' xattr.list /path/to/file hex=True\n    "
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    hex_ = kwargs.pop('hex', False)
    if kwargs:
        salt.utils.args.invalid_kwargs(kwargs)
    cmd = ['xattr', path]
    try:
        ret = salt.utils.mac_utils.execute_return_result(cmd)
    except CommandExecutionError as exc:
        if 'No such file' in exc.strerror:
            raise CommandExecutionError('File not found: {}'.format(path))
        raise CommandExecutionError('Unknown Error: {}'.format(exc.strerror))
    if not ret:
        return {}
    attrs_ids = ret.split('\n')
    attrs = {}
    for id_ in attrs_ids:
        attrs[id_] = read(path, id_, **{'hex': hex_})
    return attrs

def read(path, attribute, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Read the given attributes on the given file/directory\n\n    :param str path: The file to get attributes from\n\n    :param str attribute: The attribute to read\n\n    :param bool hex: Return the values with forced hexadecimal values\n\n    :return: A string containing the value of the named attribute\n    :rtype: str\n\n    :raises: CommandExecutionError on file not found, attribute not found, and\n        any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' xattr.read /path/to/file com.test.attr\n        salt '*' xattr.read /path/to/file com.test.attr hex=True\n    "
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    hex_ = kwargs.pop('hex', False)
    if kwargs:
        salt.utils.args.invalid_kwargs(kwargs)
    cmd = ['xattr', '-p']
    if hex_:
        cmd.append('-x')
    cmd.extend([attribute, path])
    try:
        ret = salt.utils.mac_utils.execute_return_result(cmd)
    except UnicodeDecodeError as exc:
        return exc.object.decode(errors='replace')
    except CommandExecutionError as exc:
        if 'No such file' in exc.strerror:
            raise CommandExecutionError('File not found: {}'.format(path))
        if 'No such xattr' in exc.strerror:
            raise CommandExecutionError('Attribute not found: {}'.format(attribute))
        raise CommandExecutionError('Unknown Error: {}'.format(exc.strerror))
    return ret

def write(path, attribute, value, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Causes the given attribute name to be assigned the given value\n\n    :param str path: The file(s) to get attributes from\n\n    :param str attribute: The attribute name to be written to the file/directory\n\n    :param str value: The value to assign to the given attribute\n\n    :param bool hex: Set the values with forced hexadecimal values\n\n    :return: True if successful, otherwise False\n    :rtype: bool\n\n    :raises: CommandExecutionError on file not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' xattr.write /path/to/file "com.test.attr" "value"\n\n    '
    kwargs = salt.utils.args.clean_kwargs(**kwargs)
    hex_ = kwargs.pop('hex', False)
    if kwargs:
        salt.utils.args.invalid_kwargs(kwargs)
    cmd = ['xattr', '-w']
    if hex_:
        cmd.append('-x')
    cmd.extend([attribute, value, path])
    try:
        salt.utils.mac_utils.execute_return_success(cmd)
    except CommandExecutionError as exc:
        if 'No such file' in exc.strerror:
            raise CommandExecutionError('File not found: {}'.format(path))
        raise CommandExecutionError('Unknown Error: {}'.format(exc.strerror))
    return read(path, attribute, **{'hex': hex_}) == value

def delete(path, attribute):
    if False:
        i = 10
        return i + 15
    '\n    Removes the given attribute from the file\n\n    :param str path: The file(s) to get attributes from\n\n    :param str attribute: The attribute name to be deleted from the\n        file/directory\n\n    :return: True if successful, otherwise False\n    :rtype: bool\n\n    :raises: CommandExecutionError on file not found, attribute not found, and\n        any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' xattr.delete /path/to/file "com.test.attr"\n    '
    cmd = 'xattr -d "{}" "{}"'.format(attribute, path)
    try:
        salt.utils.mac_utils.execute_return_success(cmd)
    except CommandExecutionError as exc:
        if 'No such file' in exc.strerror:
            raise CommandExecutionError('File not found: {}'.format(path))
        if 'No such xattr' in exc.strerror:
            raise CommandExecutionError('Attribute not found: {}'.format(attribute))
        raise CommandExecutionError('Unknown Error: {}'.format(exc.strerror))
    return attribute not in list_(path)

def clear(path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Causes the all attributes on the file/directory to be removed\n\n    :param str path: The file(s) to get attributes from\n\n    :return: True if successful, otherwise False\n\n    :raises: CommandExecutionError on file not found or any other unknown error\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' xattr.delete /path/to/file "com.test.attr"\n    '
    cmd = 'xattr -c "{}"'.format(path)
    try:
        salt.utils.mac_utils.execute_return_success(cmd)
    except CommandExecutionError as exc:
        if 'No such file' in exc.strerror:
            raise CommandExecutionError('File not found: {}'.format(path))
        raise CommandExecutionError('Unknown Error: {}'.format(exc.strerror))
    return list_(path) == {}