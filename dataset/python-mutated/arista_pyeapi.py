"""
Arista pyeapi
=============

.. versionadded:: 2019.2.0

Execution module to interface the connection with Arista switches, connecting to
the remote network device using the
`pyeapi <http://pyeapi.readthedocs.io/en/master/index.html>`_ library. It is
flexible enough to execute the commands both when running under an Arista Proxy
Minion, as well as running under a Regular Minion by specifying the connection
arguments, i.e., ``device_type``, ``host``, ``username``, ``password`` etc.

:codeauthor: Mircea Ulinic <ping@mirceaulinic.net>
:maturity:   new
:depends:    pyeapi
:platform:   unix

.. note::

    To understand how to correctly enable the eAPI on your switch, please check
    https://eos.arista.com/arista-eapi-101/.

Dependencies
------------

The ``pyeapi`` Execution module requires the Python Client for eAPI (pyeapi) to
be installed: ``pip install pyeapi``.

Usage
-----

This module can equally be used via the :mod:`pyeapi <salt.proxy.arista_pyeapi>`
Proxy module or directly from an arbitrary (Proxy) Minion that is running on a
machine having access to the network device API, and the ``pyeapi`` library is
installed.

When running outside of the :mod:`pyeapi Proxy <salt.proxy.arista_pyeapi>`
(i.e., from another Proxy Minion type, or regular Minion), the pyeapi connection
arguments can be either specified from the CLI when executing the command, or
in a configuration block under the ``pyeapi`` key in the configuration opts
(i.e., (Proxy) Minion configuration file), or Pillar. The module supports these
simultaneously. These fields are the exact same supported by the ``pyeapi``
Proxy Module:

transport: ``https``
    Specifies the type of connection transport to use. Valid values for the
    connection are ``socket``, ``http_local``, ``http``, and  ``https``.

host: ``localhost``
    The IP address or DNS host name of the connection device.

username: ``admin``
    The username to pass to the device to authenticate the eAPI connection.

password
    The password to pass to the device to authenticate the eAPI connection.

port
    The TCP port of the endpoint for the eAPI connection. If this keyword is
    not specified, the default value is automatically determined by the
    transport type (``80`` for ``http``, or ``443`` for ``https``).

enablepwd
    The enable mode password if required by the destination node.

Example (when not running in a ``pyeapi`` Proxy Minion):

.. code-block:: yaml

  pyeapi:
    username: test
    password: test

In case the ``username`` and ``password`` are the same on any device you are
targeting, the block above (besides other parameters specific to your
environment you might need) should suffice to be able to execute commands from
outside a ``pyeapi`` Proxy, e.g.:

.. code-block:: bash

    salt '*' pyeapi.run_commands 'show version' 'show interfaces'
    salt '*' pyeapi.config 'ntp server 1.2.3.4'

.. note::

    Remember that the above applies only when not running in a ``pyeapi`` Proxy
    Minion. If you want to use the :mod:`pyeapi Proxy <salt.proxy.arista_pyeapi>`,
    please follow the documentation notes for a proper setup.
"""
import difflib
import logging
from salt.exceptions import CommandExecutionError
from salt.utils.args import clean_kwargs
try:
    import pyeapi
    HAS_PYEAPI = True
except ImportError:
    HAS_PYEAPI = False
__proxyenabled__ = ['*']
__virtualname__ = 'pyeapi'
log = logging.getLogger(__name__)
PYEAPI_INIT_KWARGS = ['transport', 'host', 'username', 'password', 'enablepwd', 'port', 'timeout', 'return_node']

def __virtual__():
    if False:
        return 10
    '\n    Execution module available only if pyeapi is installed.\n    '
    if not HAS_PYEAPI:
        return (False, 'The pyeapi execution module requires pyeapi library to be installed: ``pip install pyeapi``')
    return __virtualname__

def _prepare_connection(**kwargs):
    if False:
        while True:
            i = 10
    '\n    Prepare the connection with the remote network device, and clean up the key\n    value pairs, removing the args used for the connection init.\n    '
    pyeapi_kwargs = __salt__['config.get']('pyeapi', {})
    pyeapi_kwargs.update(kwargs)
    (init_kwargs, fun_kwargs) = __utils__['args.prepare_kwargs'](pyeapi_kwargs, PYEAPI_INIT_KWARGS)
    if 'transport' not in init_kwargs:
        init_kwargs['transport'] = 'https'
    conn = pyeapi.client.connect(**init_kwargs)
    node = pyeapi.client.Node(conn, enablepwd=init_kwargs.get('enablepwd'))
    return (node, fun_kwargs)

def get_connection(**kwargs):
    if False:
        while True:
            i = 10
    "\n    Return the connection object to the pyeapi Node.\n\n    .. warning::\n\n        This function returns an unserializable object, hence it is not meant\n        to be used on the CLI. This should mainly be used when invoked from\n        other modules for the low level connection with the network device.\n\n    kwargs\n        Key-value dictionary with the authentication details.\n\n    USAGE Example:\n\n    .. code-block:: python\n\n        conn = __salt__['pyeapi.get_connection'](host='router1.example.com',\n                                                 username='example',\n                                                 password='example')\n        show_ver = conn.run_commands(['show version', 'show interfaces'])\n    "
    kwargs = clean_kwargs(**kwargs)
    if 'pyeapi.conn' in __proxy__:
        return __proxy__['pyeapi.conn']()
    (conn, kwargs) = _prepare_connection(**kwargs)
    return conn

def call(method, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Invoke an arbitrary pyeapi method.\n\n    method\n        The name of the pyeapi method to invoke.\n\n    args\n        A list of arguments to send to the method invoked.\n\n    kwargs\n        Key-value dictionary to send to the method invoked.\n\n    transport: ``https``\n        Specifies the type of connection transport to use. Valid values for the\n        connection are ``socket``, ``http_local``, ``http``, and  ``https``.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    host: ``localhost``\n        The IP address or DNS host name of the connection device.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    username: ``admin``\n        The username to pass to the device to authenticate the eAPI connection.\n\n         .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    password\n        The password to pass to the device to authenticate the eAPI connection.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    port\n        The TCP port of the endpoint for the eAPI connection. If this keyword is\n        not specified, the default value is automatically determined by the\n        transport type (``80`` for ``http``, or ``443`` for ``https``).\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    enablepwd\n        The enable mode password if required by the destination node.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pyeapi.call run_commands "[\'show version\']"\n    '
    kwargs = clean_kwargs(**kwargs)
    if 'pyeapi.call' in __proxy__:
        return __proxy__['pyeapi.call'](method, *args, **kwargs)
    (conn, kwargs) = _prepare_connection(**kwargs)
    ret = getattr(conn, method)(*args, **kwargs)
    return ret

def run_commands(*commands, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Sends the commands over the transport to the device.\n\n    This function sends the commands to the device using the nodes\n    transport.  This is a lower layer function that shouldn't normally\n    need to be used, preferring instead to use ``config()`` or ``enable()``.\n\n    transport: ``https``\n        Specifies the type of connection transport to use. Valid values for the\n        connection are ``socket``, ``http_local``, ``http``, and  ``https``.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    host: ``localhost``\n        The IP address or DNS host name of the connection device.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    username: ``admin``\n        The username to pass to the device to authenticate the eAPI connection.\n\n         .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    password\n        The password to pass to the device to authenticate the eAPI connection.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    port\n        The TCP port of the endpoint for the eAPI connection. If this keyword is\n        not specified, the default value is automatically determined by the\n        transport type (``80`` for ``http``, or ``443`` for ``https``).\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    enablepwd\n        The enable mode password if required by the destination node.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pyeapi.run_commands 'show version'\n        salt '*' pyeapi.run_commands 'show version' encoding=text\n        salt '*' pyeapi.run_commands 'show version' encoding=text host=cr1.thn.lon username=example password=weak\n\n    Output example:\n\n    .. code-block:: text\n\n      veos1:\n          |_\n            ----------\n            architecture:\n                i386\n            bootupTimestamp:\n                1527541728.53\n            hardwareRevision:\n            internalBuildId:\n                63d2e89a-220d-4b8a-a9b3-0524fa8f9c5f\n            internalVersion:\n                4.18.1F-4591672.4181F\n            isIntlVersion:\n                False\n            memFree:\n                501468\n            memTotal:\n                1893316\n            modelName:\n                vEOS\n            serialNumber:\n            systemMacAddress:\n                52:54:00:3f:e6:d0\n            version:\n                4.18.1F\n    "
    encoding = kwargs.pop('encoding', 'json')
    send_enable = kwargs.pop('send_enable', True)
    output = call('run_commands', commands, encoding=encoding, send_enable=send_enable, **kwargs)
    if encoding == 'text':
        ret = []
        for res in output:
            ret.append(res['output'])
        return ret
    return output

def config(commands=None, config_file=None, template_engine='jinja', context=None, defaults=None, saltenv='base', **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Configures the node with the specified commands.\n\n    This method is used to send configuration commands to the node.  It\n    will take either a string or a list and prepend the necessary commands\n    to put the session into config mode.\n\n    Returns the diff after the configuration commands are loaded.\n\n    config_file\n        The source file with the configuration commands to be sent to the\n        device.\n\n        The file can also be a template that can be rendered using the template\n        engine of choice.\n\n        This can be specified using the absolute path to the file, or using one\n        of the following URL schemes:\n\n        - ``salt://``, to fetch the file from the Salt fileserver.\n        - ``http://`` or ``https://``\n        - ``ftp://``\n        - ``s3://``\n        - ``swift://``\n\n    commands\n        The commands to send to the node in config mode.  If the commands\n        argument is a string it will be cast to a list.\n        The list of commands will also be prepended with the necessary commands\n        to put the session in config mode.\n\n        .. note::\n\n            This argument is ignored when ``config_file`` is specified.\n\n    template_engine: ``jinja``\n        The template engine to use when rendering the source file. Default:\n        ``jinja``. To simply fetch the file without attempting to render, set\n        this argument to ``None``.\n\n    context\n        Variables to add to the template context.\n\n    defaults\n        Default values of the ``context`` dict.\n\n    transport: ``https``\n        Specifies the type of connection transport to use. Valid values for the\n        connection are ``socket``, ``http_local``, ``http``, and  ``https``.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    host: ``localhost``\n        The IP address or DNS host name of the connection device.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    username: ``admin``\n        The username to pass to the device to authenticate the eAPI connection.\n\n         .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    password\n        The password to pass to the device to authenticate the eAPI connection.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    port\n        The TCP port of the endpoint for the eAPI connection. If this keyword is\n        not specified, the default value is automatically determined by the\n        transport type (``80`` for ``http``, or ``443`` for ``https``).\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    enablepwd\n        The enable mode password if required by the destination node.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' pyeapi.config commands="[\'ntp server 1.2.3.4\', \'ntp server 5.6.7.8\']"\n        salt \'*\' pyeapi.config config_file=salt://config.txt\n        salt \'*\' pyeapi.config config_file=https://bit.ly/2LGLcDy context="{\'servers\': [\'1.2.3.4\']}"\n    '
    initial_config = get_config(as_string=True, **kwargs)
    if config_file:
        file_str = __salt__['cp.get_file_str'](config_file, saltenv=saltenv)
        if file_str is False:
            raise CommandExecutionError('Source file {} not found'.format(config_file))
        log.debug('Fetched from %s', config_file)
        log.debug(file_str)
    elif commands:
        if isinstance(commands, str):
            commands = [commands]
        file_str = '\n'.join(commands)
    if template_engine:
        file_str = __salt__['file.apply_template_on_contents'](file_str, template_engine, context, defaults, saltenv)
        log.debug('Rendered:')
        log.debug(file_str)
    commands = [line for line in file_str.splitlines() if line.strip()]
    configured = call('config', commands, **kwargs)
    current_config = get_config(as_string=True, **kwargs)
    diff = difflib.unified_diff(initial_config.splitlines(1)[4:], current_config.splitlines(1)[4:])
    return ''.join([x.replace('\r', '') for x in diff])

def get_config(config='running-config', params=None, as_string=False, **kwargs):
    if False:
        return 10
    "\n    Retrieves the config from the device.\n\n    This method will retrieve the config from the node as either a string\n    or a list object.  The config to retrieve can be specified as either\n    the startup-config or the running-config.\n\n    config: ``running-config``\n        Specifies to return either the nodes ``startup-config``\n        or ``running-config``.  The default value is the ``running-config``.\n\n    params\n        A string of keywords to append to the command for retrieving the config.\n\n    as_string: ``False``\n        Flag that determines the response.  If ``True``, then the configuration\n        is returned as a raw string.  If ``False``, then the configuration is\n        returned as a list.  The default value is ``False``.\n\n    transport: ``https``\n        Specifies the type of connection transport to use. Valid values for the\n        connection are ``socket``, ``http_local``, ``http``, and  ``https``.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    host: ``localhost``\n        The IP address or DNS host name of the connection device.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    username: ``admin``\n        The username to pass to the device to authenticate the eAPI connection.\n\n         .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    password\n        The password to pass to the device to authenticate the eAPI connection.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    port\n        The TCP port of the endpoint for the eAPI connection. If this keyword is\n        not specified, the default value is automatically determined by the\n        transport type (``80`` for ``http``, or ``443`` for ``https``).\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    enablepwd\n        The enable mode password if required by the destination node.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' pyeapi.get_config\n        salt '*' pyeapi.get_config params='section snmp-server'\n        salt '*' pyeapi.get_config config='startup-config'\n    "
    return call('get_config', config=config, params=params, as_string=as_string, **kwargs)

def section(regex, config='running-config', **kwargs):
    if False:
        while True:
            i = 10
    "\n    Return a section of the config.\n\n    regex\n        A valid regular expression used to select sections of configuration to\n        return.\n\n    config: ``running-config``\n        The configuration to return. Valid values for config are\n        ``running-config`` or ``startup-config``. The default value is\n        ``running-config``.\n\n    transport: ``https``\n        Specifies the type of connection transport to use. Valid values for the\n        connection are ``socket``, ``http_local``, ``http``, and  ``https``.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    host: ``localhost``\n        The IP address or DNS host name of the connection device.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    username: ``admin``\n        The username to pass to the device to authenticate the eAPI connection.\n\n         .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    password\n        The password to pass to the device to authenticate the eAPI connection.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    port\n        The TCP port of the endpoint for the eAPI connection. If this keyword is\n        not specified, the default value is automatically determined by the\n        transport type (``80`` for ``http``, or ``443`` for ``https``).\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    enablepwd\n        The enable mode password if required by the destination node.\n\n        .. note::\n\n            This argument does not need to be specified when running in a\n            :mod:`pyeapi <salt.proxy.arista_pyeapi>` Proxy Minion.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*'\n    "
    return call('section', regex, config=config, **kwargs)