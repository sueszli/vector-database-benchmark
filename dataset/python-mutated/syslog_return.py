"""
Return data to the host operating system's syslog facility

To use the syslog returner, append '--return syslog' to the
salt command.

.. code-block:: bash

    salt '*' test.ping --return syslog

The following fields can be set in the minion conf file::

    syslog.level (optional, Default: LOG_INFO)
    syslog.facility (optional, Default: LOG_USER)
    syslog.tag (optional, Default: salt-minion)
    syslog.options (list, optional, Default: [])

Available levels, facilities, and options can be found in the
``syslog`` docs for your python version.

.. note::

    The default tag comes from ``sys.argv[0]`` which is
    usually "salt-minion" but could be different based on
    the specific environment.

Configuration example:

.. code-block:: yaml

    syslog.level: 'LOG_ERR'
    syslog.facility: 'LOG_DAEMON'
    syslog.tag: 'mysalt'
    syslog.options:
      - LOG_PID

Of course you can also nest the options:

.. code-block:: yaml

    syslog:
      level: 'LOG_ERR'
      facility: 'LOG_DAEMON'
      tag: 'mysalt'
      options:
        - LOG_PID

Alternative configuration values can be used by
prefacing the configuration. Any values not found
in the alternative configuration will be pulled from
the default location:

.. code-block:: yaml

    alternative.syslog.level: 'LOG_WARN'
    alternative.syslog.facility: 'LOG_NEWS'

To use the alternative configuration, append
``--return_config alternative`` to the salt command.

.. versionadded:: 2015.5.0

.. code-block:: bash

    salt '*' test.ping --return syslog --return_config alternative

To override individual configuration items, append
--return_kwargs '{"key:": "value"}' to the salt command.

.. versionadded:: 2016.3.0

.. code-block:: bash

    salt '*' test.ping --return syslog --return_kwargs '{"level": "LOG_DEBUG"}'

.. note::

    Syslog server implementations may have limits on the maximum
    record size received by the client. This may lead to job
    return data being truncated in the syslog server's logs. For
    example, for rsyslog on RHEL-based systems, the default
    maximum record size is approximately 2KB (which return data
    can easily exceed). This is configurable in rsyslog.conf via
    the $MaxMessageSize config parameter. Please consult your syslog
    implmentation's documentation to determine how to adjust this limit.

"""
import logging
import salt.returners
import salt.utils.jid
import salt.utils.json
try:
    import syslog
    HAS_SYSLOG = True
except ImportError:
    HAS_SYSLOG = False
log = logging.getLogger(__name__)
__virtualname__ = 'syslog'

def _get_options(ret=None):
    if False:
        while True:
            i = 10
    '\n    Get the returner options from salt.\n    '
    defaults = {'level': 'LOG_INFO', 'facility': 'LOG_USER', 'options': []}
    attrs = {'level': 'level', 'facility': 'facility', 'tag': 'tag', 'options': 'options'}
    _options = salt.returners.get_returner_options(__virtualname__, ret, attrs, __salt__=__salt__, __opts__=__opts__, defaults=defaults)
    return _options

def _verify_options(options):
    if False:
        return 10
    '\n    Verify options and log warnings\n\n    Returns True if all options can be verified,\n    otherwise False\n    '
    bitwise_args = [('level', options['level']), ('facility', options['facility'])]
    bitwise_args.extend([('option', x) for x in options['options']])
    for (opt_name, opt) in bitwise_args:
        if not hasattr(syslog, opt):
            log.error('syslog has no attribute %s', opt)
            return False
        if not isinstance(getattr(syslog, opt), int):
            log.error('%s is not a valid syslog %s', opt, opt_name)
            return False
    if 'tag' in options:
        if not isinstance(options['tag'], str):
            log.error('tag must be a string')
            return False
        if len(options['tag']) > 32:
            log.error('tag size is limited to 32 characters')
            return False
    return True

def __virtual__():
    if False:
        while True:
            i = 10
    if not HAS_SYSLOG:
        return (False, 'Could not import syslog returner; syslog is not installed.')
    return __virtualname__

def returner(ret):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return data to the local syslog\n    '
    _options = _get_options(ret)
    if not _verify_options(_options):
        return
    level = getattr(syslog, _options['level'])
    facility = getattr(syslog, _options['facility'])
    logoption = 0
    for opt in _options['options']:
        logoption = logoption | getattr(syslog, opt)
    if 'tag' in _options:
        syslog.openlog(ident=salt.utils.stringutils.to_str(_options['tag']), logoption=logoption)
    else:
        syslog.openlog(logoption=logoption)
    syslog.syslog(facility | level, salt.utils.json.dumps(ret))
    syslog.closelog()

def prep_jid(nocache=False, passed_jid=None):
    if False:
        return 10
    '\n    Do any work necessary to prepare a JID, including sending a custom id\n    '
    return passed_jid if passed_jid is not None else salt.utils.jid.gen_jid(__opts__)