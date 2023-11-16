"""
Take data from salt and "return" it into a carbon receiver

Add the following configuration to the minion configuration file:

.. code-block:: yaml

    carbon.host: <server ip address>
    carbon.port: 2003

Errors when trying to convert data to numbers may be ignored by setting
``carbon.skip_on_error`` to `True`:

.. code-block:: yaml

    carbon.skip_on_error: True

By default, data will be sent to carbon using the plaintext protocol. To use
the pickle protocol, set ``carbon.mode`` to ``pickle``:

.. code-block:: yaml

    carbon.mode: pickle

You can also specify the pattern used for the metric base path (except for virt modules metrics):
    carbon.metric_base_pattern: carbon.[minion_id].[module].[function]

These tokens can used :
    [module]: salt module
    [function]: salt function
    [minion_id]: minion id

Default is :
    carbon.metric_base_pattern: [module].[function].[minion_id]

Carbon settings may also be configured as:

.. code-block:: yaml

    carbon:
      host: <server IP or hostname>
      port: <carbon port>
      skip_on_error: True
      mode: (pickle|text)
      metric_base_pattern: <pattern> | [module].[function].[minion_id]

Alternative configuration values can be used by prefacing the configuration.
Any values not found in the alternative configuration will be pulled from
the default location:

.. code-block:: yaml

    alternative.carbon:
      host: <server IP or hostname>
      port: <carbon port>
      skip_on_error: True
      mode: (pickle|text)

To use the carbon returner, append '--return carbon' to the salt command.

.. code-block:: bash

    salt '*' test.ping --return carbon

To use the alternative configuration, append '--return_config alternative' to the salt command.

.. versionadded:: 2015.5.0

.. code-block:: bash

    salt '*' test.ping --return carbon --return_config alternative

To override individual configuration items, append --return_kwargs '{"key:": "value"}' to the salt command.

.. versionadded:: 2016.3.0

.. code-block:: bash

    salt '*' test.ping --return carbon --return_kwargs '{"skip_on_error": False}'

"""
import logging
import pickle
import socket
import struct
import time
from collections.abc import Mapping
from contextlib import contextmanager
import salt.returners
import salt.utils.jid
log = logging.getLogger(__name__)
__virtualname__ = 'carbon'

def __virtual__():
    if False:
        print('Hello World!')
    return __virtualname__

def _get_options(ret):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns options used for the carbon returner.\n    '
    attrs = {'host': 'host', 'port': 'port', 'skip': 'skip_on_error', 'mode': 'mode'}
    _options = salt.returners.get_returner_options(__virtualname__, ret, attrs, __salt__=__salt__, __opts__=__opts__)
    return _options

@contextmanager
def _carbon(host, port):
    if False:
        while True:
            i = 10
    '\n    Context manager to ensure the clean creation and destruction of a socket.\n\n    host\n        The IP or hostname of the carbon server\n    port\n        The port that carbon is listening on\n    '
    carbon_sock = None
    try:
        carbon_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        carbon_sock.connect((host, port))
    except OSError as err:
        log.error('Error connecting to %s:%s, %s', host, port, err)
        raise
    else:
        log.debug('Connected to carbon')
        yield carbon_sock
    finally:
        if carbon_sock is not None:
            log.debug('Destroying carbon socket')
            carbon_sock.shutdown(socket.SHUT_RDWR)
            carbon_sock.close()

def _send_picklemetrics(metrics):
    if False:
        i = 10
        return i + 15
    '\n    Format metrics for the carbon pickle protocol\n    '
    metrics = [(metric_name, (timestamp, value)) for (metric_name, value, timestamp) in metrics]
    data = pickle.dumps(metrics, -1)
    payload = struct.pack(b'!L', len(data)) + data
    return payload

def _send_textmetrics(metrics):
    if False:
        return 10
    '\n    Format metrics for the carbon plaintext protocol\n    '
    data = [' '.join(map(str, metric)) for metric in metrics] + ['']
    return '\n'.join(data)

def _walk(path, value, metrics, timestamp, skip):
    if False:
        print('Hello World!')
    "\n    Recursively include metrics from *value*.\n\n    path\n        The dot-separated path of the metric.\n    value\n        A dictionary or value from a dictionary. If a dictionary, ``_walk``\n        will be called again with the each key/value pair as a new set of\n        metrics.\n    metrics\n        The list of metrics that will be sent to carbon, formatted as::\n\n            (path, value, timestamp)\n    skip\n        Whether or not to skip metrics when there's an error casting the value\n        to a float. Defaults to `False`.\n    "
    log.trace('Carbon return walking path: %s, value: %s, metrics: %s, timestamp: %s', path, value, metrics, timestamp)
    if isinstance(value, Mapping):
        for (key, val) in value.items():
            _walk('{}.{}'.format(path, key), val, metrics, timestamp, skip)
    elif isinstance(value, list):
        for item in value:
            _walk('{}.{}'.format(path, item), item, metrics, timestamp, skip)
    else:
        try:
            val = float(value)
            metrics.append((path, val, timestamp))
        except (TypeError, ValueError):
            msg = 'Error in carbon returner, when trying to convert metric: {}, with val: {}'.format(path, value)
            if skip:
                log.debug(msg)
            else:
                log.info(msg)
                raise

def _send(saltdata, metric_base, opts):
    if False:
        print('Hello World!')
    '\n    Send the data to carbon\n    '
    host = opts.get('host')
    port = opts.get('port')
    skip = opts.get('skip')
    metric_base_pattern = opts.get('carbon.metric_base_pattern')
    mode = opts.get('mode').lower() if 'mode' in opts else 'text'
    log.debug('Carbon minion configured with host: %s:%s', host, port)
    log.debug('Using carbon protocol: %s', mode)
    if not (host and port):
        log.error('Host or port not defined')
        return
    timestamp = int(time.time())
    handler = _send_picklemetrics if mode == 'pickle' else _send_textmetrics
    metrics = []
    log.trace('Carbon returning walking data: %s', saltdata)
    _walk(metric_base, saltdata, metrics, timestamp, skip)
    data = handler(metrics)
    log.trace('Carbon inserting data: %s', data)
    with _carbon(host, port) as sock:
        total_sent_bytes = 0
        while total_sent_bytes < len(data):
            sent_bytes = sock.send(data[total_sent_bytes:])
            if sent_bytes == 0:
                log.error('Bytes sent 0, Connection reset?')
                return
            log.debug('Sent %s bytes to carbon', sent_bytes)
            total_sent_bytes += sent_bytes

def event_return(events):
    if False:
        return 10
    '\n    Return event data to remote carbon server\n\n    Provide a list of events to be stored in carbon\n    '
    opts = _get_options({})
    opts['skip'] = True
    for event in events:
        log.trace('Carbon returner received event: %s', event)
        metric_base = event['tag']
        saltdata = event['data'].get('data')
        _send(saltdata, metric_base, opts)

def returner(ret):
    if False:
        print('Hello World!')
    '\n    Return data to a remote carbon server using the text metric protocol\n\n    Each metric will look like::\n\n        [module].[function].[minion_id].[metric path [...]].[metric name]\n\n    '
    opts = _get_options(ret)
    metric_base = ret['fun']
    if not metric_base.startswith('virt.'):
        metric_base += '.' + ret['id'].replace('.', '_')
    saltdata = ret['return']
    _send(saltdata, metric_base, opts)

def prep_jid(nocache=False, passed_jid=None):
    if False:
        return 10
    '\n    Do any work necessary to prepare a JID, including sending a custom id\n    '
    return passed_jid if passed_jid is not None else salt.utils.jid.gen_jid(__opts__)