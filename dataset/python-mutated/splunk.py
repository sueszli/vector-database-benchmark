"""

Send json response data to Splunk via the HTTP Event Collector
Requires the following config values to be specified in config or pillar:

.. code-block:: yaml

    splunk_http_forwarder:
      token: <splunk_http_forwarder_token>
      indexer: <hostname/IP of Splunk indexer>
      sourcetype: <Destination sourcetype for data>
      index: <Destination index for data>
      verify_ssl: true

Run a test by using ``salt-call test.ping --return splunk``

Written by Scott Pack (github.com/scottjpack)

"""
import logging
import socket
import time
import requests
import salt.utils.json
_max_content_bytes = 100000
http_event_collector_debug = False
log = logging.getLogger(__name__)
__virtualname__ = 'splunk'

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return virtual name of the module.\n    :return: The virtual name of the module.\n    '
    return __virtualname__

def returner(ret):
    if False:
        while True:
            i = 10
    '\n    Send a message to Splunk via the HTTP Event Collector.\n    Requires the Splunk HTTP Event Collector running on port 8088.\n    This is available on Splunk Enterprise version 6.3 or higher.\n\n    '
    opts = _get_options()
    log.info('Options: %s', salt.utils.json.dumps(opts))
    http_collector = _create_http_event_collector(opts)
    payload = _prepare_splunk_payload(ret, opts)
    http_collector.sendEvent(payload)
    return True

def event_return(events):
    if False:
        print('Hello World!')
    '\n    Return events to Splunk via the HTTP Event Collector.\n    Requires the Splunk HTTP Event Collector running on port 8088.\n    This is available on Splunk Enterprise version 6.3 or higher.\n    '
    opts = _get_options()
    log.info('Options: %s', salt.utils.json.dumps(opts))
    http_collector = _create_http_event_collector(opts)
    for event in events:
        payload = _prepare_splunk_payload(event, opts)
        http_collector.sendEvent(payload)
    return True

def _get_options():
    if False:
        for i in range(10):
            print('nop')
    try:
        token = __salt__['config.get']('splunk_http_forwarder:token')
        indexer = __salt__['config.get']('splunk_http_forwarder:indexer')
        sourcetype = __salt__['config.get']('splunk_http_forwarder:sourcetype')
        index = __salt__['config.get']('splunk_http_forwarder:index')
        verify_ssl = __salt__['config.get']('splunk_http_forwarder:verify_ssl', default=True)
    except Exception:
        log.error('Splunk HTTP Forwarder parameters not present in config.')
        return None
    splunk_opts = {'token': token, 'indexer': indexer, 'sourcetype': sourcetype, 'index': index, 'verify_ssl': verify_ssl}
    return splunk_opts

def _create_http_event_collector(opts):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prepare a connection to the Splunk HTTP event collector.\n\n    '
    http_event_collector_key = opts['token']
    http_event_collector_host = opts['indexer']
    http_event_collector_verify_ssl = opts['verify_ssl']
    return http_event_collector(http_event_collector_key, http_event_collector_host, verify_ssl=http_event_collector_verify_ssl)

def _prepare_splunk_payload(event, opts):
    if False:
        print('Hello World!')
    '\n    Prepare a payload for submission to the Splunk HTTP event collector.\n\n    '
    opts = _get_options()
    payload = {}
    payload.update({'index': opts['index']})
    payload.update({'sourcetype': opts['sourcetype']})
    payload.update({'event': event})
    log.info('Payload: %s', salt.utils.json.dumps(payload))
    return payload

class http_event_collector:

    def __init__(self, token, http_event_server, host='', http_event_port='8088', http_event_server_ssl=True, max_bytes=_max_content_bytes, verify_ssl=True):
        if False:
            for i in range(10):
                print('nop')
        self.token = token
        self.batchEvents = []
        self.maxByteLength = max_bytes
        self.currentByteLength = 0
        self.verify_ssl = verify_ssl
        if host:
            self.host = host
        else:
            self.host = socket.gethostname()
        if http_event_server_ssl:
            buildURI = ['https://']
        else:
            buildURI = ['http://']
        for i in [http_event_server, ':', http_event_port, '/services/collector/event']:
            buildURI.append(i)
        self.server_uri = ''.join(buildURI)
        if http_event_collector_debug:
            log.debug(self.token)
            log.debug(self.server_uri)

    def sendEvent(self, payload, eventtime=''):
        if False:
            print('Hello World!')
        headers = {'Authorization': 'Splunk ' + self.token}
        if not eventtime:
            eventtime = str(int(time.time()))
        if 'host' not in payload:
            payload.update({'host': self.host})
        data = {'time': eventtime}
        data.update(payload)
        r = requests.post(self.server_uri, data=salt.utils.json.dumps(data), headers=headers, verify=self.verify_ssl)
        if http_event_collector_debug:
            log.debug(r.text)
            log.debug(data)