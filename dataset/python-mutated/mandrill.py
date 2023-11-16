"""
Mandrill
========

Send out emails using the Mandrill_ API_.

.. _Mandrill: https://mandrillapp.com
.. _API: https://mandrillapp.com/api/docs/

In the minion configuration file, the following block is required:

.. code-block:: yaml

  mandrill:
    key: <API_KEY>

.. versionadded:: 2018.3.0
"""
import logging
import salt.utils.json
import salt.utils.versions
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
__virtualname__ = 'mandrill'
log = logging.getLogger(__file__)
BASE_URL = 'https://mandrillapp.com/api'
DEFAULT_VERSION = 1

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Return the execution module virtualname.\n    '
    if HAS_REQUESTS is False:
        return (False, 'The requests python package is required for the mandrill execution module')
    return __virtualname__

def _default_ret():
    if False:
        print('Hello World!')
    '\n    Default dictionary returned.\n    '
    return {'result': False, 'comment': '', 'out': None}

def _get_api_params(api_url=None, api_version=None, api_key=None):
    if False:
        return 10
    '\n    Retrieve the API params from the config file.\n    '
    mandrill_cfg = __salt__['config.merge']('mandrill')
    if not mandrill_cfg:
        mandrill_cfg = {}
    return {'api_url': api_url or mandrill_cfg.get('api_url') or BASE_URL, 'api_key': api_key or mandrill_cfg.get('key'), 'api_version': api_version or mandrill_cfg.get('api_version') or DEFAULT_VERSION}

def _get_url(method, api_url, api_version):
    if False:
        while True:
            i = 10
    '\n    Build the API URL.\n    '
    return '{url}/{version}/{method}.json'.format(url=api_url, version=float(api_version), method=method)

def _get_headers():
    if False:
        print('Hello World!')
    '\n    Return HTTP headers required for the Mandrill API.\n    '
    return {'content-type': 'application/json', 'user-agent': 'Mandrill-Python/1.0.57'}

def _http_request(url, headers=None, data=None):
    if False:
        print('Hello World!')
    '\n    Make the HTTP request and return the body as python object.\n    '
    if not headers:
        headers = _get_headers()
    session = requests.session()
    log.debug('Querying %s', url)
    req = session.post(url, headers=headers, data=salt.utils.json.dumps(data))
    req_body = req.json()
    ret = _default_ret()
    log.debug('Status code: %d', req.status_code)
    log.debug('Response body:')
    log.debug(req_body)
    if req.status_code != 200:
        if req.status_code == 500:
            ret['comment'] = req_body.pop('message', '')
            ret['out'] = req_body
            return ret
        ret.update({'comment': req_body.get('error', '')})
        return ret
    ret.update({'result': True, 'out': req.json()})
    return ret

def send(message, asynchronous=False, ip_pool=None, send_at=None, api_url=None, api_version=None, api_key=None):
    if False:
        while True:
            i = 10
    '\n    Send out the email using the details from the ``message`` argument.\n\n    message\n        The information on the message to send. This argument must be\n        sent as dictionary with at fields as specified in the Mandrill API\n        documentation.\n\n    asynchronous: ``False``\n        Enable a background sending mode that is optimized for bulk sending.\n        In asynchronous mode, messages/send will immediately return a status of\n        "queued" for every recipient. To handle rejections when sending in asynchronous\n        mode, set up a webhook for the \'reject\' event. Defaults to false for\n        messages with no more than 10 recipients; messages with more than 10\n        recipients are always sent asynchronously, regardless of the value of\n        asynchronous.\n\n    ip_pool\n        The name of the dedicated ip pool that should be used to send the\n        message. If you do not have any dedicated IPs, this parameter has no\n        effect. If you specify a pool that does not exist, your default pool\n        will be used instead.\n\n    send_at\n        When this message should be sent as a UTC timestamp in\n        ``YYYY-MM-DD HH:MM:SS`` format. If you specify a time in the past,\n        the message will be sent immediately. An additional fee applies for\n        scheduled email, and this feature is only available to accounts with a\n        positive balance.\n\n    .. note::\n        Fur further details please consult the `API documentation <https://mandrillapp.com/api/docs/messages.dart.html>`_.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' mandrill.send message="{\'subject\': \'Hi\', \'from_email\': \'test@example.com\', \'to\': [{\'email\': \'recv@example.com\', \'type\': \'to\'}]}"\n\n    ``message`` structure example (as YAML for readability):\n\n    .. code-block:: yaml\n\n        message:\n            text: |\n                This is the body of the email.\n                This is the second line.\n            subject: Email subject\n            from_name: Test At Example Dot Com\n            from_email: test@example.com\n            to:\n              - email: recv@example.com\n                type: to\n                name: Recv At Example Dot Com\n              - email: cc@example.com\n                type: cc\n                name: CC At Example Dot Com\n            important: true\n            track_clicks: true\n            track_opens: true\n            attachments:\n              - type: text/x-yaml\n                name: yaml_file.yml\n                content: aV9hbV9zdXBlcl9jdXJpb3VzOiB0cnVl\n\n    Output example:\n\n    .. code-block:: bash\n\n        minion:\n            ----------\n            comment:\n            out:\n                |_\n                  ----------\n                  _id:\n                      c4353540a3c123eca112bbdd704ab6\n                  email:\n                      recv@example.com\n                  reject_reason:\n                      None\n                  status:\n                      sent\n            result:\n                True\n    '
    params = _get_api_params(api_url=api_url, api_version=api_version, api_key=api_key)
    url = _get_url('messages/send', api_url=params['api_url'], api_version=params['api_version'])
    data = {'key': params['api_key'], 'message': message, 'async': asynchronous, 'ip_pool': ip_pool, 'send_at': send_at}
    return _http_request(url, data=data)