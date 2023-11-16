"""
StatusPage
==========

Handle requests for the StatusPage_ API_.

.. _StatusPage: https://www.statuspage.io/
.. _API: http://doers.statuspage.io/api/v1/

In the minion configuration file, the following block is required:

.. code-block:: yaml

  statuspage:
    api_key: <API_KEY>
    page_id: <PAGE_ID>

.. versionadded:: 2017.7.0
"""
import logging
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
__virtualname__ = 'statuspage'
log = logging.getLogger(__file__)
BASE_URL = 'https://api.statuspage.io'
DEFAULT_VERSION = 1
UPDATE_FORBIDDEN_FILEDS = ['id', 'created_at', 'updated_at', 'page_id']
INSERT_FORBIDDEN_FILEDS = UPDATE_FORBIDDEN_FILEDS[:]
METHOD_OK_STATUS = {'POST': 201, 'DELETE': 204}

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Return the execution module virtualname.\n    '
    if HAS_REQUESTS is False:
        return (False, 'The requests python package is not installed')
    return __virtualname__

def _default_ret():
    if False:
        while True:
            i = 10
    '\n    Default dictionary returned.\n    '
    return {'result': False, 'comment': '', 'out': None}

def _get_api_params(api_url=None, page_id=None, api_key=None, api_version=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Retrieve the API params from the config file.\n    '
    statuspage_cfg = __salt__['config.get']('statuspage')
    if not statuspage_cfg:
        statuspage_cfg = {}
    return {'api_url': api_url or statuspage_cfg.get('api_url') or BASE_URL, 'api_page_id': page_id or statuspage_cfg.get('page_id'), 'api_key': api_key or statuspage_cfg.get('api_key'), 'api_version': api_version or statuspage_cfg.get('api_version') or DEFAULT_VERSION}

def _validate_api_params(params):
    if False:
        return 10
    '\n    Validate the API params as specified in the config file.\n    '
    return isinstance(params['api_page_id'], ((str,), str)) and isinstance(params['api_key'], ((str,), str))

def _get_headers(params):
    if False:
        print('Hello World!')
    '\n    Return HTTP headers required.\n    '
    return {'Authorization': 'OAuth {oauth}'.format(oauth=params['api_key'])}

def _http_request(url, method='GET', headers=None, data=None):
    if False:
        print('Hello World!')
    '\n    Make the HTTP request and return the body as python object.\n    '
    req = requests.request(method, url, headers=headers, data=data)
    ret = _default_ret()
    ok_status = METHOD_OK_STATUS.get(method, 200)
    if req.status_code != ok_status:
        ret.update({'comment': req.json().get('error', '')})
        return ret
    ret.update({'result': True, 'out': req.json() if method != 'DELETE' else None})
    return ret

def create(endpoint='incidents', api_url=None, page_id=None, api_key=None, api_version=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Insert a new entry under a specific endpoint.\n\n    endpoint: incidents\n        Insert under this specific endpoint.\n\n    page_id\n        Page ID. Can also be specified in the config file.\n\n    api_key\n        API key. Can also be specified in the config file.\n\n    api_version: 1\n        API version. Can also be specified in the config file.\n\n    api_url\n        Custom API URL in case the user has a StatusPage service running in a custom environment.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion' statuspage.create endpoint='components' name='my component' group_id='993vgplshj12'\n\n    Example output:\n\n    .. code-block:: bash\n\n        minion:\n            ----------\n            comment:\n            out:\n                ----------\n                created_at:\n                    2017-01-05T19:35:27.135Z\n                description:\n                    None\n                group_id:\n                    993vgplshj12\n                id:\n                    mjkmtt5lhdgc\n                name:\n                    my component\n                page_id:\n                    ksdhgfyiuhaa\n                position:\n                    7\n                status:\n                    operational\n                updated_at:\n                    2017-01-05T19:35:27.135Z\n            result:\n                True\n    "
    params = _get_api_params(api_url=api_url, page_id=page_id, api_key=api_key, api_version=api_version)
    if not _validate_api_params(params):
        log.error('Invalid API params.')
        log.error(params)
        return {'result': False, 'comment': 'Invalid API params. See log for details'}
    endpoint_sg = endpoint[:-1]
    headers = _get_headers(params)
    create_url = '{base_url}/v{version}/pages/{page_id}/{endpoint}.json'.format(base_url=params['api_url'], version=params['api_version'], page_id=params['api_page_id'], endpoint=endpoint)
    change_request = {}
    for (karg, warg) in kwargs.items():
        if warg is None or karg.startswith('__') or karg in INSERT_FORBIDDEN_FILEDS:
            continue
        change_request_key = '{endpoint_sg}[{karg}]'.format(endpoint_sg=endpoint_sg, karg=karg)
        change_request[change_request_key] = warg
    return _http_request(create_url, method='POST', headers=headers, data=change_request)

def retrieve(endpoint='incidents', api_url=None, page_id=None, api_key=None, api_version=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Retrieve a specific endpoint from the Statuspage API.\n\n    endpoint: incidents\n        Request a specific endpoint.\n\n    page_id\n        Page ID. Can also be specified in the config file.\n\n    api_key\n        API key. Can also be specified in the config file.\n\n    api_version: 1\n        API version. Can also be specified in the config file.\n\n    api_url\n        Custom API URL in case the user has a StatusPage service running in a custom environment.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion' statuspage.retrieve components\n\n    Example output:\n\n    .. code-block:: bash\n\n        minion:\n            ----------\n            comment:\n            out:\n                |_\n                  ----------\n                  backfilled:\n                      False\n                  created_at:\n                      2015-01-26T20:25:02.702Z\n                  id:\n                      kh2qwjbheqdc36\n                  impact:\n                      major\n                  impact_override:\n                      None\n                  incident_updates:\n                      |_\n                        ----------\n                        affected_components:\n                            None\n                        body:\n                            We are currently investigating this issue.\n                        created_at:\n                            2015-01-26T20:25:02.849Z\n                        display_at:\n                            2015-01-26T20:25:02.849Z\n                        id:\n                            zvx7xz2z5skr\n                        incident_id:\n                            kh2qwjbheqdc36\n                        status:\n                            investigating\n                        twitter_updated_at:\n                            None\n                        updated_at:\n                            2015-01-26T20:25:02.849Z\n                        wants_twitter_update:\n                            False\n                  monitoring_at:\n                      None\n                  name:\n                      just testing some stuff\n                  page_id:\n                      ksdhgfyiuhaa\n                  postmortem_body:\n                      None\n                  postmortem_body_last_updated_at:\n                      None\n                  postmortem_ignored:\n                      False\n                  postmortem_notified_subscribers:\n                      False\n                  postmortem_notified_twitter:\n                      False\n                  postmortem_published_at:\n                      None\n                  resolved_at:\n                      None\n                  scheduled_auto_completed:\n                      False\n                  scheduled_auto_in_progress:\n                      False\n                  scheduled_for:\n                      None\n                  scheduled_remind_prior:\n                      False\n                  scheduled_reminded_at:\n                      None\n                  scheduled_until:\n                      None\n                  shortlink:\n                      http://stspg.io/voY\n                  status:\n                      investigating\n                  updated_at:\n                      2015-01-26T20:25:13.379Z\n            result:\n                True\n    "
    params = _get_api_params(api_url=api_url, page_id=page_id, api_key=api_key, api_version=api_version)
    if not _validate_api_params(params):
        log.error('Invalid API params.')
        log.error(params)
        return {'result': False, 'comment': 'Invalid API params. See log for details'}
    headers = _get_headers(params)
    retrieve_url = '{base_url}/v{version}/pages/{page_id}/{endpoint}.json'.format(base_url=params['api_url'], version=params['api_version'], page_id=params['api_page_id'], endpoint=endpoint)
    return _http_request(retrieve_url, headers=headers)

def update(endpoint='incidents', id=None, api_url=None, page_id=None, api_key=None, api_version=None, **kwargs):
    if False:
        return 10
    "\n    Update attribute(s) of a specific endpoint.\n\n    id\n        The unique ID of the endpoint entry.\n\n    endpoint: incidents\n        Endpoint name.\n\n    page_id\n        Page ID. Can also be specified in the config file.\n\n    api_key\n        API key. Can also be specified in the config file.\n\n    api_version: 1\n        API version. Can also be specified in the config file.\n\n    api_url\n        Custom API URL in case the user has a StatusPage service running in a custom environment.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion' statuspage.update id=dz959yz2nd4l status=resolved\n\n    Example output:\n\n    .. code-block:: bash\n\n        minion:\n            ----------\n            comment:\n            out:\n                ----------\n                created_at:\n                    2017-01-03T15:25:30.718Z\n                description:\n                    None\n                group_id:\n                    993vgplshj12\n                id:\n                    dz959yz2nd4l\n                name:\n                    Management Portal\n                page_id:\n                    xzwjjdw87vpf\n                position:\n                    11\n                status:\n                    resolved\n                updated_at:\n                    2017-01-05T15:34:27.676Z\n            result:\n                True\n    "
    endpoint_sg = endpoint[:-1]
    if not id:
        log.error('Invalid %s ID', endpoint_sg)
        return {'result': False, 'comment': 'Please specify a valid {endpoint} ID'.format(endpoint=endpoint_sg)}
    params = _get_api_params(api_url=api_url, page_id=page_id, api_key=api_key, api_version=api_version)
    if not _validate_api_params(params):
        log.error('Invalid API params.')
        log.error(params)
        return {'result': False, 'comment': 'Invalid API params. See log for details'}
    headers = _get_headers(params)
    update_url = '{base_url}/v{version}/pages/{page_id}/{endpoint}/{id}.json'.format(base_url=params['api_url'], version=params['api_version'], page_id=params['api_page_id'], endpoint=endpoint, id=id)
    change_request = {}
    for (karg, warg) in kwargs.items():
        if warg is None or karg.startswith('__') or karg in UPDATE_FORBIDDEN_FILEDS:
            continue
        change_request_key = '{endpoint_sg}[{karg}]'.format(endpoint_sg=endpoint_sg, karg=karg)
        change_request[change_request_key] = warg
    return _http_request(update_url, method='PATCH', headers=headers, data=change_request)

def delete(endpoint='incidents', id=None, api_url=None, page_id=None, api_key=None, api_version=None):
    if False:
        print('Hello World!')
    "\n    Remove an entry from an endpoint.\n\n    endpoint: incidents\n        Request a specific endpoint.\n\n    page_id\n        Page ID. Can also be specified in the config file.\n\n    api_key\n        API key. Can also be specified in the config file.\n\n    api_version: 1\n        API version. Can also be specified in the config file.\n\n    api_url\n        Custom API URL in case the user has a StatusPage service running in a custom environment.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt 'minion' statuspage.delete endpoint='components' id='ftgks51sfs2d'\n\n    Example output:\n\n    .. code-block:: bash\n\n        minion:\n            ----------\n            comment:\n            out:\n                None\n            result:\n                True\n    "
    params = _get_api_params(api_url=api_url, page_id=page_id, api_key=api_key, api_version=api_version)
    if not _validate_api_params(params):
        log.error('Invalid API params.')
        log.error(params)
        return {'result': False, 'comment': 'Invalid API params. See log for details'}
    endpoint_sg = endpoint[:-1]
    if not id:
        log.error('Invalid %s ID', endpoint_sg)
        return {'result': False, 'comment': 'Please specify a valid {endpoint} ID'.format(endpoint=endpoint_sg)}
    headers = _get_headers(params)
    delete_url = '{base_url}/v{version}/pages/{page_id}/{endpoint}/{id}.json'.format(base_url=params['api_url'], version=params['api_version'], page_id=params['api_page_id'], endpoint=endpoint, id=id)
    return _http_request(delete_url, method='DELETE', headers=headers)