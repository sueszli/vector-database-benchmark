"""
Module providing a simple management interface to a marathon cluster.

Currently this only works when run through a proxy minion.

.. versionadded:: 2015.8.2
"""
import logging
import salt.utils.http
import salt.utils.json
import salt.utils.platform
from salt.exceptions import get_error_message
__proxyenabled__ = ['marathon']
log = logging.getLogger(__file__)

def __virtual__():
    if False:
        return 10
    if salt.utils.platform.is_proxy() and 'proxy' in __opts__:
        return True
    return (False, 'The marathon execution module cannot be loaded: this only works on proxy minions.')

def _base_url():
    if False:
        print('Hello World!')
    '\n    Return the proxy configured base url.\n    '
    base_url = 'http://locahost:8080'
    if 'proxy' in __opts__:
        base_url = __opts__['proxy'].get('base_url', base_url)
    return base_url

def _app_id(app_id):
    if False:
        while True:
            i = 10
    '\n    Make sure the app_id is in the correct format.\n    '
    if app_id[0] != '/':
        app_id = '/{}'.format(app_id)
    return app_id

def apps():
    if False:
        return 10
    '\n    Return a list of the currently installed app ids.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt marathon-minion-id marathon.apps\n    '
    response = salt.utils.http.query('{}/v2/apps'.format(_base_url()), decode_type='json', decode=True)
    return {'apps': [app['id'] for app in response['dict']['apps']]}

def has_app(id):
    if False:
        while True:
            i = 10
    '\n    Return whether the given app id is currently configured.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt marathon-minion-id marathon.has_app my-app\n    '
    return _app_id(id) in apps()['apps']

def app(id):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the current server configuration for the specified app.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt marathon-minion-id marathon.app my-app\n    '
    response = salt.utils.http.query('{}/v2/apps/{}'.format(_base_url(), id), decode_type='json', decode=True)
    return response['dict']

def update_app(id, config):
    if False:
        return 10
    "\n    Update the specified app with the given configuration.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt marathon-minion-id marathon.update_app my-app '<config yaml>'\n    "
    if 'id' not in config:
        config['id'] = id
    config.pop('version', None)
    config.pop('fetch', None)
    data = salt.utils.json.dumps(config)
    try:
        response = salt.utils.http.query('{}/v2/apps/{}?force=true'.format(_base_url(), id), method='PUT', decode_type='json', decode=True, data=data, header_dict={'Content-Type': 'application/json', 'Accept': 'application/json'})
        log.debug('update response: %s', response)
        return response['dict']
    except Exception as ex:
        log.error('unable to update marathon app: %s', get_error_message(ex))
        return {'exception': {'message': get_error_message(ex)}}

def rm_app(id):
    if False:
        i = 10
        return i + 15
    '\n    Remove the specified app from the server.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt marathon-minion-id marathon.rm_app my-app\n    '
    response = salt.utils.http.query('{}/v2/apps/{}'.format(_base_url(), id), method='DELETE', decode_type='json', decode=True)
    return response['dict']

def info():
    if False:
        print('Hello World!')
    '\n    Return configuration and status information about the marathon instance.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt marathon-minion-id marathon.info\n    '
    response = salt.utils.http.query('{}/v2/info'.format(_base_url()), decode_type='json', decode=True)
    return response['dict']

def restart_app(id, restart=False, force=True):
    if False:
        i = 10
        return i + 15
    '\n    Restart the current server configuration for the specified app.\n\n    :param restart: Restart the app\n    :param force: Override the current deployment\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt marathon-minion-id marathon.restart_app my-app\n\n    By default, this will only check if the app exists in marathon. It does\n    not check if there are any tasks associated with it or if the app is suspended.\n\n    .. code-block:: bash\n\n        salt marathon-minion-id marathon.restart_app my-app true true\n\n    The restart option needs to be set to True to actually issue a rolling\n    restart to marathon.\n\n    The force option tells marathon to ignore the current app deployment if\n    there is one.\n    '
    ret = {'restarted': None}
    if not restart:
        ret['restarted'] = False
        return ret
    try:
        response = salt.utils.http.query('{}/v2/apps/{}/restart?force={}'.format(_base_url(), _app_id(id), force), method='POST', decode_type='json', decode=True, header_dict={'Content-Type': 'application/json', 'Accept': 'application/json'})
        log.debug('restart response: %s', response)
        ret['restarted'] = True
        ret.update(response['dict'])
        return ret
    except Exception as ex:
        log.error('unable to restart marathon app: %s', ex.message)
        return {'exception': {'message': ex.message}}