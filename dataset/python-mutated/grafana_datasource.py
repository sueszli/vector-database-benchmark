"""
Manage Grafana v2.0 data sources

.. versionadded:: 2016.3.0

.. code-block:: yaml

    grafana:
      grafana_timeout: 3
      grafana_token: qwertyuiop
      grafana_url: 'https://url.com'

.. code-block:: yaml

    Ensure influxdb data source is present:
      grafana_datasource.present:
        - name: influxdb
        - type: influxdb
        - url: http://localhost:8086
        - access: proxy
        - basic_auth: true
        - basic_auth_user: myuser
        - basic_auth_password: mypass
        - is_default: true
"""
import requests

def __virtual__():
    if False:
        return 10
    'Only load if grafana v2.0 is configured.'
    if __salt__['config.get']('grafana_version', 1) == 2:
        return True
    return (False, 'Not configured for grafana_version 2')

def present(name, type, url, access='proxy', user='', password='', database='', basic_auth=False, basic_auth_user='', basic_auth_password='', is_default=False, json_data=None, profile='grafana'):
    if False:
        return 10
    "\n    Ensure that a data source is present.\n\n    name\n        Name of the data source.\n\n    type\n        Which type of data source it is ('graphite', 'influxdb' etc.).\n\n    url\n        The URL to the data source API.\n\n    user\n        Optional - user to authenticate with the data source\n\n    password\n        Optional - password to authenticate with the data source\n\n    basic_auth\n        Optional - set to True to use HTTP basic auth to authenticate with the\n        data source.\n\n    basic_auth_user\n        Optional - HTTP basic auth username.\n\n    basic_auth_password\n        Optional - HTTP basic auth password.\n\n    is_default\n        Default: False\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    ret = {'name': name, 'result': None, 'comment': None, 'changes': {}}
    datasource = _get_datasource(profile, name)
    data = _get_json_data(name, type, url, access, user, password, database, basic_auth, basic_auth_user, basic_auth_password, is_default, json_data)
    if datasource:
        requests.put(_get_url(profile, datasource['id']), data, headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
        ret['result'] = True
        ret['changes'] = _diff(datasource, data)
        if ret['changes']['new'] or ret['changes']['old']:
            ret['comment'] = 'Data source {} updated'.format(name)
        else:
            ret['changes'] = {}
            ret['comment'] = 'Data source {} already up-to-date'.format(name)
    else:
        requests.post('{}/api/datasources'.format(profile['grafana_url']), data, headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
        ret['result'] = True
        ret['comment'] = 'New data source {} added'.format(name)
        ret['changes'] = data
    return ret

def absent(name, profile='grafana'):
    if False:
        print('Hello World!')
    '\n    Ensure that a data source is present.\n\n    name\n        Name of the data source to remove.\n    '
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    ret = {'result': None, 'comment': None, 'changes': {}}
    datasource = _get_datasource(profile, name)
    if not datasource:
        ret['result'] = True
        ret['comment'] = 'Data source {} already absent'.format(name)
        return ret
    requests.delete(_get_url(profile, datasource['id']), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    ret['result'] = True
    ret['comment'] = 'Data source {} was deleted'.format(name)
    return ret

def _get_url(profile, datasource_id):
    if False:
        print('Hello World!')
    return '{}/api/datasources/{}'.format(profile['grafana_url'], datasource_id)

def _get_datasource(profile, name):
    if False:
        for i in range(10):
            print('nop')
    response = requests.get('{}/api/datasources'.format(profile['grafana_url']), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    data = response.json()
    for datasource in data:
        if datasource['name'] == name:
            return datasource
    return None

def _get_headers(profile):
    if False:
        print('Hello World!')
    return {'Accept': 'application/json', 'Authorization': 'Bearer {}'.format(profile['grafana_token'])}

def _get_json_data(name, type, url, access='proxy', user='', password='', database='', basic_auth=False, basic_auth_user='', basic_auth_password='', is_default=False, json_data=None):
    if False:
        print('Hello World!')
    return {'name': name, 'type': type, 'url': url, 'access': access, 'user': user, 'password': password, 'database': database, 'basicAuth': basic_auth, 'basicAuthUser': basic_auth_user, 'basicAuthPassword': basic_auth_password, 'isDefault': is_default, 'jsonData': json_data}

def _diff(old, new):
    if False:
        for i in range(10):
            print('nop')
    old_keys = old.keys()
    old = old.copy()
    new = new.copy()
    for key in old_keys:
        if key == 'id' or key == 'orgId':
            del old[key]
        elif key not in new.keys():
            del old[key]
        elif old[key] == new[key]:
            del old[key]
            del new[key]
    return {'old': old, 'new': new}