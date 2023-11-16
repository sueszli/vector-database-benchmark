"""
Manage Grafana v4.0 data sources

.. versionadded:: 2017.7.0

:configuration: This state requires a configuration profile to be configured
    in the minion config, minion pillar, or master config. The module will use
    the 'grafana' key by default, if defined.

    Example configuration using basic authentication:

    .. code-block:: yaml

        grafana:
          grafana_url: http://grafana.localhost
          grafana_user: admin
          grafana_password: admin
          grafana_timeout: 3

    Example configuration using token based authentication:

    .. code-block:: yaml

        grafana:
          grafana_url: http://grafana.localhost
          grafana_token: token
          grafana_timeout: 3

The behavior of this module is to create data sources if the do not exists, and
to update data sources if the already exists.

.. code-block:: yaml

    Ensure influxdb data source is present:
      grafana4_datasource.present:
        - name: influxdb
        - type: influxdb
        - url: http://localhost:8086
        - access: proxy
        - basic_auth: true
        - basic_auth_user: myuser
        - basic_auth_password: mypass
        - is_default: true
"""
from salt.utils.dictdiffer import deep_diff

def __virtual__():
    if False:
        print('Hello World!')
    'Only load if grafana4 module is available'
    if 'grafana4.get_datasource' in __salt__:
        return True
    return (False, 'grafana4 module could not be loaded')

def present(name, type, url, access=None, user=None, password=None, database=None, basic_auth=None, basic_auth_user=None, basic_auth_password=None, tls_auth=None, json_data=None, is_default=None, with_credentials=None, type_logo_url=None, orgname=None, profile='grafana'):
    if False:
        while True:
            i = 10
    '\n    Ensure that a data source is present.\n\n    name\n        Name of the data source.\n\n    type\n        Type of the datasource (\'graphite\', \'influxdb\' etc.).\n\n    access\n        Use proxy or direct. Default: proxy\n\n    url\n        The URL to the data source API.\n\n    user\n        Optional - user to authenticate with the data source.\n\n    password\n        Optional - password to authenticate with the data source.\n\n    database\n        Optional - database to use with the data source.\n\n    basic_auth\n        Optional - set to True to use HTTP basic auth to authenticate with the\n        data source.\n\n    basic_auth_user\n        Optional - HTTP basic auth username.\n\n    basic_auth_password\n        Optional - HTTP basic auth password.\n\n    json_data\n        Optional - additional json data to post (eg. "timeInterval").\n\n    is_default\n        Optional - set data source as default.\n\n    with_credentials\n        Optional - Whether credentials such as cookies or auth headers should\n        be sent with cross-site requests.\n\n    type_logo_url\n        Optional - Logo to use for this datasource.\n\n    orgname\n        Name of the organization in which the data source should be present.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is \'grafana\'.\n    '
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    ret = {'name': name, 'result': None, 'comment': None, 'changes': {}}
    datasource = __salt__['grafana4.get_datasource'](name, orgname, profile)
    data = _get_json_data(name=name, type=type, url=url, access=access, user=user, password=password, database=database, basicAuth=basic_auth, basicAuthUser=basic_auth_user, basicAuthPassword=basic_auth_password, tlsAuth=tls_auth, jsonData=json_data, isDefault=is_default, withCredentials=with_credentials, typeLogoUrl=type_logo_url, defaults=datasource)
    if not datasource:
        if __opts__['test']:
            ret['comment'] = 'Datasource {} will be created'.format(name)
            return ret
        __salt__['grafana4.create_datasource'](profile=profile, **data)
        datasource = __salt__['grafana4.get_datasource'](name, profile=profile)
        ret['result'] = True
        ret['comment'] = 'New data source {} added'.format(name)
        ret['changes'] = data
        return ret
    for key in data:
        if key not in datasource:
            datasource[key] = None
    if data == datasource:
        ret['comment'] = 'Data source {} already up-to-date'.format(name)
        return ret
    if __opts__['test']:
        ret['comment'] = 'Datasource {} will be updated'.format(name)
        return ret
    __salt__['grafana4.update_datasource'](datasource['id'], profile=profile, **data)
    ret['result'] = True
    ret['changes'] = deep_diff(datasource, data, ignore=['id', 'orgId', 'readOnly'])
    ret['comment'] = 'Data source {} updated'.format(name)
    return ret

def absent(name, orgname=None, profile='grafana'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Ensure that a data source is present.\n\n    name\n        Name of the data source to remove.\n\n    orgname\n        Name of the organization from which the data source should be absent.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    ret = {'name': name, 'result': None, 'comment': None, 'changes': {}}
    datasource = __salt__['grafana4.get_datasource'](name, orgname, profile)
    if not datasource:
        ret['result'] = True
        ret['comment'] = 'Data source {} already absent'.format(name)
        return ret
    if __opts__['test']:
        ret['comment'] = 'Datasource {} will be deleted'.format(name)
        return ret
    __salt__['grafana4.delete_datasource'](datasource['id'], profile=profile)
    ret['result'] = True
    ret['changes'][name] = 'Absent'
    ret['comment'] = 'Data source {} was deleted'.format(name)
    return ret

def _get_json_data(defaults=None, **kwargs):
    if False:
        while True:
            i = 10
    if defaults is None:
        defaults = {}
    for (k, v) in kwargs.items():
        if v is None:
            kwargs[k] = defaults.get(k)
    return kwargs