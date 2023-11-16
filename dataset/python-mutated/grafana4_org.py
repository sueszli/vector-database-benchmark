"""
Manage Grafana v4.0 orgs

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

.. code-block:: yaml

    Ensure foobar org is present:
      grafana4_org.present:
        - name: foobar
        - theme:  ""
        - home_dashboard_id: 0
        - timezone: "utc"
        - address1: ""
        - address2: ""
        - city: ""
        - zip_code: ""
        - state: ""
        - country: ""
"""
from requests.exceptions import HTTPError
import salt.utils.dictupdate as dictupdate
from salt.utils.dictdiffer import deep_diff

def __virtual__():
    if False:
        while True:
            i = 10
    'Only load if grafana4 module is available'
    if 'grafana4.get_org' in __salt__:
        return True
    return (False, 'grafana4 module could not be loaded')

def present(name, users=None, theme=None, home_dashboard_id=None, timezone=None, address1=None, address2=None, city=None, zip_code=None, address_state=None, country=None, profile='grafana'):
    if False:
        return 10
    '\n    Ensure that an organization is present.\n\n    name\n        Name of the org.\n\n    users\n        Optional - Dict of user/role associated with the org. Example:\n\n        .. code-block:: yaml\n\n            users:\n              foo: Viewer\n              bar: Editor\n\n    theme\n        Optional - Selected theme for the org.\n\n    home_dashboard_id\n        Optional - Home dashboard for the org.\n\n    timezone\n        Optional - Timezone for the org (one of: "browser", "utc", or "").\n\n    address1\n        Optional - address1 of the org.\n\n    address2\n        Optional - address2 of the org.\n\n    city\n        Optional - city of the org.\n\n    zip_code\n        Optional - zip_code of the org.\n\n    address_state\n        Optional - state of the org.\n\n    country\n        Optional - country of the org.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is \'grafana\'.\n    '
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    ret = {'name': name, 'result': None, 'comment': None, 'changes': {}}
    create = False
    try:
        org = __salt__['grafana4.get_org'](name, profile)
    except HTTPError as e:
        if e.response.status_code == 404:
            create = True
        else:
            raise
    if create:
        if __opts__['test']:
            ret['comment'] = 'Org {} will be created'.format(name)
            return ret
        __salt__['grafana4.create_org'](profile=profile, name=name)
        org = __salt__['grafana4.get_org'](name, profile)
        ret['changes'] = org
        ret['comment'] = 'New org {} added'.format(name)
    data = _get_json_data(address1=address1, address2=address2, city=city, zipCode=zip_code, state=address_state, country=country, defaults=org['address'])
    if data != org['address']:
        if __opts__['test']:
            ret['comment'] = 'Org {} address will be updated'.format(name)
            return ret
        __salt__['grafana4.update_org_address'](name, profile=profile, **data)
        if create:
            dictupdate.update(ret['changes']['address'], data)
        else:
            dictupdate.update(ret['changes'], deep_diff(org['address'], data))
    prefs = __salt__['grafana4.get_org_prefs'](name, profile=profile)
    data = _get_json_data(theme=theme, homeDashboardId=home_dashboard_id, timezone=timezone, defaults=prefs)
    if data != prefs:
        if __opts__['test']:
            ret['comment'] = 'Org {} prefs will be updated'.format(name)
            return ret
        __salt__['grafana4.update_org_prefs'](name, profile=profile, **data)
        if create:
            dictupdate.update(ret['changes'], data)
        else:
            dictupdate.update(ret['changes'], deep_diff(prefs, data))
    if users:
        db_users = {}
        for item in __salt__['grafana4.get_org_users'](name, profile=profile):
            db_users[item['login']] = {'userId': item['userId'], 'role': item['role']}
        for (username, role) in users.items():
            if username in db_users:
                if role is False:
                    if __opts__['test']:
                        ret['comment'] = 'Org {} user {} will be deleted'.format(name, username)
                        return ret
                    __salt__['grafana4.delete_org_user'](db_users[username]['userId'], profile=profile)
                elif role != db_users[username]['role']:
                    if __opts__['test']:
                        ret['comment'] = 'Org {} user {} role will be updated'.format(name, username)
                        return ret
                    __salt__['grafana4.update_org_user'](db_users[username]['userId'], loginOrEmail=username, role=role, profile=profile)
            elif role:
                if __opts__['test']:
                    ret['comment'] = 'Org {} user {} will be created'.format(name, username)
                    return ret
                __salt__['grafana4.create_org_user'](loginOrEmail=username, role=role, profile=profile)
        new_db_users = {}
        for item in __salt__['grafana4.get_org_users'](name, profile=profile):
            new_db_users[item['login']] = {'userId': item['userId'], 'role': item['role']}
        if create:
            dictupdate.update(ret['changes'], new_db_users)
        else:
            dictupdate.update(ret['changes'], deep_diff(db_users, new_db_users))
    ret['result'] = True
    if not create:
        if ret['changes']:
            ret['comment'] = 'Org {} updated'.format(name)
        else:
            ret['changes'] = {}
            ret['comment'] = 'Org {} already up-to-date'.format(name)
    return ret

def absent(name, profile='grafana'):
    if False:
        i = 10
        return i + 15
    "\n    Ensure that a org is present.\n\n    name\n        Name of the org to remove.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    ret = {'name': name, 'result': None, 'comment': None, 'changes': {}}
    org = __salt__['grafana4.get_org'](name, profile)
    if not org:
        ret['result'] = True
        ret['comment'] = 'Org {} already absent'.format(name)
        return ret
    if __opts__['test']:
        ret['comment'] = 'Org {} will be deleted'.format(name)
        return ret
    __salt__['grafana4.delete_org'](org['id'], profile=profile)
    ret['result'] = True
    ret['changes'][name] = 'Absent'
    ret['comment'] = 'Org {} was deleted'.format(name)
    return ret

def _get_json_data(defaults=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if defaults is None:
        defaults = {}
    for (k, v) in kwargs.items():
        if v is None:
            kwargs[k] = defaults.get(k)
    return kwargs