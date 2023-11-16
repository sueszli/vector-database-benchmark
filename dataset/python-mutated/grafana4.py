"""
Module for working with the Grafana v4 API

.. versionadded:: 2017.7.0

:depends: requests

:configuration: This module requires a configuration profile to be configured
    in the minion config, minion pillar, or master config.
    The module will use the 'grafana' key by default, if defined.

    For example:

    .. code-block:: yaml

        grafana:
            grafana_url: http://grafana.localhost
            grafana_user: admin
            grafana_password: admin
            grafana_timeout: 3
"""
try:
    import requests
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
__virtualname__ = 'grafana4'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Only load if requests is installed\n    '
    if HAS_LIBS:
        return __virtualname__
    else:
        return (False, 'The "{}" module could not be loaded: "requests" is not installed.'.format(__virtualname__))

def _get_headers(profile):
    if False:
        return 10
    headers = {'Content-type': 'application/json'}
    if profile.get('grafana_token', False):
        headers['Authorization'] = 'Bearer {}'.format(profile['grafana_token'])
    return headers

def _get_auth(profile):
    if False:
        return 10
    if profile.get('grafana_token', False):
        return None
    return requests.auth.HTTPBasicAuth(profile['grafana_user'], profile['grafana_password'])

def get_users(profile='grafana'):
    if False:
        print('Hello World!')
    "\n    List all users.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.get_users\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.get('{}/api/users'.format(profile['grafana_url']), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def get_user(login, profile='grafana'):
    if False:
        print('Hello World!')
    "\n    Show a single user.\n\n    login\n        Login of the user.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.get_user <login>\n    "
    data = get_users(profile)
    for user in data:
        if user['login'] == login:
            return user
    return None

def get_user_data(userid, profile='grafana'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get user data.\n\n    userid\n        Id of the user.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.get_user_data <user_id>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.get('{}/api/users/{}'.format(profile['grafana_url'], userid), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def create_user(profile='grafana', **kwargs):
    if False:
        print('Hello World!')
    "\n    Create a new user.\n\n    login\n        Login of the new user.\n\n    password\n        Password of the new user.\n\n    email\n        Email of the new user.\n\n    name\n        Optional - Full name of the new user.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.create_user login=<login> password=<password> email=<email>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.post('{}/api/admin/users'.format(profile['grafana_url']), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def update_user(userid, profile='grafana', **kwargs):
    if False:
        print('Hello World!')
    "\n    Update an existing user.\n\n    userid\n        Id of the user.\n\n    login\n        Optional - Login of the user.\n\n    email\n        Optional - Email of the user.\n\n    name\n        Optional - Full name of the user.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.update_user <user_id> login=<login> email=<email>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.put('{}/api/users/{}'.format(profile['grafana_url'], userid), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def update_user_password(userid, profile='grafana', **kwargs):
    if False:
        return 10
    "\n    Update a user password.\n\n    userid\n        Id of the user.\n\n    password\n        New password of the user.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.update_user_password <user_id> password=<password>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.put('{}/api/admin/users/{}/password'.format(profile['grafana_url'], userid), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def update_user_permissions(userid, profile='grafana', **kwargs):
    if False:
        print('Hello World!')
    "\n    Update a user password.\n\n    userid\n        Id of the user.\n\n    isGrafanaAdmin\n        Whether user is a Grafana admin.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.update_user_permissions <user_id> isGrafanaAdmin=<true|false>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.put('{}/api/admin/users/{}/permissions'.format(profile['grafana_url'], userid), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def delete_user(userid, profile='grafana'):
    if False:
        i = 10
        return i + 15
    "\n    Delete a user.\n\n    userid\n        Id of the user.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.delete_user <user_id>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.delete('{}/api/admin/users/{}'.format(profile['grafana_url'], userid), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def get_user_orgs(userid, profile='grafana'):
    if False:
        print('Hello World!')
    "\n    Get the list of organisations a user belong to.\n\n    userid\n        Id of the user.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.get_user_orgs <user_id>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.get('{}/api/users/{}/orgs'.format(profile['grafana_url'], userid), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def delete_user_org(userid, orgid, profile='grafana'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove a user from an organization.\n\n    userid\n        Id of the user.\n\n    orgid\n        Id of the organization.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.delete_user_org <user_id> <org_id>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.delete('{}/api/orgs/{}/users/{}'.format(profile['grafana_url'], orgid, userid), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def get_orgs(profile='grafana'):
    if False:
        while True:
            i = 10
    "\n    List all organizations.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.get_orgs\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.get('{}/api/orgs'.format(profile['grafana_url']), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def get_org(name, profile='grafana'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Show a single organization.\n\n    name\n        Name of the organization.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.get_org <name>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.get('{}/api/orgs/name/{}'.format(profile['grafana_url'], name), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def switch_org(orgname, profile='grafana'):
    if False:
        i = 10
        return i + 15
    "\n    Switch the current organization.\n\n    name\n        Name of the organization to switch to.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.switch_org <name>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    org = get_org(orgname, profile)
    response = requests.post('{}/api/user/using/{}'.format(profile['grafana_url'], org['id']), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return org

def get_org_users(orgname=None, profile='grafana'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the list of users that belong to the organization.\n\n    orgname\n        Name of the organization.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.get_org_users <orgname>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.get('{}/api/org/users'.format(profile['grafana_url']), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def create_org_user(orgname=None, profile='grafana', **kwargs):
    if False:
        return 10
    "\n    Add user to the organization.\n\n    loginOrEmail\n        Login or email of the user.\n\n    role\n        Role of the user for this organization. Should be one of:\n            - Admin\n            - Editor\n            - Read Only Editor\n            - Viewer\n\n    orgname\n        Name of the organization in which users are added.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.create_org_user <orgname> loginOrEmail=<loginOrEmail> role=<role>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.post('{}/api/org/users'.format(profile['grafana_url']), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def update_org_user(userid, orgname=None, profile='grafana', **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Update user role in the organization.\n\n    userid\n        Id of the user.\n\n    loginOrEmail\n        Login or email of the user.\n\n    role\n        Role of the user for this organization. Should be one of:\n            - Admin\n            - Editor\n            - Read Only Editor\n            - Viewer\n\n    orgname\n        Name of the organization in which users are updated.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.update_org_user <user_id> <orgname> loginOrEmail=<loginOrEmail> role=<role>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.patch('{}/api/org/users/{}'.format(profile['grafana_url'], userid), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def delete_org_user(userid, orgname=None, profile='grafana'):
    if False:
        print('Hello World!')
    "\n    Remove user from the organization.\n\n    userid\n        Id of the user.\n\n    orgname\n        Name of the organization in which users are updated.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.delete_org_user <user_id> <orgname>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.delete('{}/api/org/users/{}'.format(profile['grafana_url'], userid), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def get_org_address(orgname=None, profile='grafana'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get the organization address.\n\n    orgname\n        Name of the organization in which users are updated.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.get_org_address <orgname>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.get('{}/api/org/address'.format(profile['grafana_url']), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def update_org_address(orgname=None, profile='grafana', **kwargs):
    if False:
        return 10
    "\n    Update the organization address.\n\n    orgname\n        Name of the organization in which users are updated.\n\n    address1\n        Optional - address1 of the org.\n\n    address2\n        Optional - address2 of the org.\n\n    city\n        Optional - city of the org.\n\n    zip_code\n        Optional - zip_code of the org.\n\n    state\n        Optional - state of the org.\n\n    country\n        Optional - country of the org.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.update_org_address <orgname> country=<country>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.put('{}/api/org/address'.format(profile['grafana_url']), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def get_org_prefs(orgname=None, profile='grafana'):
    if False:
        while True:
            i = 10
    "\n    Get the organization preferences.\n\n    orgname\n        Name of the organization in which users are updated.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.get_org_prefs <orgname>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.get('{}/api/org/preferences'.format(profile['grafana_url']), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def update_org_prefs(orgname=None, profile='grafana', **kwargs):
    if False:
        return 10
    '\n    Update the organization preferences.\n\n    orgname\n        Name of the organization in which users are updated.\n\n    theme\n        Selected theme for the org.\n\n    homeDashboardId\n        Home dashboard for the org.\n\n    timezone\n        Timezone for the org (one of: "browser", "utc", or "").\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is \'grafana\'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' grafana4.update_org_prefs <orgname> theme=<theme> timezone=<timezone>\n    '
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.put('{}/api/org/preferences'.format(profile['grafana_url']), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def create_org(profile='grafana', **kwargs):
    if False:
        return 10
    "\n    Create a new organization.\n\n    name\n        Name of the organization.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.create_org <name>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.post('{}/api/orgs'.format(profile['grafana_url']), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def update_org(orgid, profile='grafana', **kwargs):
    if False:
        print('Hello World!')
    "\n    Update an existing organization.\n\n    orgid\n        Id of the organization.\n\n    name\n        New name of the organization.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.update_org <org_id> name=<name>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.put('{}/api/orgs/{}'.format(profile['grafana_url'], orgid), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def delete_org(orgid, profile='grafana'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete an organization.\n\n    orgid\n        Id of the organization.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.delete_org <org_id>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.delete('{}/api/orgs/{}'.format(profile['grafana_url'], orgid), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def get_datasources(orgname=None, profile='grafana'):
    if False:
        print('Hello World!')
    "\n    List all datasources in an organisation.\n\n    orgname\n        Name of the organization.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.get_datasources <orgname>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.get('{}/api/datasources'.format(profile['grafana_url']), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def get_datasource(name, orgname=None, profile='grafana'):
    if False:
        while True:
            i = 10
    "\n    Show a single datasource in an organisation.\n\n    name\n        Name of the datasource.\n\n    orgname\n        Name of the organization.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.get_datasource <name> <orgname>\n    "
    data = get_datasources(orgname=orgname, profile=profile)
    for datasource in data:
        if datasource['name'] == name:
            return datasource
    return None

def create_datasource(orgname=None, profile='grafana', **kwargs):
    if False:
        print('Hello World!')
    '\n    Create a new datasource in an organisation.\n\n    name\n        Name of the data source.\n\n    type\n        Type of the datasource (\'graphite\', \'influxdb\' etc.).\n\n    access\n        Use proxy or direct.\n\n    url\n        The URL to the data source API.\n\n    user\n        Optional - user to authenticate with the data source.\n\n    password\n        Optional - password to authenticate with the data source.\n\n    database\n        Optional - database to use with the data source.\n\n    basicAuth\n        Optional - set to True to use HTTP basic auth to authenticate with the\n        data source.\n\n    basicAuthUser\n        Optional - HTTP basic auth username.\n\n    basicAuthPassword\n        Optional - HTTP basic auth password.\n\n    jsonData\n        Optional - additional json data to post (eg. "timeInterval").\n\n    isDefault\n        Optional - set data source as default.\n\n    withCredentials\n        Optional - Whether credentials such as cookies or auth headers should\n        be sent with cross-site requests.\n\n    typeLogoUrl\n        Optional - Logo to use for this datasource.\n\n    orgname\n        Name of the organization in which the data source should be created.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is \'grafana\'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' grafana4.create_datasource\n\n    '
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.post('{}/api/datasources'.format(profile['grafana_url']), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def update_datasource(datasourceid, orgname=None, profile='grafana', **kwargs):
    if False:
        return 10
    '\n    Update a datasource.\n\n    datasourceid\n        Id of the datasource.\n\n    name\n        Name of the data source.\n\n    type\n        Type of the datasource (\'graphite\', \'influxdb\' etc.).\n\n    access\n        Use proxy or direct.\n\n    url\n        The URL to the data source API.\n\n    user\n        Optional - user to authenticate with the data source.\n\n    password\n        Optional - password to authenticate with the data source.\n\n    database\n        Optional - database to use with the data source.\n\n    basicAuth\n        Optional - set to True to use HTTP basic auth to authenticate with the\n        data source.\n\n    basicAuthUser\n        Optional - HTTP basic auth username.\n\n    basicAuthPassword\n        Optional - HTTP basic auth password.\n\n    jsonData\n        Optional - additional json data to post (eg. "timeInterval").\n\n    isDefault\n        Optional - set data source as default.\n\n    withCredentials\n        Optional - Whether credentials such as cookies or auth headers should\n        be sent with cross-site requests.\n\n    typeLogoUrl\n        Optional - Logo to use for this datasource.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is \'grafana\'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' grafana4.update_datasource <datasourceid>\n\n    '
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.put('{}/api/datasources/{}'.format(profile['grafana_url'], datasourceid), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return {}

def delete_datasource(datasourceid, orgname=None, profile='grafana'):
    if False:
        return 10
    "\n    Delete a datasource.\n\n    datasourceid\n        Id of the datasource.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.delete_datasource <datasource_id>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    response = requests.delete('{}/api/datasources/{}'.format(profile['grafana_url'], datasourceid), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def get_dashboard(slug, orgname=None, profile='grafana'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get a dashboard.\n\n    slug\n        Slug (name) of the dashboard.\n\n    orgname\n        Name of the organization.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.get_dashboard <slug>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.get('{}/api/dashboards/db/{}'.format(profile['grafana_url'], slug), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    data = response.json()
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        response.raise_for_status()
    return data.get('dashboard')

def delete_dashboard(slug, orgname=None, profile='grafana'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Delete a dashboard.\n\n    slug\n        Slug (name) of the dashboard.\n\n    orgname\n        Name of the organization.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.delete_dashboard <slug>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.delete('{}/api/dashboards/db/{}'.format(profile['grafana_url'], slug), auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def create_update_dashboard(orgname=None, profile='grafana', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create or update a dashboard.\n\n    dashboard\n        A dict that defines the dashboard to create/update.\n\n    overwrite\n        Whether the dashboard should be overwritten if already existing.\n\n    orgname\n        Name of the organization.\n\n    profile\n        Configuration profile used to connect to the Grafana instance.\n        Default is 'grafana'.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' grafana4.create_update_dashboard dashboard=<dashboard> overwrite=True orgname=<orgname>\n    "
    if isinstance(profile, str):
        profile = __salt__['config.option'](profile)
    if orgname:
        switch_org(orgname, profile)
    response = requests.post('{}/api/dashboards/db'.format(profile.get('grafana_url')), json=kwargs, auth=_get_auth(profile), headers=_get_headers(profile), timeout=profile.get('grafana_timeout', 3))
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()