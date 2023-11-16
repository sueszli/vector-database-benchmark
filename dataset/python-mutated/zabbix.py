"""
Support for Zabbix

:optdepends:    - zabbix server

:configuration: This module is not usable until the zabbix user and zabbix password are specified either in a pillar
    or in the minion's config file. Zabbix url should be also specified.

    .. code-block:: yaml

        zabbix.user: Admin
        zabbix.password: mypassword
        zabbix.url: http://127.0.0.1/zabbix/api_jsonrpc.php


    Connection arguments from the minion config file can be overridden on the CLI by using arguments with
    ``_connection_`` prefix.

    .. code-block:: bash

        zabbix.apiinfo_version _connection_user=Admin _connection_password=zabbix _connection_url=http://host/zabbix/

:codeauthor: Jiri Kotlin <jiri.kotlin@ultimum.io>
"""
import logging
import os
import socket
import urllib.error
import salt.utils.data
import salt.utils.files
import salt.utils.http
import salt.utils.json
from salt.exceptions import SaltException
from salt.utils.versions import Version
log = logging.getLogger(__name__)
__deprecated__ = (3009, 'zabbix', 'https://github.com/salt-extensions/saltext-zabbix')
INTERFACE_DEFAULT_PORTS = [10050, 161, 623, 12345]
ZABBIX_TOP_LEVEL_OBJECTS = ('hostgroup', 'template', 'host', 'maintenance', 'action', 'drule', 'service', 'proxy', 'screen', 'usergroup', 'mediatype', 'script', 'valuemap')
ZABBIX_ID_MAPPER = {'action': 'actionid', 'alert': 'alertid', 'application': 'applicationid', 'dhost': 'dhostid', 'dservice': 'dserviceid', 'dcheck': 'dcheckid', 'drule': 'druleid', 'event': 'eventid', 'graph': 'graphid', 'graphitem': 'gitemid', 'graphprototype': 'graphid', 'history': 'itemid', 'host': 'hostid', 'hostgroup': 'groupid', 'hostinterface': 'interfaceid', 'hostprototype': 'hostid', 'iconmap': 'iconmapid', 'image': 'imageid', 'item': 'itemid', 'itemprototype': 'itemid', 'service': 'serviceid', 'discoveryrule': 'itemid', 'maintenance': 'maintenanceid', 'map': 'sysmapid', 'usermedia': 'mediaid', 'mediatype': 'mediatypeid', 'proxy': 'proxyid', 'screen': 'screenid', 'screenitem': 'screenitemid', 'script': 'scriptid', 'template': 'templateid', 'templatescreen': 'screenid', 'templatescreenitem': 'screenitemid', 'trend': 'itemid', 'trigger': 'triggerid', 'triggerprototype': 'triggerid', 'user': 'userid', 'usergroup': 'usrgrpid', 'usermacro': 'globalmacroid', 'valuemap': 'valuemapid', 'httptest': 'httptestid'}
__virtualname__ = 'zabbix'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load the module if all modules are imported correctly.\n    '
    return __virtualname__

def _frontend_url():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tries to guess the url of zabbix frontend.\n\n    .. versionadded:: 2016.3.0\n    '
    hostname = socket.gethostname()
    frontend_url = 'http://' + hostname + '/zabbix/api_jsonrpc.php'
    try:
        try:
            response = salt.utils.http.query(frontend_url)
            error = response['error']
        except urllib.error.HTTPError as http_e:
            error = str(http_e)
        if error.find('412: Precondition Failed'):
            return frontend_url
        else:
            raise KeyError
    except (ValueError, KeyError):
        return False

def _query(method, params, url, auth=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    JSON request to Zabbix API.\n\n    .. versionadded:: 2016.3.0\n\n    :param method: actual operation to perform via the API\n    :param params: parameters required for specific method\n    :param url: url of zabbix api\n    :param auth: auth token for zabbix api (only for methods with required authentication)\n\n    :return: Response from API with desired data in JSON format. In case of error returns more specific description.\n\n    .. versionchanged:: 2017.7.0\n    '
    unauthenticated_methods = ['user.login', 'apiinfo.version']
    header_dict = {'Content-type': 'application/json'}
    data = {'jsonrpc': '2.0', 'id': 0, 'method': method, 'params': params}
    if method not in unauthenticated_methods:
        data['auth'] = auth
    data = salt.utils.json.dumps(data)
    log.info('_QUERY input:\nurl: %s\ndata: %s', str(url), str(data))
    try:
        result = salt.utils.http.query(url, method='POST', data=data, header_dict=header_dict, decode_type='json', decode=True, status=True, headers=True)
        log.info('_QUERY result: %s', str(result))
        if 'error' in result:
            raise SaltException('Zabbix API: Status: {} ({})'.format(result['status'], result['error']))
        ret = result.get('dict', {})
        if 'error' in ret:
            raise SaltException('Zabbix API: {} ({})'.format(ret['error']['message'], ret['error']['data']))
        return ret
    except ValueError as err:
        raise SaltException(f'URL or HTTP headers are probably not correct! ({err})')
    except OSError as err:
        raise SaltException(f'Check hostname in URL! ({err})')

def _login(**kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Log in to the API and generate the authentication token.\n\n    .. versionadded:: 2016.3.0\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: On success connargs dictionary with auth token and frontend url, False on failure.\n\n    "
    connargs = dict()

    def _connarg(name, key=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Add key to connargs, only if name exists in our kwargs or, as zabbix.<name> in __opts__ or __pillar__\n\n        Evaluate in said order - kwargs, opts, then pillar. To avoid collision with other functions,\n        kwargs-based connection arguments are prefixed with 'connection_' (i.e. '_connection_user', etc.).\n\n        Inspired by mysql salt module.\n        "
        if key is None:
            key = name
        if name in kwargs:
            connargs[key] = kwargs[name]
        else:
            prefix = '_connection_'
            if name.startswith(prefix):
                try:
                    name = name[len(prefix):]
                except IndexError:
                    return
            val = __salt__['config.get'](f'zabbix.{name}', None) or __salt__['config.get'](f'zabbix:{name}', None)
            if val is not None:
                connargs[key] = val
    _connarg('_connection_user', 'user')
    _connarg('_connection_password', 'password')
    _connarg('_connection_url', 'url')
    if 'url' not in connargs:
        connargs['url'] = _frontend_url()
    try:
        if connargs['user'] and connargs['password'] and connargs['url']:
            params = {'user': connargs['user'], 'password': connargs['password']}
            method = 'user.login'
            ret = _query(method, params, connargs['url'])
            auth = ret['result']
            connargs['auth'] = auth
            connargs.pop('user', None)
            connargs.pop('password', None)
            return connargs
        else:
            raise KeyError
    except KeyError as err:
        raise SaltException(f'URL is probably not correct! ({err})')

def _params_extend(params, _ignore_name=False, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Extends the params dictionary by values from keyword arguments.\n\n    .. versionadded:: 2016.3.0\n\n    :param params: Dictionary with parameters for zabbix API.\n    :param _ignore_name: Salt State module is passing first line as 'name' parameter. If API uses optional parameter\n    'name' (for ex. host_create, user_create method), please use 'visible_name' or 'firstname' instead of 'name' to\n    not mess these values.\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: Extended params dictionary with parameters.\n\n    "
    for key in kwargs:
        if not key.startswith('_'):
            params.setdefault(key, kwargs[key])
    if _ignore_name:
        params.pop('name', None)
        if 'firstname' in params:
            params['name'] = params.pop('firstname')
        elif 'visible_name' in params:
            params['name'] = params.pop('visible_name')
    return params

def _map_to_list_of_dicts(source, key):
    if False:
        print('Hello World!')
    '\n    Maps list of values to list of dicts of values, eg:\n        [usrgrpid1, usrgrpid2, ...] => [{"usrgrpid": usrgrpid1}, {"usrgrpid": usrgrpid2}, ...]\n\n    :param source:  list of values\n    :param key: name of dict key\n    :return: List of dicts in format: [{key: elem}, ...]\n    '
    output = []
    for elem in source:
        output.append({key: elem})
    return output

def get_zabbix_id_mapper():
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2017.7.0\n\n    Make ZABBIX_ID_MAPPER constant available to state modules.\n\n    :return: ZABBIX_ID_MAPPER\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.get_zabbix_id_mapper\n    "
    return ZABBIX_ID_MAPPER

def substitute_params(input_object, extend_params=None, filter_key='name', **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2017.7.0\n\n    Go through Zabbix object params specification and if needed get given object ID from Zabbix API and put it back\n    as a value. Definition of the object is done via dict with keys "query_object" and "query_name".\n\n    :param input_object: Zabbix object type specified in state file\n    :param extend_params: Specify query with params\n    :param filter_key: Custom filtering key (default: name)\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module\'s docstring)\n\n    :return: Params structure with values converted to string for further comparison purposes\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' zabbix.substitute_params \'{"query_object": "object_name", "query_name": "specific_object_name"}\'\n    '
    if extend_params is None:
        extend_params = {}
    if isinstance(input_object, list):
        return [substitute_params(oitem, extend_params, filter_key, **kwargs) for oitem in input_object]
    elif isinstance(input_object, dict):
        if 'query_object' in input_object:
            query_params = {}
            if input_object['query_object'] not in ZABBIX_TOP_LEVEL_OBJECTS:
                query_params.update(extend_params)
            try:
                query_params.update({'filter': {filter_key: input_object['query_name']}})
                return get_object_id_by_params(input_object['query_object'], query_params, **kwargs)
            except KeyError:
                raise SaltException('Qyerying object ID requested but object name not provided: {}'.format(input_object))
        else:
            return {key: substitute_params(val, extend_params, filter_key, **kwargs) for (key, val) in input_object.items()}
    else:
        return str(input_object)

def compare_params(defined, existing, return_old_value=False):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2017.7.0\n\n    Compares Zabbix object definition against existing Zabbix object.\n\n    :param defined: Zabbix object definition taken from sls file.\n    :param existing: Existing Zabbix object taken from result of an API call.\n    :param return_old_value: Default False. If True, returns dict("old"=old_val, "new"=new_val) for rollback purpose.\n    :return: Params that are different from existing object. Result extended by\n        object ID can be passed directly to Zabbix API update method.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' zabbix.compare_params new_zabbix_object_dict existing_zabbix_onject_dict\n    '
    if not isinstance(defined, type(existing)):
        raise SaltException('Zabbix object comparison failed (data type mismatch). Expecting {}, got {}. Existing value: "{}", defined value: "{}").'.format(type(existing), type(defined), existing, defined))
    if not salt.utils.data.is_iter(defined):
        if str(defined) != str(existing) and return_old_value:
            return {'new': str(defined), 'old': str(existing)}
        elif str(defined) != str(existing) and (not return_old_value):
            return str(defined)
    if isinstance(defined, list):
        if len(defined) != len(existing):
            log.info('Different list length!')
            return {'new': defined, 'old': existing} if return_old_value else defined
        else:
            difflist = []
            for ditem in defined:
                d_in_e = []
                for eitem in existing:
                    comp = compare_params(ditem, eitem, return_old_value)
                    if return_old_value:
                        d_in_e.append(comp['new'])
                    else:
                        d_in_e.append(comp)
                if all(d_in_e):
                    difflist.append(ditem)
            if any(difflist) and return_old_value:
                return {'new': defined, 'old': existing}
            elif any(difflist) and (not return_old_value):
                return defined
    if isinstance(defined, dict):
        try:
            if set(defined) <= set(existing):
                intersection = set(defined) & set(existing)
                diffdict = {'new': {}, 'old': {}} if return_old_value else {}
                for i in intersection:
                    comp = compare_params(defined[i], existing[i], return_old_value)
                    if return_old_value:
                        if comp or (not comp and isinstance(comp, list)):
                            diffdict['new'].update({i: defined[i]})
                            diffdict['old'].update({i: existing[i]})
                    elif comp or (not comp and isinstance(comp, list)):
                        diffdict.update({i: defined[i]})
                return diffdict
            return {'new': defined, 'old': existing} if return_old_value else defined
        except TypeError:
            raise SaltException('Zabbix object comparison failed (data type mismatch). Expecting {}, got {}. Existing value: "{}", defined value: "{}").'.format(type(existing), type(defined), existing, defined))

def get_object_id_by_params(obj, params=None, **connection_args):
    if False:
        return 10
    "\n    .. versionadded:: 2017.7.0\n\n    Get ID of single Zabbix object specified by its name.\n\n    :param obj: Zabbix object type\n    :param params: Parameters by which object is uniquely identified\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: object ID\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.get_object_id_by_params object_type params=zabbix_api_query_parameters_dict\n    "
    if params is None:
        params = {}
    res = run_query(obj + '.get', params, **connection_args)
    if res and len(res) == 1:
        return str(res[0][ZABBIX_ID_MAPPER[obj]])
    else:
        raise SaltException('Zabbix API: Object does not exist or bad Zabbix user permissions or other unexpected result. Called method {} with params {}. Result: {}'.format(obj + '.get', params, res))

def apiinfo_version(**connection_args):
    if False:
        print('Hello World!')
    "\n    Retrieve the version of the Zabbix API.\n\n    .. versionadded:: 2016.3.0\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: On success string with Zabbix API version, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.apiinfo_version\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'apiinfo.version'
            params = {}
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']
        else:
            raise KeyError
    except KeyError:
        return False

def user_create(alias, passwd, usrgrps, **connection_args):
    if False:
        return 10
    "\n    .. versionadded:: 2016.3.0\n\n    Create new zabbix user\n\n    .. note::\n        This function accepts all standard user properties: keyword argument\n        names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/2.0/manual/appendix/api/user/definitions#user\n\n    :param alias: user alias\n    :param passwd: user's password\n    :param usrgrps: user groups to add the user to\n\n    :param _connection_user: zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: url of zabbix frontend (can also be set in opts or pillar, see module's docstring)\n\n    :param firstname: string with firstname of the user, use 'firstname' instead of 'name' parameter to not mess\n                      with value supplied from Salt sls file.\n\n    :return: On success string with id of the created user.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.user_create james password007 '[7, 12]' firstname='James Bond'\n    "
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    username_field = 'alias'
    if Version(zabbix_version) > Version('5.2'):
        username_field = 'username'
    try:
        if conn_args:
            method = 'user.create'
            params = {username_field: alias, 'passwd': passwd, 'usrgrps': []}
            if not isinstance(usrgrps, list):
                usrgrps = [usrgrps]
            params['usrgrps'] = _map_to_list_of_dicts(usrgrps, 'usrgrpid')
            params = _params_extend(params, _ignore_name=True, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['userids']
        else:
            raise KeyError
    except KeyError:
        return ret

def user_delete(users, **connection_args):
    if False:
        print('Hello World!')
    "\n    Delete zabbix users.\n\n    .. versionadded:: 2016.3.0\n\n    :param users: array of users (userids) to delete\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: On success array with userids of deleted users.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.user_delete 15\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'user.delete'
            if not isinstance(users, list):
                params = [users]
            else:
                params = users
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['userids']
        else:
            raise KeyError
    except KeyError:
        return ret

def user_exists(alias, **connection_args):
    if False:
        while True:
            i = 10
    "\n    Checks if user with given alias exists.\n\n    .. versionadded:: 2016.3.0\n\n    :param alias: user alias\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: True if user exists, else False.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.user_exists james\n    "
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    username_field = 'alias'
    if Version(zabbix_version) > Version('5.2'):
        username_field = 'username'
    try:
        if conn_args:
            method = 'user.get'
            params = {'output': 'extend', 'filter': {username_field: alias}}
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return True if ret['result'] else False
        else:
            raise KeyError
    except KeyError:
        return ret

def user_get(alias=None, userids=None, **connection_args):
    if False:
        i = 10
        return i + 15
    "\n    Retrieve users according to the given parameters.\n\n    .. versionadded:: 2016.3.0\n\n    :param alias: user alias\n    :param userids: return only users with the given IDs\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: Array with details of convenient users, False on failure of if no user found.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.user_get james\n    "
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    username_field = 'alias'
    if Version(zabbix_version) > Version('5.2'):
        username_field = 'username'
    try:
        if conn_args:
            method = 'user.get'
            params = {'output': 'extend', 'selectUsrgrps': 'extend', 'selectMedias': 'extend', 'selectMediatypes': 'extend', 'filter': {}}
            if not userids and (not alias):
                return {'result': False, 'comment': 'Please submit alias or userids parameter to retrieve users.'}
            if alias:
                params['filter'].setdefault(username_field, alias)
            if userids:
                params.setdefault('userids', userids)
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result'] if ret['result'] else False
        else:
            raise KeyError
    except KeyError:
        return ret

def user_update(userid, **connection_args):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2016.3.0\n\n    Update existing users\n\n    .. note::\n        This function accepts all standard user properties: keyword argument\n        names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/2.0/manual/appendix/api/user/definitions#user\n\n    :param userid: id of the user to update\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: Id of the updated user on success.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.user_update 16 visible_name='James Brown'\n    "
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    medias = connection_args.pop('medias', None)
    if medias is None:
        medias = connection_args.pop('user_medias', None)
    else:
        medias.extend(connection_args.pop('user_medias', []))
    try:
        if conn_args:
            method = 'user.update'
            params = {'userid': userid}
            if Version(zabbix_version) < Version('3.4') and medias is not None:
                ret = {'result': False, 'comment': 'Setting medias available in Zabbix 3.4+'}
                return ret
            elif Version(zabbix_version) > Version('5.0') and medias is not None:
                params['medias'] = medias
            elif medias is not None:
                params['user_medias'] = medias
            if 'usrgrps' in connection_args:
                params['usrgrps'] = _map_to_list_of_dicts(connection_args.pop('usrgrps'), 'usrgrpid')
            params = _params_extend(params, _ignore_name=True, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['userids']
        else:
            raise KeyError
    except KeyError:
        return ret

def user_getmedia(userids=None, **connection_args):
    if False:
        return 10
    "\n    .. versionadded:: 2016.3.0\n\n    Retrieve media according to the given parameters\n\n    .. note::\n        This function accepts all standard usermedia.get properties: keyword\n        argument names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/3.2/manual/api/reference/usermedia/get\n\n    :param userids: return only media that are used by the given users\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: List of retrieved media, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.user_getmedia\n    "
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    if Version(zabbix_version) > Version('3.4'):
        users = user_get(userids=userids, **connection_args)
        medias = []
        for user in users:
            medias.extend(user.get('medias', []))
        return medias
    try:
        if conn_args:
            method = 'usermedia.get'
            params = {}
            if userids:
                params['userids'] = userids
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']
        else:
            raise KeyError
    except KeyError:
        return ret

def user_addmedia(userids, active, mediatypeid, period, sendto, severity, **connection_args):
    if False:
        print('Hello World!')
    "\n    Add new media to multiple users. Available only for Zabbix version 3.4 or older.\n\n    .. versionadded:: 2016.3.0\n\n    :param userids: ID of the user that uses the media\n    :param active: Whether the media is enabled (0 enabled, 1 disabled)\n    :param mediatypeid: ID of the media type used by the media\n    :param period: Time when the notifications can be sent as a time period\n    :param sendto: Address, user name or other identifier of the recipient\n    :param severity: Trigger severities to send notifications about\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: IDs of the created media.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.user_addmedia 4 active=0 mediatypeid=1 period='1-7,00:00-24:00' sendto='support2@example.com'\n        severity=63\n\n    "
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    method = 'user.addmedia'
    if Version(zabbix_version) > Version('3.4'):
        ret = {'result': False, 'comment': "Method '{}' removed in Zabbix 4.0+ use 'user.update'".format(method)}
        return ret
    try:
        if conn_args:
            method = 'user.addmedia'
            params = {'users': []}
            if not isinstance(userids, list):
                userids = [userids]
            for user in userids:
                params['users'].append({'userid': user})
            params['medias'] = [{'active': active, 'mediatypeid': mediatypeid, 'period': period, 'sendto': sendto, 'severity': severity}]
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['mediaids']
        else:
            raise KeyError
    except KeyError:
        return ret

def user_deletemedia(mediaids, **connection_args):
    if False:
        while True:
            i = 10
    "\n    Delete media by id. Available only for Zabbix version 3.4 or older.\n\n    .. versionadded:: 2016.3.0\n\n    :param mediaids: IDs of the media to delete\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: IDs of the deleted media, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.user_deletemedia 27\n    "
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    method = 'user.deletemedia'
    if Version(zabbix_version) > Version('3.4'):
        ret = {'result': False, 'comment': "Method '{}' removed in Zabbix 4.0+ use 'user.update'".format(method)}
        return ret
    try:
        if conn_args:
            if not isinstance(mediaids, list):
                mediaids = [mediaids]
            params = mediaids
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['mediaids']
        else:
            raise KeyError
    except KeyError:
        return ret

def user_list(**connection_args):
    if False:
        print('Hello World!')
    "\n    Retrieve all of the configured users.\n\n    .. versionadded:: 2016.3.0\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: Array with user details.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.user_list\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'user.get'
            params = {'output': 'extend'}
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']
        else:
            raise KeyError
    except KeyError:
        return ret

def usergroup_create(name, **connection_args):
    if False:
        return 10
    "\n    .. versionadded:: 2016.3.0\n\n    Create new user group\n\n    .. note::\n        This function accepts all standard user group properties: keyword\n        argument names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/2.0/manual/appendix/api/usergroup/definitions#user_group\n\n    :param name: name of the user group\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return:  IDs of the created user groups.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usergroup_create GroupName\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'usergroup.create'
            params = {'name': name}
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['usrgrpids']
        else:
            raise KeyError
    except KeyError:
        return ret

def usergroup_delete(usergroupids, **connection_args):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2016.3.0\n\n    :param usergroupids: IDs of the user groups to delete\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: IDs of the deleted user groups.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usergroup_delete 28\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'usergroup.delete'
            if not isinstance(usergroupids, list):
                usergroupids = [usergroupids]
            params = usergroupids
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['usrgrpids']
        else:
            raise KeyError
    except KeyError:
        return ret

def usergroup_exists(name=None, node=None, nodeids=None, **connection_args):
    if False:
        i = 10
        return i + 15
    "\n    Checks if at least one user group that matches the given filter criteria exists\n\n    .. versionadded:: 2016.3.0\n\n    :param name: names of the user groups\n    :param node: name of the node the user groups must belong to (This will override the nodeids parameter.)\n    :param nodeids: IDs of the nodes the user groups must belong to\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: True if at least one user group that matches the given filter criteria exists, else False.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usergroup_exists Guests\n    "
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    try:
        if conn_args:
            if Version(zabbix_version) > Version('2.5'):
                if not name:
                    name = ''
                ret = usergroup_get(name, None, **connection_args)
                return bool(ret)
            else:
                method = 'usergroup.exists'
                params = {}
                if not name and (not node) and (not nodeids):
                    return {'result': False, 'comment': 'Please submit name, node or nodeids parameter to check if at least one user group exists.'}
                if name:
                    params['name'] = name
                if Version(zabbix_version) < Version('2.4'):
                    if node:
                        params['node'] = node
                    if nodeids:
                        params['nodeids'] = nodeids
                ret = _query(method, params, conn_args['url'], conn_args['auth'])
                return ret['result']
        else:
            raise KeyError
    except KeyError:
        return ret

def usergroup_get(name=None, usrgrpids=None, userids=None, **connection_args):
    if False:
        print('Hello World!')
    "\n    .. versionadded:: 2016.3.0\n\n    Retrieve user groups according to the given parameters\n\n    .. note::\n        This function accepts all usergroup_get properties: keyword argument\n        names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/2.4/manual/api/reference/usergroup/get\n\n    :param name: names of the user groups\n    :param usrgrpids: return only user groups with the given IDs\n    :param userids: return only user groups that contain the given users\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: Array with convenient user groups details, False if no user group found or on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usergroup_get Guests\n    "
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'usergroup.get'
            if Version(zabbix_version) > Version('2.5'):
                params = {'selectRights': 'extend', 'output': 'extend', 'filter': {}}
            else:
                params = {'output': 'extend', 'filter': {}}
            if not name and (not usrgrpids) and (not userids):
                return False
            if name:
                params['filter'].setdefault('name', name)
            if usrgrpids:
                params.setdefault('usrgrpids', usrgrpids)
            if userids:
                params.setdefault('userids', userids)
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return False if not ret['result'] else ret['result']
        else:
            raise KeyError
    except KeyError:
        return ret

def usergroup_update(usrgrpid, **connection_args):
    if False:
        for i in range(10):
            print('nop')
    "\n    .. versionadded:: 2016.3.0\n\n    Update existing user group\n\n    .. note::\n        This function accepts all standard user group properties: keyword\n        argument names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/2.4/manual/api/reference/usergroup/object#user_group\n\n    :param usrgrpid: ID of the user group to update.\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: IDs of the updated user group, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usergroup_update 8 name=guestsRenamed\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'usergroup.update'
            params = {'usrgrpid': usrgrpid}
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['usrgrpids']
        else:
            raise KeyError
    except KeyError:
        return ret

def usergroup_list(**connection_args):
    if False:
        i = 10
        return i + 15
    "\n    Retrieve all enabled user groups.\n\n    .. versionadded:: 2016.3.0\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: Array with enabled user groups details, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usergroup_list\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'usergroup.get'
            params = {'output': 'extend'}
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']
        else:
            raise KeyError
    except KeyError:
        return ret

def host_create(host, groups, interfaces, **connection_args):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2016.3.0\n\n    Create new host\n\n    .. note::\n        This function accepts all standard host properties: keyword argument\n        names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/2.4/manual/api/reference/host/object#host\n\n    :param host: technical name of the host\n    :param groups: groupids of host groups to add the host to\n    :param interfaces: interfaces to be created for the host\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module\'s docstring)\n    :param visible_name: string with visible name of the host, use\n        \'visible_name\' instead of \'name\' parameter to not mess with value\n        supplied from Salt sls file.\n\n    return: ID of the created host.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' zabbix.host_create technicalname 4\n        interfaces=\'{type: 1, main: 1, useip: 1, ip: "192.168.3.1", dns: "", port: 10050}\'\n        visible_name=\'Host Visible Name\' inventory_mode=0 inventory=\'{"alias": "something"}\'\n    '
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'host.create'
            params = {'host': host}
            if not isinstance(groups, list):
                groups = [groups]
            params['groups'] = _map_to_list_of_dicts(groups, 'groupid')
            if not isinstance(interfaces, list):
                interfaces = [interfaces]
            params['interfaces'] = interfaces
            params = _params_extend(params, _ignore_name=True, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['hostids']
        else:
            raise KeyError
    except KeyError:
        return ret

def host_delete(hostids, **connection_args):
    if False:
        i = 10
        return i + 15
    "\n    Delete hosts.\n\n    .. versionadded:: 2016.3.0\n\n    :param hostids: Hosts (hostids) to delete.\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: IDs of the deleted hosts.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.host_delete 10106\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'host.delete'
            if not isinstance(hostids, list):
                params = [hostids]
            else:
                params = hostids
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['hostids']
        else:
            raise KeyError
    except KeyError:
        return ret

def host_exists(host=None, hostid=None, name=None, node=None, nodeids=None, **connection_args):
    if False:
        for i in range(10):
            print('nop')
    "\n    Checks if at least one host that matches the given filter criteria exists.\n\n    .. versionadded:: 2016.3.0\n\n    :param host: technical name of the host\n    :param hostids: Hosts (hostids) to delete.\n    :param name: visible name of the host\n    :param node: name of the node the hosts must belong to (zabbix API < 2.4)\n    :param nodeids: IDs of the node the hosts must belong to (zabbix API < 2.4)\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: IDs of the deleted hosts, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.host_exists 'Zabbix server'\n    "
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    try:
        if conn_args:
            if Version(zabbix_version) > Version('2.5'):
                if not host:
                    host = None
                if not name:
                    name = None
                if not hostid:
                    hostid = None
                ret = host_get(host, name, hostid, **connection_args)
                return bool(ret)
            else:
                method = 'host.exists'
                params = {}
                if hostid:
                    params['hostid'] = hostid
                if host:
                    params['host'] = host
                if name:
                    params['name'] = name
                if Version(zabbix_version) < Version('2.4'):
                    if node:
                        params['node'] = node
                    if nodeids:
                        params['nodeids'] = nodeids
                if not hostid and (not host) and (not name) and (not node) and (not nodeids):
                    return {'result': False, 'comment': 'Please submit hostid, host, name, node or nodeids parameter tocheck if at least one host that matches the given filter criteria exists.'}
                ret = _query(method, params, conn_args['url'], conn_args['auth'])
                return ret['result']
        else:
            raise KeyError
    except KeyError:
        return ret

def host_get(host=None, name=None, hostids=None, **connection_args):
    if False:
        return 10
    "\n    .. versionadded:: 2016.3.0\n\n    Retrieve hosts according to the given parameters\n\n    .. note::\n        This function accepts all optional host.get parameters: keyword\n        argument names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/2.4/manual/api/reference/host/get\n\n    :param host: technical name of the host\n    :param name: visible name of the host\n    :param hostids: ids of the hosts\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n\n    :return: Array with convenient hosts details, False if no host found or on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.host_get 'Zabbix server'\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'host.get'
            params = {'output': 'extend', 'filter': {}}
            if not name and (not hostids) and (not host):
                return False
            if name:
                params['filter'].setdefault('name', name)
            if hostids:
                params.setdefault('hostids', hostids)
            if host:
                params['filter'].setdefault('host', host)
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result'] if ret['result'] else False
        else:
            raise KeyError
    except KeyError:
        return ret

def host_update(hostid, **connection_args):
    if False:
        while True:
            i = 10
    "\n    .. versionadded:: 2016.3.0\n\n    Update existing hosts\n\n    .. note::\n        This function accepts all standard host and host.update properties:\n        keyword argument names differ depending on your zabbix version, see the\n        documentation for `host objects`_ and the documentation for `updating\n        hosts`_.\n\n        .. _`host objects`: https://www.zabbix.com/documentation/2.4/manual/api/reference/host/object#host\n        .. _`updating hosts`: https://www.zabbix.com/documentation/2.4/manual/api/reference/host/update\n\n    :param hostid: ID of the host to update\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n    :param visible_name: string with visible name of the host, use\n        'visible_name' instead of 'name' parameter to not mess with value\n        supplied from Salt sls file.\n\n    :return: ID of the updated host.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.host_update 10084 name='Zabbix server2'\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'host.update'
            params = {'hostid': hostid}
            if 'groups' in connection_args:
                params['groups'] = _map_to_list_of_dicts(connection_args.pop('groups'), 'groupid')
            params = _params_extend(params, _ignore_name=True, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['hostids']
        else:
            raise KeyError
    except KeyError:
        return ret

def host_inventory_get(hostids, **connection_args):
    if False:
        return 10
    "\n    Retrieve host inventory according to the given parameters.\n    See: https://www.zabbix.com/documentation/2.4/manual/api/reference/host/object#host_inventory\n\n    .. versionadded:: 2019.2.0\n\n    :param hostids: ID of the host to query\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: Array with host inventory fields, populated or not, False if host inventory is disabled or on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.host_inventory_get 101054\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'host.get'
            params = {'selectInventory': 'extend'}
            if hostids:
                params.setdefault('hostids', hostids)
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result'][0]['inventory'] if ret['result'] and ret['result'][0]['inventory'] else False
        else:
            raise KeyError
    except KeyError:
        return ret

def host_inventory_set(hostid, **connection_args):
    if False:
        for i in range(10):
            print('nop')
    "\n    Update host inventory items\n    NOTE: This function accepts all standard host: keyword argument names for inventory\n    see: https://www.zabbix.com/documentation/2.4/manual/api/reference/host/object#host_inventory\n\n    .. versionadded:: 2019.2.0\n\n    :param hostid: ID of the host to update\n    :param clear_old: Set to True in order to remove all existing inventory items before setting the specified items\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: ID of the updated host, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.host_inventory_set 101054 asset_tag=jml3322 type=vm clear_old=True\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            params = {}
            clear_old = False
            method = 'host.update'
            if connection_args.get('clear_old'):
                clear_old = True
            connection_args.pop('clear_old', None)
            inventory_mode = connection_args.pop('inventory_mode', '0')
            inventory_params = dict(_params_extend(params, **connection_args))
            for key in inventory_params:
                params.pop(key, None)
            if hostid:
                params.setdefault('hostid', hostid)
            if clear_old:
                params['inventory_mode'] = '-1'
                ret = _query(method, params, conn_args['url'], conn_args['auth'])
            params['inventory_mode'] = inventory_mode
            params['inventory'] = inventory_params
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']
        else:
            raise KeyError
    except KeyError:
        return ret

def host_list(**connection_args):
    if False:
        print('Hello World!')
    "\n    Retrieve all hosts.\n\n    .. versionadded:: 2016.3.0\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: Array with details about hosts, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.host_list\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'host.get'
            params = {'output': 'extend'}
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']
        else:
            raise KeyError
    except KeyError:
        return ret

def hostgroup_create(name, **connection_args):
    if False:
        return 10
    "\n    .. versionadded:: 2016.3.0\n\n    Create a host group\n\n    .. note::\n        This function accepts all standard host group properties: keyword\n        argument names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/2.4/manual/api/reference/hostgroup/object#host_group\n\n    :param name: name of the host group\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: ID of the created host group.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.hostgroup_create MyNewGroup\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'hostgroup.create'
            params = {'name': name}
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['groupids']
        else:
            raise KeyError
    except KeyError:
        return ret

def hostgroup_delete(hostgroupids, **connection_args):
    if False:
        while True:
            i = 10
    "\n    Delete the host group.\n\n    .. versionadded:: 2016.3.0\n\n    :param hostgroupids: IDs of the host groups to delete\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: ID of the deleted host groups, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.hostgroup_delete 23\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'hostgroup.delete'
            if not isinstance(hostgroupids, list):
                params = [hostgroupids]
            else:
                params = hostgroupids
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['groupids']
        else:
            raise KeyError
    except KeyError:
        return ret

def hostgroup_exists(name=None, groupid=None, node=None, nodeids=None, **connection_args):
    if False:
        return 10
    "\n    Checks if at least one host group that matches the given filter criteria exists.\n\n    .. versionadded:: 2016.3.0\n\n    :param name: names of the host groups\n    :param groupid: host group IDs\n    :param node: name of the node the host groups must belong to (zabbix API < 2.4)\n    :param nodeids: IDs of the nodes the host groups must belong to (zabbix API < 2.4)\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: True if at least one host group exists, False if not or on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.hostgroup_exists MyNewGroup\n    "
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    try:
        if conn_args:
            if Version(zabbix_version) > Version('2.5'):
                if not groupid:
                    groupid = None
                if not name:
                    name = None
                ret = hostgroup_get(name, groupid, **connection_args)
                return bool(ret)
            else:
                params = {}
                method = 'hostgroup.exists'
                if groupid:
                    params['groupid'] = groupid
                if name:
                    params['name'] = name
                if Version(zabbix_version) < Version('2.4'):
                    if node:
                        params['node'] = node
                    if nodeids:
                        params['nodeids'] = nodeids
                if not groupid and (not name) and (not node) and (not nodeids):
                    return {'result': False, 'comment': 'Please submit groupid, name, node or nodeids parameter tocheck if at least one host group that matches the given filter criteria exists.'}
                ret = _query(method, params, conn_args['url'], conn_args['auth'])
                return ret['result']
        else:
            raise KeyError
    except KeyError:
        return ret

def hostgroup_get(name=None, groupids=None, hostids=None, **connection_args):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2016.3.0\n\n    Retrieve host groups according to the given parameters\n\n    .. note::\n        This function accepts all standard hostgroup.get properities: keyword\n        argument names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/2.2/manual/api/reference/hostgroup/get\n\n    :param name: names of the host groups\n    :param groupid: host group IDs\n    :param node: name of the node the host groups must belong to\n    :param nodeids: IDs of the nodes the host groups must belong to\n    :param hostids: return only host groups that contain the given hosts\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: Array with host groups details, False if no convenient host group found or on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.hostgroup_get MyNewGroup\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'hostgroup.get'
            params = {'output': 'extend'}
            if not groupids and (not name) and (not hostids):
                return False
            if name:
                name_dict = {'name': name}
                params.setdefault('filter', name_dict)
            if groupids:
                params.setdefault('groupids', groupids)
            if hostids:
                params.setdefault('hostids', hostids)
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result'] if ret['result'] else False
        else:
            raise KeyError
    except KeyError:
        return ret

def hostgroup_update(groupid, name=None, **connection_args):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2016.3.0\n\n    Update existing hosts group\n\n    .. note::\n        This function accepts all standard host group properties: keyword\n        argument names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/2.4/manual/api/reference/hostgroup/object#host_group\n\n    :param groupid: ID of the host group to update\n    :param name: name of the host group\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: IDs of updated host groups.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.hostgroup_update 24 name='Renamed Name'\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'hostgroup.update'
            params = {'groupid': groupid}
            if name:
                params['name'] = name
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['groupids']
        else:
            raise KeyError
    except KeyError:
        return ret

def hostgroup_list(**connection_args):
    if False:
        return 10
    "\n    Retrieve all host groups.\n\n    .. versionadded:: 2016.3.0\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: Array with details about host groups, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.hostgroup_list\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'hostgroup.get'
            params = {'output': 'extend'}
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']
        else:
            raise KeyError
    except KeyError:
        return ret

def hostinterface_get(hostids, **connection_args):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2016.3.0\n\n    Retrieve host groups according to the given parameters\n\n    .. note::\n        This function accepts all standard hostinterface.get properities:\n        keyword argument names differ depending on your zabbix version, see\n        here__.\n\n        .. __: https://www.zabbix.com/documentation/2.4/manual/api/reference/hostinterface/get\n\n    :param hostids: Return only host interfaces used by the given hosts.\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: Array with host interfaces details, False if no convenient host interfaces found or on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.hostinterface_get 101054\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'hostinterface.get'
            params = {'output': 'extend'}
            if hostids:
                params.setdefault('hostids', hostids)
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result'] if ret['result'] else False
        else:
            raise KeyError
    except KeyError:
        return ret

def hostinterface_create(hostid, ip_, dns='', main=1, if_type=1, useip=1, port=None, **connection_args):
    if False:
        return 10
    "\n    .. versionadded:: 2016.3.0\n\n    Create new host interface\n\n    .. note::\n        This function accepts all standard host group interface: keyword\n        argument names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/3.0/manual/api/reference/hostinterface/object\n\n    :param hostid: ID of the host the interface belongs to\n\n    :param ip_: IP address used by the interface\n\n    :param dns: DNS name used by the interface\n\n    :param main: whether the interface is used as default on the host (0 - not default, 1 - default)\n\n    :param port: port number used by the interface\n\n    :param type: Interface type (1 - agent; 2 - SNMP; 3 - IPMI; 4 - JMX)\n\n    :param useip: Whether the connection should be made via IP (0 - connect\n        using host DNS name; 1 - connect using host IP address for this host\n        interface)\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or\n        pillar, see module's docstring)\n\n    :param _connection_password: Optional - zabbix password (can also be set in\n        opts or pillar, see module's docstring)\n\n    :param _connection_url: Optional - url of zabbix frontend (can also be set\n        in opts, pillar, see module's docstring)\n\n    :return: ID of the created host interface, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.hostinterface_create 10105 192.193.194.197\n    "
    conn_args = _login(**connection_args)
    ret = False
    if not port:
        port = INTERFACE_DEFAULT_PORTS[if_type]
    try:
        if conn_args:
            method = 'hostinterface.create'
            params = {'hostid': hostid, 'ip': ip_, 'dns': dns, 'main': main, 'port': port, 'type': if_type, 'useip': useip}
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['interfaceids']
        else:
            raise KeyError
    except KeyError:
        return ret

def hostinterface_delete(interfaceids, **connection_args):
    if False:
        while True:
            i = 10
    "\n    Delete host interface\n\n    .. versionadded:: 2016.3.0\n\n    :param interfaceids: IDs of the host interfaces to delete\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: ID of deleted host interfaces, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.hostinterface_delete 50\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'hostinterface.delete'
            if isinstance(interfaceids, list):
                params = interfaceids
            else:
                params = [interfaceids]
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['interfaceids']
        else:
            raise KeyError
    except KeyError:
        return ret

def hostinterface_update(interfaceid, **connection_args):
    if False:
        i = 10
        return i + 15
    "\n    .. versionadded:: 2016.3.0\n\n    Update host interface\n\n    .. note::\n        This function accepts all standard hostinterface: keyword argument\n        names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/2.4/manual/api/reference/hostinterface/object#host_interface\n\n    :param interfaceid: ID of the hostinterface to update\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: ID of the updated host interface, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.hostinterface_update 6 ip_=0.0.0.2\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'hostinterface.update'
            params = {'interfaceid': interfaceid}
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['interfaceids']
        else:
            raise KeyError
    except KeyError:
        return ret

def usermacro_get(macro=None, hostids=None, templateids=None, hostmacroids=None, globalmacroids=None, globalmacro=False, **connection_args):
    if False:
        return 10
    "\n    Retrieve user macros according to the given parameters.\n\n    Args:\n        macro:          name of the usermacro\n        hostids:        Return macros for the given hostids\n        templateids:    Return macros for the given templateids\n        hostmacroids:   Return macros with the given hostmacroids\n        globalmacroids: Return macros with the given globalmacroids (implies globalmacro=True)\n        globalmacro:    if True, returns only global macros\n\n\n        optional connection_args:\n                _connection_user: zabbix user (can also be set in opts or pillar, see module's docstring)\n                _connection_password: zabbix password (can also be set in opts or pillar, see module's docstring)\n                _connection_url: url of zabbix frontend (can also be set in opts or pillar, see module's docstring)\n\n    Returns:\n        Array with usermacro details, False if no usermacro found or on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usermacro_get macro='{$SNMP_COMMUNITY}'\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'usermacro.get'
            params = {'output': 'extend', 'filter': {}}
            if macro:
                if isinstance(macro, dict):
                    macro = '{' + str(next(iter(macro))) + '}'
                if not macro.startswith('{') and (not macro.endswith('}')):
                    macro = '{' + macro + '}'
                params['filter'].setdefault('macro', macro)
            if hostids:
                params.setdefault('hostids', hostids)
            elif templateids:
                params.setdefault('templateids', hostids)
            if hostmacroids:
                params.setdefault('hostmacroids', hostmacroids)
            elif globalmacroids:
                globalmacro = True
                params.setdefault('globalmacroids', globalmacroids)
            if globalmacro:
                params = _params_extend(params, globalmacro=True)
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result'] if ret['result'] else False
        else:
            raise KeyError
    except KeyError:
        return ret

def usermacro_create(macro, value, hostid, **connection_args):
    if False:
        print('Hello World!')
    "\n    Create new host usermacro.\n\n    :param macro: name of the host usermacro\n    :param value: value of the host usermacro\n    :param hostid: hostid or templateid\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    return: ID of the created host usermacro.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usermacro_create '{$SNMP_COMMUNITY}' 'public' 1\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            params = {}
            method = 'usermacro.create'
            if macro:
                if isinstance(macro, dict):
                    macro = '{' + str(next(iter(macro))) + '}'
                if not macro.startswith('{') and (not macro.endswith('}')):
                    macro = '{' + macro + '}'
                params['macro'] = macro
            params['value'] = value
            params['hostid'] = hostid
            params = _params_extend(params, _ignore_name=True, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['hostmacroids'][0]
        else:
            raise KeyError
    except KeyError:
        return ret

def usermacro_createglobal(macro, value, **connection_args):
    if False:
        print('Hello World!')
    "\n    Create new global usermacro.\n\n    :param macro: name of the global usermacro\n    :param value: value of the global usermacro\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    return: ID of the created global usermacro.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usermacro_createglobal '{$SNMP_COMMUNITY}' 'public'\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            params = {}
            method = 'usermacro.createglobal'
            if macro:
                if isinstance(macro, dict):
                    macro = '{' + str(next(iter(macro))) + '}'
                if not macro.startswith('{') and (not macro.endswith('}')):
                    macro = '{' + macro + '}'
                params['macro'] = macro
            params['value'] = value
            params = _params_extend(params, _ignore_name=True, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['globalmacroids'][0]
        else:
            raise KeyError
    except KeyError:
        return ret

def usermacro_delete(macroids, **connection_args):
    if False:
        print('Hello World!')
    "\n    Delete host usermacros.\n\n    :param macroids: macroids of the host usermacros\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    return: IDs of the deleted host usermacro.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usermacro_delete 21\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'usermacro.delete'
            if isinstance(macroids, list):
                params = macroids
            else:
                params = [macroids]
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['hostmacroids']
        else:
            raise KeyError
    except KeyError:
        return ret

def usermacro_deleteglobal(macroids, **connection_args):
    if False:
        i = 10
        return i + 15
    "\n    Delete global usermacros.\n\n    :param macroids: macroids of the global usermacros\n\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    return: IDs of the deleted global usermacro.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usermacro_deleteglobal 21\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'usermacro.deleteglobal'
            if isinstance(macroids, list):
                params = macroids
            else:
                params = [macroids]
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['globalmacroids']
        else:
            raise KeyError
    except KeyError:
        return ret

def usermacro_update(hostmacroid, value, **connection_args):
    if False:
        i = 10
        return i + 15
    "\n    Update existing host usermacro.\n\n    :param hostmacroid: id of the host usermacro\n    :param value: new value of the host usermacro\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    return: ID of the update host usermacro.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usermacro_update 1 'public'\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            params = {}
            method = 'usermacro.update'
            params['hostmacroid'] = hostmacroid
            params['value'] = value
            params = _params_extend(params, _ignore_name=True, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['hostmacroids'][0]
        else:
            raise KeyError
    except KeyError:
        return ret

def usermacro_updateglobal(globalmacroid, value, **connection_args):
    if False:
        for i in range(10):
            print('nop')
    "\n    Update existing global usermacro.\n\n    :param globalmacroid: id of the host usermacro\n    :param value: new value of the host usermacro\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    return: ID of the update global usermacro.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.usermacro_updateglobal 1 'public'\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            params = {}
            method = 'usermacro.updateglobal'
            params['globalmacroid'] = globalmacroid
            params['value'] = value
            params = _params_extend(params, _ignore_name=True, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['globalmacroids'][0]
        else:
            raise KeyError
    except KeyError:
        return ret

def mediatype_get(name=None, mediatypeids=None, **connection_args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Retrieve mediatypes according to the given parameters.\n\n    Args:\n        name:         Name or description of the mediatype\n        mediatypeids: ids of the mediatypes\n\n        optional connection_args:\n                _connection_user: zabbix user (can also be set in opts or pillar, see module\'s docstring)\n                _connection_password: zabbix password (can also be set in opts or pillar, see module\'s docstring)\n                _connection_url: url of zabbix frontend (can also be set in opts or pillar, see module\'s docstring)\n\n                all optional mediatype.get parameters: keyword argument names depends on your zabbix version, see:\n\n                https://www.zabbix.com/documentation/2.2/manual/api/reference/mediatype/get\n\n    Returns:\n        Array with mediatype details, False if no mediatype found or on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' zabbix.mediatype_get name=\'Email\'\n        salt \'*\' zabbix.mediatype_get mediatypeids="[\'1\', \'2\', \'3\']"\n    '
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'mediatype.get'
            params = {'output': 'extend', 'filter': {}}
            if name:
                if Version(zabbix_version) >= Version('4.4'):
                    params['filter'].setdefault('name', name)
                else:
                    params['filter'].setdefault('description', name)
            if mediatypeids:
                params.setdefault('mediatypeids', mediatypeids)
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result'] if ret['result'] else False
        else:
            raise KeyError
    except KeyError:
        return ret

def mediatype_create(name, mediatype, **connection_args):
    if False:
        while True:
            i = 10
    "\n    Create new mediatype\n\n    .. note::\n        This function accepts all standard mediatype properties: keyword\n        argument names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/3.0/manual/api/reference/mediatype/object\n\n    :param mediatype: media type - 0: email, 1: script, 2: sms, 3: Jabber, 100: Ez Texting\n    :param exec_path: exec path - Required for script and Ez Texting types, see Zabbix API docs\n    :param gsm_modem: exec path - Required for sms type, see Zabbix API docs\n    :param smtp_email: email address from which notifications will be sent, required for email type\n    :param smtp_helo: SMTP HELO, required for email type\n    :param smtp_server: SMTP server, required for email type\n    :param status: whether the media type is enabled - 0: enabled, 1: disabled\n    :param username: authentication user, required for Jabber and Ez Texting types\n    :param passwd: authentication password, required for Jabber and Ez Texting types\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    return: ID of the created mediatype.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.mediatype_create 'Email' 0 smtp_email='noreply@example.com'\n        smtp_server='mailserver.example.com' smtp_helo='zabbix.example.com'\n    "
    conn_args = _login(**connection_args)
    zabbix_version = apiinfo_version(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'mediatype.create'
            if Version(zabbix_version) >= Version('4.4'):
                params = {'name': name}
                _ignore_name = False
            else:
                params = {'description': name}
                _ignore_name = True
            params['type'] = mediatype
            params = _params_extend(params, _ignore_name, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['mediatypeid']
        else:
            raise KeyError
    except KeyError:
        return ret

def mediatype_delete(mediatypeids, **connection_args):
    if False:
        while True:
            i = 10
    "\n    Delete mediatype\n\n\n    :param interfaceids: IDs of the mediatypes to delete\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module's docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module's docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module's docstring)\n\n    :return: ID of deleted mediatype, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' zabbix.mediatype_delete 3\n    "
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'mediatype.delete'
            if isinstance(mediatypeids, list):
                params = mediatypeids
            else:
                params = [mediatypeids]
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['mediatypeids']
        else:
            raise KeyError
    except KeyError:
        return ret

def mediatype_update(mediatypeid, name=False, mediatype=False, **connection_args):
    if False:
        while True:
            i = 10
    '\n    Update existing mediatype\n\n    .. note::\n        This function accepts all standard mediatype properties: keyword\n        argument names differ depending on your zabbix version, see here__.\n\n        .. __: https://www.zabbix.com/documentation/3.0/manual/api/reference/mediatype/object\n\n    :param mediatypeid: ID of the mediatype to update\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module\'s docstring)\n\n    :return: IDs of the updated mediatypes, False on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' zabbix.usergroup_update 8 name="Email update"\n    '
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'mediatype.update'
            params = {'mediatypeid': mediatypeid}
            if name:
                params['description'] = name
            if mediatype:
                params['type'] = mediatype
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result']['mediatypeids']
        else:
            raise KeyError
    except KeyError:
        return ret

def template_get(name=None, host=None, templateids=None, **connection_args):
    if False:
        while True:
            i = 10
    '\n    Retrieve templates according to the given parameters.\n\n    Args:\n        host: technical name of the template\n        name: visible name of the template\n        hostids: ids of the templates\n\n        optional connection_args:\n                _connection_user: zabbix user (can also be set in opts or pillar, see module\'s docstring)\n                _connection_password: zabbix password (can also be set in opts or pillar, see module\'s docstring)\n                _connection_url: url of zabbix frontend (can also be set in opts or pillar, see module\'s docstring)\n\n                all optional template.get parameters: keyword argument names depends on your zabbix version, see:\n\n                https://www.zabbix.com/documentation/2.4/manual/api/reference/template/get\n\n    Returns:\n        Array with convenient template details, False if no template found or on failure.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' zabbix.template_get name=\'Template OS Linux\'\n        salt \'*\' zabbix.template_get templateids="[\'10050\', \'10001\']"\n    '
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            method = 'template.get'
            params = {'output': 'extend', 'filter': {}}
            if name:
                params['filter'].setdefault('name', name)
            if host:
                params['filter'].setdefault('host', host)
            if templateids:
                params.setdefault('templateids', templateids)
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            return ret['result'] if ret['result'] else False
        else:
            raise KeyError
    except KeyError:
        return ret

def run_query(method, params, **connection_args):
    if False:
        print('Hello World!')
    '\n    Send Zabbix API call\n\n    Args:\n        method: actual operation to perform via the API\n        params: parameters required for specific method\n\n        optional connection_args:\n                _connection_user: zabbix user (can also be set in opts or pillar, see module\'s docstring)\n                _connection_password: zabbix password (can also be set in opts or pillar, see module\'s docstring)\n                _connection_url: url of zabbix frontend (can also be set in opts or pillar, see module\'s docstring)\n\n                all optional template.get parameters: keyword argument names depends on your zabbix version, see:\n\n                https://www.zabbix.com/documentation/2.4/manual/api/reference/\n\n    Returns:\n        Response from Zabbix API\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' zabbix.run_query proxy.create \'{"host": "zabbixproxy.domain.com", "status": "5"}\'\n    '
    conn_args = _login(**connection_args)
    ret = False
    try:
        if conn_args:
            params = _params_extend(params, **connection_args)
            ret = _query(method, params, conn_args['url'], conn_args['auth'])
            if isinstance(ret['result'], bool):
                return ret['result']
            return ret['result'] if ret['result'] else False
        else:
            raise KeyError
    except KeyError:
        return ret

def configuration_import(config_file, rules=None, file_format='xml', **connection_args):
    if False:
        i = 10
        return i + 15
    '\n    .. versionadded:: 2017.7.0\n\n    Imports Zabbix configuration specified in file to Zabbix server.\n\n    :param config_file: File with Zabbix config (local or remote)\n    :param rules: Optional - Rules that have to be different from default (defaults are the same as in Zabbix web UI.)\n    :param file_format: Config file format (default: xml)\n    :param _connection_user: Optional - zabbix user (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_password: Optional - zabbix password (can also be set in opts or pillar, see module\'s docstring)\n    :param _connection_url: Optional - url of zabbix frontend (can also be set in opts, pillar, see module\'s docstring)\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' zabbix.configuration_import salt://zabbix/config/zabbix_templates.xml         "{\'screens\': {\'createMissing\': True, \'updateExisting\': True}}"\n    '
    zabbix_version = apiinfo_version(**connection_args)
    if rules is None:
        rules = {}
    default_rules = {'discoveryRules': {'createMissing': True, 'updateExisting': True, 'deleteMissing': False}, 'graphs': {'createMissing': True, 'updateExisting': True, 'deleteMissing': False}, 'groups': {'createMissing': True}, 'hosts': {'createMissing': False, 'updateExisting': False}, 'images': {'createMissing': False, 'updateExisting': False}, 'items': {'createMissing': True, 'updateExisting': True, 'deleteMissing': False}, 'maps': {'createMissing': False, 'updateExisting': False}, 'screens': {'createMissing': False, 'updateExisting': False}, 'templateLinkage': {'createMissing': True}, 'templates': {'createMissing': True, 'updateExisting': True}, 'templateScreens': {'createMissing': True, 'updateExisting': True, 'deleteMissing': False}, 'triggers': {'createMissing': True, 'updateExisting': True, 'deleteMissing': False}, 'valueMaps': {'createMissing': True, 'updateExisting': False}}
    if Version(zabbix_version) >= Version('3.2'):
        default_rules['httptests'] = {'createMissing': True, 'updateExisting': True, 'deleteMissing': False}
    if Version(zabbix_version) >= Version('3.4'):
        default_rules['applications'] = {'createMissing': True, 'deleteMissing': False}
    else:
        default_rules['applications'] = {'createMissing': True, 'updateExisting': True, 'deleteMissing': False}
    new_rules = dict(default_rules)
    if rules:
        for rule in rules:
            if rule in new_rules:
                new_rules[rule].update(rules[rule])
            else:
                new_rules[rule] = rules[rule]
    if 'salt://' in config_file:
        tmpfile = salt.utils.files.mkstemp()
        cfile = __salt__['cp.get_file'](config_file, tmpfile)
        if not cfile or os.path.getsize(cfile) == 0:
            return {'name': config_file, 'result': False, 'message': 'Failed to fetch config file.'}
    else:
        cfile = config_file
        if not os.path.isfile(cfile):
            return {'name': config_file, 'result': False, 'message': 'Invalid file path.'}
    with salt.utils.files.fopen(cfile, mode='r') as fp_:
        xml = fp_.read()
    if 'salt://' in config_file:
        salt.utils.files.safe_rm(cfile)
    params = {'format': file_format, 'rules': new_rules, 'source': xml}
    log.info('CONFIGURATION IMPORT: rules: %s', str(params['rules']))
    try:
        run_query('configuration.import', params, **connection_args)
        return {'name': config_file, 'result': True, 'message': 'Zabbix API "configuration.import" method called successfully.'}
    except SaltException as exc:
        return {'name': config_file, 'result': False, 'message': str(exc)}