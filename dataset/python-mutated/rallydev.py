"""
Support for RallyDev

.. versionadded:: 2015.8.0

Requires a ``username`` and a ``password`` in ``/etc/salt/minion``:

.. code-block:: yaml

    rallydev:
      username: myuser@example.com
      password: 123pass
"""
import logging
import salt.utils.http
import salt.utils.json
from salt.exceptions import SaltInvocationError
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load the module if apache is installed\n    '
    if not __opts__.get('rallydev', {}).get('username', None):
        return (False, 'The rallydev execution module failed to load: rallydev:username not defined in config.')
    if not __opts__.get('rallydev', {}).get('password', None):
        return (False, 'The rallydev execution module failed to load: rallydev:password not defined in config.')
    return True

def _get_token():
    if False:
        return 10
    '\n    Get an auth token\n    '
    username = __opts__.get('rallydev', {}).get('username', None)
    password = __opts__.get('rallydev', {}).get('password', None)
    path = 'https://rally1.rallydev.com/slm/webservice/v2.0/security/authorize'
    result = salt.utils.http.query(path, decode=True, decode_type='json', text=True, status=True, username=username, password=password, cookies=True, persist_session=True, opts=__opts__)
    if 'dict' not in result:
        return None
    return result['dict']['OperationResult']['SecurityToken']

def _query(action=None, command=None, args=None, method='GET', header_dict=None, data=None):
    if False:
        print('Hello World!')
    '\n    Make a web call to RallyDev.\n    '
    token = _get_token()
    username = __opts__.get('rallydev', {}).get('username', None)
    password = __opts__.get('rallydev', {}).get('password', None)
    path = 'https://rally1.rallydev.com/slm/webservice/v2.0/'
    if action:
        path += action
    if command:
        path += '/{}'.format(command)
    log.debug('RallyDev URL: %s', path)
    if not isinstance(args, dict):
        args = {}
    args['key'] = token
    if header_dict is None:
        header_dict = {'Content-type': 'application/json'}
    if method != 'POST':
        header_dict['Accept'] = 'application/json'
    decode = True
    if method == 'DELETE':
        decode = False
    return_content = None
    result = salt.utils.http.query(path, method, params=args, data=data, header_dict=header_dict, decode=decode, decode_type='json', text=True, status=True, username=username, password=password, cookies=True, persist_session=True, opts=__opts__)
    log.debug('RallyDev Response Status Code: %s', result['status'])
    if 'error' in result:
        log.error(result['error'])
        return [result['status'], result['error']]
    return [result['status'], result.get('dict', {})]

def list_items(name):
    if False:
        return 10
    '\n    List items of a particular type\n\n    CLI Examples:\n\n    .. code-block:: bash\n\n        salt myminion rallydev.list_<item name>s\n        salt myminion rallydev.list_users\n        salt myminion rallydev.list_artifacts\n    '
    (status, result) = _query(action=name)
    return result

def query_item(name, query_string, order='Rank'):
    if False:
        return 10
    "\n    Query a type of record for one or more items. Requires a valid query string.\n    See https://rally1.rallydev.com/slm/doc/webservice/introduction.jsp for\n    information on query syntax.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion rallydev.query_<item name> <query string> [<order>]\n        salt myminion rallydev.query_task '(Name contains github)'\n        salt myminion rallydev.query_task '(Name contains reactor)' Rank\n    "
    (status, result) = _query(action=name, args={'query': query_string, 'order': order})
    return result

def show_item(name, id_):
    if False:
        while True:
            i = 10
    '\n    Show an item\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion rallydev.show_<item name> <item id>\n    '
    (status, result) = _query(action=name, command=id_)
    return result

def update_item(name, id_, field=None, value=None, postdata=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Update an item. Either a field and a value, or a chunk of POST data, may be\n    used, but not both.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion rallydev.update_<item name> <item id> field=<field> value=<value>\n        salt myminion rallydev.update_<item name> <item id> postdata=<post data>\n    '
    if field and value:
        if postdata:
            raise SaltInvocationError('Either a field and a value, or a chunk of POST data, may be specified, but not both.')
        postdata = {name.title(): {field: value}}
    if postdata is None:
        raise SaltInvocationError('Either a field and a value, or a chunk of POST data must be specified.')
    (status, result) = _query(action=name, command=id_, method='POST', data=salt.utils.json.dumps(postdata))
    return result

def show_artifact(id_):
    if False:
        while True:
            i = 10
    '\n    Show an artifact\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion rallydev.show_artifact <artifact id>\n    '
    return show_item('artifact', id_)

def list_users():
    if False:
        print('Hello World!')
    '\n    List the users\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion rallydev.list_users\n    '
    return list_items('user')

def show_user(id_):
    if False:
        i = 10
        return i + 15
    '\n    Show a user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion rallydev.show_user <user id>\n    '
    return show_item('user', id_)

def update_user(id_, field, value):
    if False:
        i = 10
        return i + 15
    '\n    Update a user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion rallydev.update_user <user id> <field> <new value>\n    '
    return update_item('user', id_, field, value)

def query_user(query_string, order='UserName'):
    if False:
        print('Hello World!')
    "\n    Update a user\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt myminion rallydev.query_user '(Name contains Jo)'\n    "
    return query_item('user', query_string, order)