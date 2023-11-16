"""
Module for working with the Glassfish/Payara 4.x management API
.. versionadded:: 2016.11.0
:depends: requests
"""
import urllib.parse
import salt.defaults.exitcodes
import salt.utils.json
from salt.exceptions import CommandExecutionError
try:
    import requests
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
__virtualname__ = 'glassfish'
DEFAULT_SERVER = {'ssl': False, 'url': 'localhost', 'port': 4848, 'user': None, 'password': None}

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if requests is installed\n    '
    if HAS_LIBS:
        return __virtualname__
    else:
        return (False, 'The "{}" module could not be loaded: "requests" is not installed.'.format(__virtualname__))

def _get_headers():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return fixed dict with headers (JSON data + mandatory "Requested by" header)\n    '
    return {'Accept': 'application/json', 'Content-Type': 'application/json', 'X-Requested-By': 'GlassFish REST HTML interface'}

def _get_auth(username, password):
    if False:
        while True:
            i = 10
    '\n    Returns the HTTP auth header\n    '
    if username and password:
        return requests.auth.HTTPBasicAuth(username, password)
    else:
        return None

def _get_url(ssl, url, port, path):
    if False:
        return 10
    '\n    Returns the URL of the endpoint\n    '
    if ssl:
        return 'https://{}:{}/management/domain/{}'.format(url, port, path)
    else:
        return 'http://{}:{}/management/domain/{}'.format(url, port, path)

def _get_server(server):
    if False:
        while True:
            i = 10
    '\n    Returns the server information if provided, or the defaults\n    '
    return server if server else DEFAULT_SERVER

def _clean_data(data):
    if False:
        while True:
            i = 10
    '\n    Removes SaltStack params from **kwargs\n    '
    for key in list(data):
        if key.startswith('__pub'):
            del data[key]
    return data

def _api_response(response):
    if False:
        i = 10
        return i + 15
    '\n    Check response status code + success_code returned by glassfish\n    '
    if response.status_code == 404:
        __context__['retcode'] = salt.defaults.exitcodes.SALT_BUILD_FAIL
        raise CommandExecutionError("Element doesn't exists")
    if response.status_code == 401:
        __context__['retcode'] = salt.defaults.exitcodes.SALT_BUILD_FAIL
        raise CommandExecutionError('Bad username or password')
    elif response.status_code == 200 or response.status_code == 500:
        try:
            data = salt.utils.json.loads(response.content)
            if data['exit_code'] != 'SUCCESS':
                __context__['retcode'] = salt.defaults.exitcodes.SALT_BUILD_FAIL
                raise CommandExecutionError(data['message'])
            return data
        except ValueError:
            __context__['retcode'] = salt.defaults.exitcodes.SALT_BUILD_FAIL
            raise CommandExecutionError('The server returned no data')
    else:
        response.raise_for_status()

def _api_get(path, server=None):
    if False:
        print('Hello World!')
    '\n    Do a GET request to the API\n    '
    server = _get_server(server)
    response = requests.get(url=_get_url(server['ssl'], server['url'], server['port'], path), auth=_get_auth(server['user'], server['password']), headers=_get_headers(), verify=True)
    return _api_response(response)

def _api_post(path, data, server=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Do a POST request to the API\n    '
    server = _get_server(server)
    response = requests.post(url=_get_url(server['ssl'], server['url'], server['port'], path), auth=_get_auth(server['user'], server['password']), headers=_get_headers(), data=salt.utils.json.dumps(data), verify=True)
    return _api_response(response)

def _api_delete(path, data, server=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Do a DELETE request to the API\n    '
    server = _get_server(server)
    response = requests.delete(url=_get_url(server['ssl'], server['url'], server['port'], path), auth=_get_auth(server['user'], server['password']), headers=_get_headers(), params=data, verify=True)
    return _api_response(response)

def _enum_elements(name, server=None):
    if False:
        print('Hello World!')
    '\n    Enum elements\n    '
    elements = []
    data = _api_get(name, server)
    if any(data['extraProperties']['childResources']):
        for element in data['extraProperties']['childResources']:
            elements.append(element)
        return elements
    return None

def _get_element_properties(name, element_type, server=None):
    if False:
        i = 10
        return i + 15
    "\n    Get an element's properties\n    "
    properties = {}
    data = _api_get('{}/{}/property'.format(element_type, name), server)
    if any(data['extraProperties']['properties']):
        for element in data['extraProperties']['properties']:
            properties[element['name']] = element['value']
        return properties
    return {}

def _get_element(name, element_type, server=None, with_properties=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get an element with or without properties\n    '
    element = {}
    name = urllib.parse.quote(name, safe='')
    data = _api_get('{}/{}'.format(element_type, name), server)
    if any(data['extraProperties']['entity']):
        for (key, value) in data['extraProperties']['entity'].items():
            element[key] = value
        if with_properties:
            element['properties'] = _get_element_properties(name, element_type)
        return element
    return None

def _create_element(name, element_type, data, server=None):
    if False:
        while True:
            i = 10
    '\n    Create a new element\n    '
    if 'properties' in data:
        data['property'] = ''
        for (key, value) in data['properties'].items():
            if not data['property']:
                data['property'] += '{}={}'.format(key, value.replace(':', '\\:'))
            else:
                data['property'] += ':{}={}'.format(key, value.replace(':', '\\:'))
        del data['properties']
    _api_post(element_type, _clean_data(data), server)
    return urllib.parse.unquote(name)

def _update_element(name, element_type, data, server=None):
    if False:
        return 10
    '\n    Update an element, including its properties\n    '
    name = urllib.parse.quote(name, safe='')
    if 'properties' in data:
        properties = []
        for (key, value) in data['properties'].items():
            properties.append({'name': key, 'value': value})
        _api_post('{}/{}/property'.format(element_type, name), properties, server)
        del data['properties']
        if not data:
            return urllib.parse.unquote(name)
    update_data = _get_element(name, element_type, server, with_properties=False)
    if update_data:
        update_data.update(data)
    else:
        __context__['retcode'] = salt.defaults.exitcodes.SALT_BUILD_FAIL
        raise CommandExecutionError('Cannot update {}'.format(name))
    _api_post('{}/{}'.format(element_type, name), _clean_data(update_data), server)
    return urllib.parse.unquote(name)

def _delete_element(name, element_type, data, server=None):
    if False:
        return 10
    '\n    Delete an element\n    '
    _api_delete('{}/{}'.format(element_type, urllib.parse.quote(name, safe='')), data, server)
    return name

def enum_connector_c_pool(server=None):
    if False:
        i = 10
        return i + 15
    '\n    Enum connection pools\n    '
    return _enum_elements('resources/connector-connection-pool', server)

def get_connector_c_pool(name, server=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get a specific connection pool\n    '
    return _get_element(name, 'resources/connector-connection-pool', server)

def create_connector_c_pool(name, server=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Create a connection pool\n    '
    defaults = {'connectionDefinitionName': 'javax.jms.ConnectionFactory', 'resourceAdapterName': 'jmsra', 'associateWithThread': False, 'connectionCreationRetryAttempts': 0, 'connectionCreationRetryIntervalInSeconds': 0, 'connectionLeakReclaim': False, 'connectionLeakTimeoutInSeconds': 0, 'description': '', 'failAllConnections': False, 'id': name, 'idleTimeoutInSeconds': 300, 'isConnectionValidationRequired': False, 'lazyConnectionAssociation': False, 'lazyConnectionEnlistment': False, 'matchConnections': True, 'maxConnectionUsageCount': 0, 'maxPoolSize': 32, 'maxWaitTimeInMillis': 60000, 'ping': False, 'poolResizeQuantity': 2, 'pooling': True, 'steadyPoolSize': 8, 'target': 'server', 'transactionSupport': '', 'validateAtmostOncePeriodInSeconds': 0}
    data = defaults
    data.update(kwargs)
    if data['transactionSupport'] and data['transactionSupport'] not in ('XATransaction', 'LocalTransaction', 'NoTransaction'):
        raise CommandExecutionError('Invalid transaction support')
    return _create_element(name, 'resources/connector-connection-pool', data, server)

def update_connector_c_pool(name, server=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Update a connection pool\n    '
    if 'transactionSupport' in kwargs and kwargs['transactionSupport'] not in ('XATransaction', 'LocalTransaction', 'NoTransaction'):
        raise CommandExecutionError('Invalid transaction support')
    return _update_element(name, 'resources/connector-connection-pool', kwargs, server)

def delete_connector_c_pool(name, target='server', cascade=True, server=None):
    if False:
        return 10
    '\n    Delete a connection pool\n    '
    data = {'target': target, 'cascade': cascade}
    return _delete_element(name, 'resources/connector-connection-pool', data, server)

def enum_connector_resource(server=None):
    if False:
        while True:
            i = 10
    '\n    Enum connection resources\n    '
    return _enum_elements('resources/connector-resource', server)

def get_connector_resource(name, server=None):
    if False:
        i = 10
        return i + 15
    '\n    Get a specific connection resource\n    '
    return _get_element(name, 'resources/connector-resource', server)

def create_connector_resource(name, server=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a connection resource\n    '
    defaults = {'description': '', 'enabled': True, 'id': name, 'poolName': '', 'objectType': 'user', 'target': 'server'}
    data = defaults
    data.update(kwargs)
    if not data['poolName']:
        raise CommandExecutionError('No pool name!')
    for (key, value) in list(data.items()):
        del data[key]
        data[key.lower()] = value
    return _create_element(name, 'resources/connector-resource', data, server)

def update_connector_resource(name, server=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Update a connection resource\n    '
    if 'jndiName' in kwargs:
        del kwargs['jndiName']
    return _update_element(name, 'resources/connector-resource', kwargs, server)

def delete_connector_resource(name, target='server', server=None):
    if False:
        print('Hello World!')
    '\n    Delete a connection resource\n    '
    return _delete_element(name, 'resources/connector-resource', {'target': target}, server)

def enum_admin_object_resource(server=None):
    if False:
        i = 10
        return i + 15
    '\n    Enum JMS destinations\n    '
    return _enum_elements('resources/admin-object-resource', server)

def get_admin_object_resource(name, server=None):
    if False:
        while True:
            i = 10
    '\n    Get a specific JMS destination\n    '
    return _get_element(name, 'resources/admin-object-resource', server)

def create_admin_object_resource(name, server=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Create a JMS destination\n    '
    defaults = {'description': '', 'className': 'com.sun.messaging.Queue', 'enabled': True, 'id': name, 'resAdapter': 'jmsra', 'resType': 'javax.jms.Queue', 'target': 'server'}
    data = defaults
    data.update(kwargs)
    if data['resType'] == 'javax.jms.Queue':
        data['className'] = 'com.sun.messaging.Queue'
    elif data['resType'] == 'javax.jms.Topic':
        data['className'] = 'com.sun.messaging.Topic'
    else:
        raise CommandExecutionError('resType should be "javax.jms.Queue" or "javax.jms.Topic"!')
    if data['resAdapter'] != 'jmsra':
        raise CommandExecutionError('resAdapter should be "jmsra"!')
    if 'resType' in data:
        data['restype'] = data['resType']
        del data['resType']
    if 'className' in data:
        data['classname'] = data['className']
        del data['className']
    return _create_element(name, 'resources/admin-object-resource', data, server)

def update_admin_object_resource(name, server=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Update a JMS destination\n    '
    if 'jndiName' in kwargs:
        del kwargs['jndiName']
    return _update_element(name, 'resources/admin-object-resource', kwargs, server)

def delete_admin_object_resource(name, target='server', server=None):
    if False:
        i = 10
        return i + 15
    '\n    Delete a JMS destination\n    '
    return _delete_element(name, 'resources/admin-object-resource', {'target': target}, server)

def enum_jdbc_connection_pool(server=None):
    if False:
        i = 10
        return i + 15
    '\n    Enum JDBC pools\n    '
    return _enum_elements('resources/jdbc-connection-pool', server)

def get_jdbc_connection_pool(name, server=None):
    if False:
        while True:
            i = 10
    '\n    Get a specific JDBC pool\n    '
    return _get_element(name, 'resources/jdbc-connection-pool', server)

def create_jdbc_connection_pool(name, server=None, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Create a connection resource\n    '
    defaults = {'allowNonComponentCallers': False, 'associateWithThread': False, 'connectionCreationRetryAttempts': '0', 'connectionCreationRetryIntervalInSeconds': '10', 'connectionLeakReclaim': False, 'connectionLeakTimeoutInSeconds': '0', 'connectionValidationMethod': 'table', 'datasourceClassname': '', 'description': '', 'driverClassname': '', 'failAllConnections': False, 'idleTimeoutInSeconds': '300', 'initSql': '', 'isConnectionValidationRequired': False, 'isIsolationLevelGuaranteed': True, 'lazyConnectionAssociation': False, 'lazyConnectionEnlistment': False, 'matchConnections': False, 'maxConnectionUsageCount': '0', 'maxPoolSize': '32', 'maxWaitTimeInMillis': 60000, 'name': name, 'nonTransactionalConnections': False, 'ping': False, 'poolResizeQuantity': '2', 'pooling': True, 'resType': '', 'sqlTraceListeners': '', 'statementCacheSize': '0', 'statementLeakReclaim': False, 'statementLeakTimeoutInSeconds': '0', 'statementTimeoutInSeconds': '-1', 'steadyPoolSize': '8', 'target': 'server', 'transactionIsolationLevel': '', 'validateAtmostOncePeriodInSeconds': '0', 'validationClassname': '', 'validationTableName': '', 'wrapJdbcObjects': True}
    data = defaults
    data.update(kwargs)
    if data['resType'] not in ('javax.sql.DataSource', 'javax.sql.XADataSource', 'javax.sql.ConnectionPoolDataSource', 'java.sql.Driver'):
        raise CommandExecutionError('Invalid resource type')
    if data['connectionValidationMethod'] not in ('auto-commit', 'meta-data', 'table', 'custom-validation'):
        raise CommandExecutionError('Invalid connection validation method')
    if data['transactionIsolationLevel'] and data['transactionIsolationLevel'] not in ('read-uncommitted', 'read-committed', 'repeatable-read', 'serializable'):
        raise CommandExecutionError('Invalid transaction isolation level')
    if not data['datasourceClassname'] and data['resType'] in ('javax.sql.DataSource', 'javax.sql.ConnectionPoolDataSource', 'javax.sql.XADataSource'):
        raise CommandExecutionError('No datasource class name while using datasource resType')
    if not data['driverClassname'] and data['resType'] == 'java.sql.Driver':
        raise CommandExecutionError('No driver class nime while using driver resType')
    return _create_element(name, 'resources/jdbc-connection-pool', data, server)

def update_jdbc_connection_pool(name, server=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Update a JDBC pool\n    '
    return _update_element(name, 'resources/jdbc-connection-pool', kwargs, server)

def delete_jdbc_connection_pool(name, target='server', cascade=False, server=None):
    if False:
        i = 10
        return i + 15
    '\n    Delete a JDBC pool\n    '
    data = {'target': target, 'cascade': cascade}
    return _delete_element(name, 'resources/jdbc-connection-pool', data, server)

def enum_jdbc_resource(server=None):
    if False:
        return 10
    '\n    Enum JDBC resources\n    '
    return _enum_elements('resources/jdbc-resource', server)

def get_jdbc_resource(name, server=None):
    if False:
        i = 10
        return i + 15
    '\n    Get a specific JDBC resource\n    '
    return _get_element(name, 'resources/jdbc-resource', server)

def create_jdbc_resource(name, server=None, **kwargs):
    if False:
        return 10
    '\n    Create a JDBC resource\n    '
    defaults = {'description': '', 'enabled': True, 'id': name, 'poolName': '', 'target': 'server'}
    data = defaults
    data.update(kwargs)
    if not data['poolName']:
        raise CommandExecutionError('No pool name!')
    return _create_element(name, 'resources/jdbc-resource', data, server)

def update_jdbc_resource(name, server=None, **kwargs):
    if False:
        return 10
    '\n    Update a JDBC resource\n    '
    if 'jndiName' in kwargs:
        del kwargs['jndiName']
    return _update_element(name, 'resources/jdbc-resource', kwargs, server)

def delete_jdbc_resource(name, target='server', server=None):
    if False:
        while True:
            i = 10
    '\n    Delete a JDBC resource\n    '
    return _delete_element(name, 'resources/jdbc-resource', {'target': target}, server)

def get_system_properties(server=None):
    if False:
        return 10
    '\n    Get system properties\n    '
    properties = {}
    data = _api_get('system-properties', server)
    if any(data['extraProperties']['systemProperties']):
        for element in data['extraProperties']['systemProperties']:
            properties[element['name']] = element['value']
        return properties
    return {}

def update_system_properties(data, server=None):
    if False:
        i = 10
        return i + 15
    '\n    Update system properties\n    '
    _api_post('system-properties', _clean_data(data), server)
    return data

def delete_system_properties(name, server=None):
    if False:
        return 10
    '\n    Delete a system property\n    '
    _api_delete('system-properties/{}'.format(name), None, server)