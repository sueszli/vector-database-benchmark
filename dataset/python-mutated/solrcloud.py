"""
Module for solrcloud configuration

.. versionadded:: 2017.7.0

For now, module is limited to http-exposed API. It doesn't implement config upload via Solr zkCli
"""
import logging
import salt.utils.http as http
from salt.exceptions import SaltInvocationError
log = logging.getLogger(__name__)
'\nCore properties type definition.\nReference: https://cwiki.apache.org/confluence/display/solr/Defining+core.properties\n'
STRING_PROPS_LIST = ['config', 'schema', 'dataDir', 'configSet', 'properties', 'coreNodeName', 'ulogDir', 'shard', 'collection', 'roles']
BOOL_PROPS_LIST = ['transient', 'loadOnStartup']
'\nCollections options type definition\nReference: https://cwiki.apache.org/confluence/display/solr/Collections+API#CollectionsAPI-api1\n'
STRING_OPTIONS_LIST = ['collection.configName', 'router.field', 'async', 'rule', 'snitch']
INT_OPTIONS_LIST = ['numShards', 'replicationFactor', 'maxShardsPerNode']
BOOL_OPTIONS_LIST = ['autoAddReplicas', 'createNodeSet.shuffle']
LIST_OPTIONS_LIST = ['shards', 'createNodeSet']
DICT_OPTIONS_LIST = ['properties']
'\nCollection unmodifiable options\nReference: https://cwiki.apache.org/confluence/display/solr/Collections+API#CollectionsAPI-modifycoll\n'
CREATION_ONLY_OPTION = ['maxShardsPerNode', 'replicationFactor', 'autoAddReplicas', 'collection.configName', 'rule', 'snitch']

def __virtual__():
    if False:
        print('Hello World!')
    return 'solrcloud'

def _query(url, solr_url='http://localhost:8983/solr/', **kwargs):
    if False:
        i = 10
        return i + 15
    '\n\n    Internal function to query solrcloud\n\n    :param url: relative solr URL\n    :param solr_url: solr base URL\n    :param kwargs: additional args passed to http.query call\n    :return: Query JSON answer converted to python dict\n    :rtype: dict\n\n    '
    if not isinstance(solr_url, str):
        raise ValueError('solr_url must be a string')
    if solr_url[-1:] != '/':
        solr_url = solr_url + '/'
    query_result = http.query(solr_url + url, decode_type='json', decode=True, raise_error=True, **kwargs)
    if 'error' in query_result:
        if query_result['status'] == 404:
            raise SaltInvocationError('Got a 404 when trying to contact solr at {solr_url}{url}. Please check your solr URL.'.format(solr_url=solr_url, url=url))
        else:
            raise SaltInvocationError('Got a {status} error when calling {solr_url}{url} : {error}'.format(status=str(query_result['status']), solr_url=solr_url, url=url, error=query_result['error']))
    else:
        return query_result['dict']

def _validate_core_properties(properties):
    if False:
        while True:
            i = 10
    '\n\n    Internal function to validate core properties\n\n    '
    props_string = ''
    for (prop_name, prop_value) in properties.items():
        if prop_name in BOOL_PROPS_LIST:
            if not isinstance(prop_value, bool):
                raise ValueError('Option "' + prop_name + '" value must be an boolean')
            props_string = props_string + '&property.' + prop_name + '=' + ('true' if prop_value else 'false')
        elif prop_name in STRING_PROPS_LIST:
            if not isinstance(prop_value, str):
                raise ValueError('In option "properties", core property "' + prop_name + '" value must be a string')
            props_string = props_string + '&property.' + prop_name + '=' + prop_value
        else:
            props_string = props_string + '&property.' + str(prop_name) + '=' + str(prop_value)
    return props_string

def _validate_collection_options(options):
    if False:
        while True:
            i = 10
    '\n\n    Internal function to validate collections options\n\n    '
    options_string = ''
    for (option_name, option_value) in options.items():
        if option_name in STRING_OPTIONS_LIST:
            if not isinstance(option_value, str):
                raise ValueError('Option "' + option_name + '" value must be a string')
            options_string = options_string + '&' + option_name + '=' + option_value
        elif option_name in INT_OPTIONS_LIST:
            if not isinstance(option_value, int):
                raise ValueError('Option "' + option_name + '" value must be an int')
            options_string = options_string + '&' + option_name + '=' + str(option_value)
        elif option_name in BOOL_OPTIONS_LIST:
            if not isinstance(option_value, bool):
                raise ValueError('Option "' + option_name + '" value must be an boolean')
            options_string = options_string + '&' + option_name + '=' + ('true' if option_value else 'false')
        elif option_name in LIST_OPTIONS_LIST:
            if not isinstance(option_value, list):
                raise ValueError('Option "' + option_name + '" value must be a list of strings')
            options_string = options_string + '&' + option_name + '=' + ', '.join(option_value)
        elif option_name in DICT_OPTIONS_LIST:
            if not isinstance(option_value, dict):
                raise ValueError('Option "' + option_name + '" value must be an dict')
            options_string = options_string + _validate_core_properties(option_value)
        else:
            raise ValueError('Unknown option "' + option_name + '"')
    return options_string

def collection_creation_options():
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Get collection option list that can only be defined at creation\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solrcloud.collection_creation_options\n    "
    return CREATION_ONLY_OPTION

def cluster_status(**kwargs):
    if False:
        print('Hello World!')
    "\n\n    Get cluster status\n\n    Additional parameters (kwargs) may be passed, they will be proxied to http.query\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solrcloud.cluster_status\n    "
    return _query('admin/collections?action=CLUSTERSTATUS&wt=json', **kwargs)['cluster']

def alias_exists(alias_name, **kwargs):
    if False:
        print('Hello World!')
    "\n\n    Check alias existence\n\n    Additional parameters (kwargs) may be passed, they will be proxied to http.query\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solrcloud.alias_exists my_alias\n    "
    if not isinstance(alias_name, str):
        raise ValueError('Alias name must be a string')
    cluster = cluster_status(**kwargs)
    return 'aliases' in cluster and alias_name in cluster['aliases']

def alias_get_collections(alias_name, **kwargs):
    if False:
        return 10
    "\n\n    Get collection list for an alias\n\n    Additional parameters (kwargs) may be passed, they will be proxied to http.query\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solrcloud.alias_get my_alias\n    "
    if not isinstance(alias_name, str):
        raise ValueError('Alias name must be a string')
    collection_aliases = [(k_v[0], k_v[1]['aliases']) for k_v in cluster_status(**kwargs)['collections'].items() if 'aliases' in k_v[1]]
    aliases = [k_v1[0] for k_v1 in [k_v for k_v in collection_aliases if alias_name in k_v[1]]]
    return aliases

def alias_set_collections(alias_name, collections=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n\n    Define an alias\n\n    Additional parameters (kwargs) may be passed, they will be proxied to http.query\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solrcloud.alias_set my_alias collections=[collection1, colletion2]\n    "
    if not isinstance(collections, list):
        raise SaltInvocationError('Collection parameter must be defined and contain a list of collection name')
    for collection in collections:
        if not isinstance(collection, str):
            raise ValueError('Collection name must be a string')
    return _query('admin/collections?action=CREATEALIAS&name={alias}&wt=json&collections={collections}'.format(alias=alias_name, collections=', '.join(collections)), **kwargs)

def collection_reload(collection, **kwargs):
    if False:
        print('Hello World!')
    "\n\n    Check if a collection exists\n\n    Additional parameters (kwargs) may be passed, they will be proxied to http.query\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solrcloud.collection_reload collection_name\n\n    "
    _query('admin/collections?action=RELOAD&name={collection}&wt=json'.format(collection=collection), **kwargs)

def collection_list(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    List all collections\n\n    Additional parameters (kwargs) may be passed, they will be proxied to http.query\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solrcloud.collection_list\n\n    "
    return _query('admin/collections?action=LIST&wt=json', **kwargs)['collections']

def collection_exists(collection_name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Check if a collection exists\n\n    Additional parameters (kwargs) may be passed, they will be proxied to http.query\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solrcloud.collection_exists collection_name\n\n    "
    if not isinstance(collection_name, str):
        raise ValueError('Collection name must be a string')
    return collection_name in collection_list(**kwargs)

def collection_backup(collection_name, location, backup_name=None, **kwargs):
    if False:
        print('Hello World!')
    "\n\n    Create a backup for a collection.\n\n    Additional parameters (kwargs) may be passed, they will be proxied to http.query\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solrcloud.core_backup collection_name /mnt/nfs_backup\n    "
    if not collection_exists(collection_name, **kwargs):
        raise ValueError("Collection doesn't exists")
    if backup_name is not None:
        backup_name = '&name={}'.format(backup_name)
    else:
        backup_name = ''
    _query('{collection}/replication?command=BACKUP&location={location}{backup_name}&wt=json'.format(collection=collection_name, backup_name=backup_name, location=location), **kwargs)

def collection_backup_all(location, backup_name=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n\n    Create a backup for all collection present on the server.\n\n    Additional parameters (kwargs) may be passed, they will be proxied to http.query\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solrcloud.core_backup /mnt/nfs_backup\n    "
    for collection_name in collection_list(**kwargs):
        if backup_name is not None:
            backup_name = '&name={backup}.{collection}'.format(backup=backup_name, collection=collection_name)
        else:
            backup_name = ''
        _query('{collection}/replication?command=BACKUP&location={location}{backup_name}&wt=json'.format(collection=collection_name, backup_name=backup_name, location=location), **kwargs)

def collection_create(collection_name, options=None, **kwargs):
    if False:
        return 10
    '\n\n    Create a collection,\n\n    Additional parameters (kwargs) may be passed, they will be proxied to http.query\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' solrcloud.collection_create collection_name\n\n    Collection creation options may be passed using the "options" parameter.\n    Do not include option "name" since it already specified by the mandatory parameter "collection_name"\n\n    .. code-block:: bash\n\n        salt \'*\' solrcloud.collection_create collection_name options={"replicationFactor":2, "numShards":3}\n\n    Cores options may be passed using the "properties" key in options.\n    Do not include property "name"\n\n    .. code-block:: bash\n\n        salt \'*\' solrcloud.collection_create collection_name options={"replicationFactor":2, "numShards":3,             "properties":{"dataDir":"/srv/solr/hugePartitionSollection"}}\n    '
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise SaltInvocationError('options parameter must be a dictionary')
    options_string = _validate_collection_options(options)
    _query('admin/collections?action=CREATE&wt=json&name=' + collection_name + options_string, **kwargs)

def collection_check_options(options):
    if False:
        print('Hello World!')
    '\n    Check collections options\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' solrcloud.collection_check_options \'{"replicationFactor":4}\'\n    '
    try:
        _validate_collection_options(options)
        return True
    except ValueError:
        return False

def collection_get_options(collection_name, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get collection options\n\n    Additional parameters (kwargs) may be passed, they will be proxied to http.query\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solrcloud.collection_get_options collection_name\n    "
    cluster = cluster_status(**kwargs)
    options = {'collection.configName': cluster['collections'][collection_name]['configName'], 'router.name': cluster['collections'][collection_name]['router']['name'], 'replicationFactor': int(cluster['collections'][collection_name]['replicationFactor']), 'maxShardsPerNode': int(cluster['collections'][collection_name]['maxShardsPerNode']), 'autoAddReplicas': cluster['collections'][collection_name]['autoAddReplicas'] is True}
    if 'rule' in cluster['collections'][collection_name]:
        options['rule'] = cluster['collections'][collection_name]['rule']
    if 'snitch' in cluster['collections'][collection_name]:
        options['snitch'] = cluster['collections'][collection_name]['rule']
    return options

def collection_set_options(collection_name, options, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Change collection options\n\n    Additional parameters (kwargs) may be passed, they will be proxied to http.query\n\n    Note that not every parameter can be changed after collection creation\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' solrcloud.collection_set_options collection_name options={"replicationFactor":4}\n    '
    for option in list(options.keys()):
        if option not in CREATION_ONLY_OPTION:
            raise ValueError('Option ' + option + " can't be modified after collection creation.")
    options_string = _validate_collection_options(options)
    _query('admin/collections?action=MODIFYCOLLECTION&wt=json&collection=' + collection_name + options_string, **kwargs)