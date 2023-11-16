"""
Apache Solr Salt Module

Author: Jed Glazner
Version: 0.2.1
Modified: 12/09/2011

This module uses HTTP requests to talk to the apache solr request handlers
to gather information and report errors. Because of this the minion doesn't
necessarily need to reside on the actual slave.  However if you want to
use the signal function the minion must reside on the physical solr host.

This module supports multi-core and standard setups.  Certain methods are
master/slave specific.  Make sure you set the solr.type. If you have
questions or want a feature request please ask.

Coming Features in 0.3
----------------------

1. Add command for checking for replication failures on slaves
2. Improve match_index_versions since it's pointless on busy solr masters
3. Add additional local fs checks for backups to make sure they succeeded

Override these in the minion config
-----------------------------------

solr.cores
    A list of core names e.g. ['core1','core2'].
    An empty list indicates non-multicore setup.
solr.baseurl
    The root level URL to access solr via HTTP
solr.request_timeout
    The number of seconds before timing out an HTTP/HTTPS/FTP request. If
    nothing is specified then the python global timeout setting is used.
solr.type
    Possible values are 'master' or 'slave'
solr.backup_path
    The path to store your backups. If you are using cores and you can specify
    to append the core name to the path in the backup method.
solr.num_backups
    For versions of solr >= 3.5. Indicates the number of backups to keep. This
    option is ignored if your version is less.
solr.init_script
    The full path to your init script with start/stop options
solr.dih.options
    A list of options to pass to the DIH.

Required Options for DIH
------------------------

clean : False
    Clear the index before importing
commit : True
    Commit the documents to the index upon completion
optimize : True
    Optimize the index after commit is complete
verbose : True
    Get verbose output
"""
import os
import urllib.request
import salt.utils.json
import salt.utils.path

def __virtual__():
    if False:
        print('Hello World!')
    '\n    PRIVATE METHOD\n    Solr needs to be installed to use this.\n\n    Return: str/bool\n    '
    if salt.utils.path.which('solr'):
        return 'solr'
    if salt.utils.path.which('apache-solr'):
        return 'solr'
    return (False, 'The solr execution module failed to load: requires both the solr and apache-solr binaries in the path.')

def _get_none_or_value(value):
    if False:
        i = 10
        return i + 15
    '\n    PRIVATE METHOD\n    Checks to see if the value of a primitive or built-in container such as\n    a list, dict, set, tuple etc is empty or none. None type is returned if the\n    value is empty/None/False. Number data types that are 0 will return None.\n\n    value : obj\n        The primitive or built-in container to evaluate.\n\n    Return: None or value\n    '
    if value is None:
        return None
    elif not value:
        return value
    elif isinstance(value, str):
        if value.lower() == 'none':
            return None
        return value
    else:
        return None

def _check_for_cores():
    if False:
        while True:
            i = 10
    "\n    PRIVATE METHOD\n    Checks to see if using_cores has been set or not. if it's been set\n    return it, otherwise figure it out and set it. Then return it\n\n    Return: boolean\n\n        True if one or more cores defined in __opts__['solr.cores']\n    "
    return len(__salt__['config.option']('solr.cores')) > 0

def _get_return_dict(success=True, data=None, errors=None, warnings=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    PRIVATE METHOD\n    Creates a new return dict with default values. Defaults may be overwritten.\n\n    success : boolean (True)\n        True indicates a successful result.\n    data : dict<str,obj> ({})\n        Data to be returned to the caller.\n    errors : list<str> ([()])\n        A list of error messages to be returned to the caller\n    warnings : list<str> ([])\n        A list of warnings to be returned to the caller.\n\n    Return: dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n    "
    data = {} if data is None else data
    errors = [] if errors is None else errors
    warnings = [] if warnings is None else warnings
    ret = {'success': success, 'data': data, 'errors': errors, 'warnings': warnings}
    return ret

def _update_return_dict(ret, success, data, errors=None, warnings=None):
    if False:
        while True:
            i = 10
    "\n    PRIVATE METHOD\n    Updates the return dictionary and returns it.\n\n    ret : dict<str,obj>\n        The original return dict to update. The ret param should have\n        been created from _get_return_dict()\n    success : boolean (True)\n        True indicates a successful result.\n    data : dict<str,obj> ({})\n        Data to be returned to the caller.\n    errors : list<str> ([()])\n        A list of error messages to be returned to the caller\n    warnings : list<str> ([])\n        A list of warnings to be returned to the caller.\n\n    Return: dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n    "
    errors = [] if errors is None else errors
    warnings = [] if warnings is None else warnings
    ret['success'] = success
    ret['data'].update(data)
    ret['errors'] = ret['errors'] + errors
    ret['warnings'] = ret['warnings'] + warnings
    return ret

def _format_url(handler, host=None, core_name=None, extra=None):
    if False:
        return 10
    "\n    PRIVATE METHOD\n    Formats the URL based on parameters, and if cores are used or not\n\n    handler : str\n        The request handler to hit.\n    host : str (None)\n        The solr host to query. __opts__['host'] is default\n    core_name : str (None)\n        The name of the solr core if using cores. Leave this blank if you\n        are not using cores or if you want to check all cores.\n    extra : list<str> ([])\n        A list of name value pairs in string format. e.g. ['name=value']\n\n    Return: str\n        Fully formatted URL (http://<host>:<port>/solr/<handler>?wt=json&<extra>)\n    "
    extra = [] if extra is None else extra
    if _get_none_or_value(host) is None or host == 'None':
        host = __salt__['config.option']('solr.host')
    port = __salt__['config.option']('solr.port')
    baseurl = __salt__['config.option']('solr.baseurl')
    if _get_none_or_value(core_name) is None:
        if extra is None or len(extra) == 0:
            return 'http://{}:{}{}/{}?wt=json'.format(host, port, baseurl, handler)
        else:
            return 'http://{}:{}{}/{}?wt=json&{}'.format(host, port, baseurl, handler, '&'.join(extra))
    elif extra is None or len(extra) == 0:
        return 'http://{}:{}{}/{}/{}?wt=json'.format(host, port, baseurl, core_name, handler)
    else:
        return 'http://{}:{}{}/{}/{}?wt=json&{}'.format(host, port, baseurl, core_name, handler, '&'.join(extra))

def _auth(url):
    if False:
        while True:
            i = 10
    '\n    Install an auth handler for urllib2\n    '
    user = __salt__['config.get']('solr.user', False)
    password = __salt__['config.get']('solr.passwd', False)
    realm = __salt__['config.get']('solr.auth_realm', 'Solr')
    if user and password:
        basic = urllib.request.HTTPBasicAuthHandler()
        basic.add_password(realm=realm, uri=url, user=user, passwd=password)
        digest = urllib.request.HTTPDigestAuthHandler()
        digest.add_password(realm=realm, uri=url, user=user, passwd=password)
        urllib.request.install_opener(urllib.request.build_opener(basic, digest))

def _http_request(url, request_timeout=None):
    if False:
        while True:
            i = 10
    "\n    PRIVATE METHOD\n    Uses salt.utils.json.load to fetch the JSON results from the solr API.\n\n    url : str\n        a complete URL that can be passed to urllib.open\n    request_timeout : int (None)\n        The number of seconds before the timeout should fail. Leave blank/None\n        to use the default. __opts__['solr.request_timeout']\n\n    Return: dict<str,obj>::\n\n         {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n    "
    _auth(url)
    try:
        request_timeout = __salt__['config.option']('solr.request_timeout')
        kwargs = {} if request_timeout is None else {'timeout': request_timeout}
        data = salt.utils.json.load(urllib.request.urlopen(url, **kwargs))
        return _get_return_dict(True, data, [])
    except Exception as err:
        return _get_return_dict(False, {}, ['{} : {}'.format(url, err)])

def _replication_request(command, host=None, core_name=None, params=None):
    if False:
        return 10
    "\n    PRIVATE METHOD\n    Performs the requested replication command and returns a dictionary with\n    success, errors and data as keys. The data object will contain the JSON\n    response.\n\n    command : str\n        The replication command to execute.\n    host : str (None)\n        The solr host to query. __opts__['host'] is default\n    core_name: str (None)\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to check all cores.\n    params : list<str> ([])\n        Any additional parameters you want to send. Should be a lsit of\n        strings in name=value format. e.g. ['name=value']\n\n    Return: dict<str, obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n    "
    params = [] if params is None else params
    extra = ['command={}'.format(command)] + params
    url = _format_url('replication', host=host, core_name=core_name, extra=extra)
    return _http_request(url)

def _get_admin_info(command, host=None, core_name=None):
    if False:
        i = 10
        return i + 15
    "\n    PRIVATE METHOD\n    Calls the _http_request method and passes the admin command to execute\n    and stores the data. This data is fairly static but should be refreshed\n    periodically to make sure everything this OK. The data object will contain\n    the JSON response.\n\n    command : str\n        The admin command to execute.\n    host : str (None)\n        The solr host to query. __opts__['host'] is default\n    core_name: str (None)\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to check all cores.\n\n    Return: dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n    "
    url = _format_url('admin/{}'.format(command), host, core_name=core_name)
    resp = _http_request(url)
    return resp

def _is_master():
    if False:
        i = 10
        return i + 15
    "\n    PRIVATE METHOD\n    Simple method to determine if the minion is configured as master or slave\n\n    Return: boolean::\n\n        True if __opts__['solr.type'] = master\n    "
    return __salt__['config.option']('solr.type') == 'master'

def _merge_options(options):
    if False:
        print('Hello World!')
    "\n    PRIVATE METHOD\n    updates the default import options from __opts__['solr.dih.import_options']\n    with the dictionary passed in.  Also converts booleans to strings\n    to pass to solr.\n\n    options : dict<str,boolean>\n        Dictionary the over rides the default options defined in\n        __opts__['solr.dih.import_options']\n\n    Return: dict<str,boolean>::\n\n        {option:boolean}\n    "
    defaults = __salt__['config.option']('solr.dih.import_options')
    if isinstance(options, dict):
        defaults.update(options)
    for (key, val) in defaults.items():
        if isinstance(val, bool):
            defaults[key] = str(val).lower()
    return defaults

def _pre_index_check(handler, host=None, core_name=None):
    if False:
        while True:
            i = 10
    "\n    PRIVATE METHOD - MASTER CALL\n    Does a pre-check to make sure that all the options are set and that\n    we can talk to solr before trying to send a command to solr. This\n    Command should only be issued to masters.\n\n    handler : str\n        The import handler to check the state of\n    host : str (None):\n        The solr host to query. __opts__['host'] is default\n    core_name (None):\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to check all cores.\n        REQUIRED if you are using cores.\n\n    Return:  dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n    "
    if _get_none_or_value(host) is None and (not _is_master()):
        err = ['solr.pre_indexing_check can only be called by "master" minions']
        return _get_return_dict(False, err)
    if _get_none_or_value(core_name) is None and _check_for_cores():
        errors = ['solr.full_import is not safe to multiple handlers at once']
        return _get_return_dict(False, errors=errors)
    resp = import_status(handler, host, core_name)
    if resp['success']:
        status = resp['data']['status']
        if status == 'busy':
            warn = ['An indexing process is already running.']
            return _get_return_dict(True, warnings=warn)
        if status != 'idle':
            errors = ['Unknown status: "{}"'.format(status)]
            return _get_return_dict(False, data=resp['data'], errors=errors)
    else:
        errors = ['Status check failed. Response details: {}'.format(resp)]
        return _get_return_dict(False, data=resp['data'], errors=errors)
    return resp

def _find_value(ret_dict, key, path=None):
    if False:
        print('Hello World!')
    "\n    PRIVATE METHOD\n    Traverses a dictionary of dictionaries/lists to find key\n    and return the value stored.\n    TODO:// this method doesn't really work very well, and it's not really\n            very useful in its current state. The purpose for this method is\n            to simplify parsing the JSON output so you can just pass the key\n            you want to find and have it return the value.\n    ret : dict<str,obj>\n        The dictionary to search through. Typically this will be a dict\n        returned from solr.\n    key : str\n        The key (str) to find in the dictionary\n\n    Return: list<dict<str,obj>>::\n\n        [{path:path, value:value}]\n    "
    if path is None:
        path = key
    else:
        path = '{}:{}'.format(path, key)
    ret = []
    for (ikey, val) in ret_dict.items():
        if ikey == key:
            ret.append({path: val})
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    ret = ret + _find_value(item, key, path)
        if isinstance(val, dict):
            ret = ret + _find_value(val, key, path)
    return ret

def lucene_version(core_name=None):
    if False:
        return 10
    "\n    Gets the lucene version that solr is using. If you are running a multi-core\n    setup you should specify a core name since all the cores run under the same\n    servlet container, they will all have the same version.\n\n    core_name : str (None)\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to check all cores.\n\n    Return: dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.lucene_version\n    "
    ret = _get_return_dict()
    if _get_none_or_value(core_name) is None and _check_for_cores():
        success = True
        for name in __salt__['config.option']('solr.cores'):
            resp = _get_admin_info('system', core_name=name)
            if resp['success']:
                version_num = resp['data']['lucene']['lucene-spec-version']
                data = {name: {'lucene_version': version_num}}
            else:
                data = {name: {'lucene_version': None}}
                success = False
            ret = _update_return_dict(ret, success, data, resp['errors'])
        return ret
    else:
        resp = _get_admin_info('system', core_name=core_name)
        if resp['success']:
            version_num = resp['data']['lucene']['lucene-spec-version']
            return _get_return_dict(True, {'version': version_num}, resp['errors'])
        else:
            return resp

def version(core_name=None):
    if False:
        while True:
            i = 10
    "\n    Gets the solr version for the core specified.  You should specify a core\n    here as all the cores will run under the same servlet container and so will\n    all have the same version.\n\n    core_name : str (None)\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to check all cores.\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.version\n    "
    ret = _get_return_dict()
    if _get_none_or_value(core_name) is None and _check_for_cores():
        success = True
        for name in __opts__['solr.cores']:
            resp = _get_admin_info('system', core_name=name)
            if resp['success']:
                lucene = resp['data']['lucene']
                data = {name: {'version': lucene['solr-spec-version']}}
            else:
                success = False
                data = {name: {'version': None}}
            ret = _update_return_dict(ret, success, data, resp['errors'], resp['warnings'])
        return ret
    else:
        resp = _get_admin_info('system', core_name=core_name)
        if resp['success']:
            version_num = resp['data']['lucene']['solr-spec-version']
            return _get_return_dict(True, {'version': version_num}, resp['errors'], resp['warnings'])
        else:
            return resp

def optimize(host=None, core_name=None):
    if False:
        print('Hello World!')
    "\n    Search queries fast, but it is a very expensive operation. The ideal\n    process is to run this with a master/slave configuration.  Then you\n    can optimize the master, and push the optimized index to the slaves.\n    If you are running a single solr instance, or if you are going to run\n    this on a slave be aware than search performance will be horrible\n    while this command is being run. Additionally it can take a LONG time\n    to run and your HTTP request may timeout. If that happens adjust your\n    timeout settings.\n\n    host : str (None)\n        The solr host to query. __opts__['host'] is default.\n    core_name : str (None)\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to check all cores.\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.optimize music\n    "
    ret = _get_return_dict()
    if _get_none_or_value(core_name) is None and _check_for_cores():
        success = True
        for name in __salt__['config.option']('solr.cores'):
            url = _format_url('update', host=host, core_name=name, extra=['optimize=true'])
            resp = _http_request(url)
            if resp['success']:
                data = {name: {'data': resp['data']}}
                ret = _update_return_dict(ret, success, data, resp['errors'], resp['warnings'])
            else:
                success = False
                data = {name: {'data': resp['data']}}
                ret = _update_return_dict(ret, success, data, resp['errors'], resp['warnings'])
        return ret
    else:
        url = _format_url('update', host=host, core_name=core_name, extra=['optimize=true'])
        return _http_request(url)

def ping(host=None, core_name=None):
    if False:
        return 10
    "\n    Does a health check on solr, makes sure solr can talk to the indexes.\n\n    host : str (None)\n        The solr host to query. __opts__['host'] is default.\n    core_name : str (None)\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to check all cores.\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.ping music\n    "
    ret = _get_return_dict()
    if _get_none_or_value(core_name) is None and _check_for_cores():
        success = True
        for name in __opts__['solr.cores']:
            resp = _get_admin_info('ping', host=host, core_name=name)
            if resp['success']:
                data = {name: {'status': resp['data']['status']}}
            else:
                success = False
                data = {name: {'status': None}}
            ret = _update_return_dict(ret, success, data, resp['errors'])
        return ret
    else:
        resp = _get_admin_info('ping', host=host, core_name=core_name)
        return resp

def is_replication_enabled(host=None, core_name=None):
    if False:
        i = 10
        return i + 15
    "\n    SLAVE CALL\n    Check for errors, and determine if a slave is replicating or not.\n\n    host : str (None)\n        The solr host to query. __opts__['host'] is default.\n    core_name : str (None)\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to check all cores.\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.is_replication_enabled music\n    "
    ret = _get_return_dict()
    success = True
    if _is_master() and host is None:
        errors = ['Only "slave" minions can run "is_replication_enabled"']
        return ret.update({'success': False, 'errors': errors})

    def _checks(ret, success, resp, core):
        if False:
            return 10
        if response['success']:
            slave = resp['data']['details']['slave']
            enabled = 'false'
            master_url = slave['masterUrl']
            if 'ERROR' in slave:
                success = False
                err = '{}: {} - {}'.format(core, slave['ERROR'], master_url)
                resp['errors'].append(err)
                data = slave if core is None else {core: {'data': slave}}
            else:
                enabled = slave['masterDetails']['master']['replicationEnabled']
            if enabled == 'false':
                resp['warnings'].append('Replication is disabled on master.')
                success = False
            if slave['isPollingDisabled'] == 'true':
                success = False
                resp['warning'].append('Polling is disabled')
            ret = _update_return_dict(ret, success, data, resp['errors'], resp['warnings'])
        return (ret, success)
    if _get_none_or_value(core_name) is None and _check_for_cores():
        for name in __opts__['solr.cores']:
            response = _replication_request('details', host=host, core_name=name)
            (ret, success) = _checks(ret, success, response, name)
    else:
        response = _replication_request('details', host=host, core_name=core_name)
        (ret, success) = _checks(ret, success, response, core_name)
    return ret

def match_index_versions(host=None, core_name=None):
    if False:
        i = 10
        return i + 15
    "\n    SLAVE CALL\n    Verifies that the master and the slave versions are in sync by\n    comparing the index version. If you are constantly pushing updates\n    the index the master and slave versions will seldom match. A solution\n    to this is pause indexing every so often to allow the slave to replicate\n    and then call this method before allowing indexing to resume.\n\n    host : str (None)\n        The solr host to query. __opts__['host'] is default.\n    core_name : str (None)\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to check all cores.\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.match_index_versions music\n    "
    ret = _get_return_dict()
    success = True
    if _is_master() and _get_none_or_value(host) is None:
        return ret.update({'success': False, 'errors': ['solr.match_index_versions can only be called by "slave" minions']})

    def _match(ret, success, resp, core):
        if False:
            print('Hello World!')
        if response['success']:
            slave = resp['data']['details']['slave']
            master_url = resp['data']['details']['slave']['masterUrl']
            if 'ERROR' in slave:
                error = slave['ERROR']
                success = False
                err = '{}: {} - {}'.format(core, error, master_url)
                resp['errors'].append(err)
                data = slave if core is None else {core: {'data': slave}}
            else:
                versions = {'master': slave['masterDetails']['master']['replicatableIndexVersion'], 'slave': resp['data']['details']['indexVersion'], 'next_replication': slave['nextExecutionAt'], 'failed_list': []}
                if 'replicationFailedAtList' in slave:
                    versions.update({'failed_list': slave['replicationFailedAtList']})
                if versions['master'] != versions['slave']:
                    success = False
                    resp['errors'].append('Master and Slave index versions do not match.')
                data = versions if core is None else {core: {'data': versions}}
            ret = _update_return_dict(ret, success, data, resp['errors'], resp['warnings'])
        else:
            success = False
            err = resp['errors']
            data = resp['data']
            ret = _update_return_dict(ret, success, data, errors=err)
        return (ret, success)
    if _get_none_or_value(core_name) is None and _check_for_cores():
        success = True
        for name in __opts__['solr.cores']:
            response = _replication_request('details', host=host, core_name=name)
            (ret, success) = _match(ret, success, response, name)
    else:
        response = _replication_request('details', host=host, core_name=core_name)
        (ret, success) = _match(ret, success, response, core_name)
    return ret

def replication_details(host=None, core_name=None):
    if False:
        print('Hello World!')
    "\n    Get the full replication details.\n\n    host : str (None)\n        The solr host to query. __opts__['host'] is default.\n    core_name : str (None)\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to check all cores.\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.replication_details music\n    "
    ret = _get_return_dict()
    if _get_none_or_value(core_name) is None:
        success = True
        for name in __opts__['solr.cores']:
            resp = _replication_request('details', host=host, core_name=name)
            data = {name: {'data': resp['data']}}
            ret = _update_return_dict(ret, success, data, resp['errors'], resp['warnings'])
    else:
        resp = _replication_request('details', host=host, core_name=core_name)
        if resp['success']:
            ret = _update_return_dict(ret, resp['success'], resp['data'], resp['errors'], resp['warnings'])
        else:
            return resp
    return ret

def backup(host=None, core_name=None, append_core_to_path=False):
    if False:
        print('Hello World!')
    "\n    Tell solr make a backup.  This method can be mis-leading since it uses the\n    backup API.  If an error happens during the backup you are not notified.\n    The status: 'OK' in the response simply means that solr received the\n    request successfully.\n\n    host : str (None)\n        The solr host to query. __opts__['host'] is default.\n    core_name : str (None)\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to check all cores.\n    append_core_to_path : boolean (False)\n        If True add the name of the core to the backup path. Assumes that\n        minion backup path is not None.\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.backup music\n    "
    path = __opts__['solr.backup_path']
    num_backups = __opts__['solr.num_backups']
    if path is not None:
        if not path.endswith(os.path.sep):
            path += os.path.sep
    ret = _get_return_dict()
    if _get_none_or_value(core_name) is None and _check_for_cores():
        success = True
        for name in __opts__['solr.cores']:
            params = []
            if path is not None:
                path = path + name if append_core_to_path else path
                params.append('&location={}'.format(path + name))
            params.append('&numberToKeep={}'.format(num_backups))
            resp = _replication_request('backup', host=host, core_name=name, params=params)
            if not resp['success']:
                success = False
            data = {name: {'data': resp['data']}}
            ret = _update_return_dict(ret, success, data, resp['errors'], resp['warnings'])
        return ret
    else:
        if core_name is not None and path is not None:
            if append_core_to_path:
                path += core_name
        if path is not None:
            params = ['location={}'.format(path)]
        params.append('&numberToKeep={}'.format(num_backups))
        resp = _replication_request('backup', host=host, core_name=core_name, params=params)
        return resp

def set_is_polling(polling, host=None, core_name=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    SLAVE CALL\n    Prevent the slaves from polling the master for updates.\n\n    polling : boolean\n        True will enable polling. False will disable it.\n    host : str (None)\n        The solr host to query. __opts__['host'] is default.\n    core_name : str (None)\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to check all cores.\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.set_is_polling False\n    "
    ret = _get_return_dict()
    if _is_master() and _get_none_or_value(host) is None:
        err = ['solr.set_is_polling can only be called by "slave" minions']
        return ret.update({'success': False, 'errors': err})
    cmd = 'enablepoll' if polling else 'disapblepoll'
    if _get_none_or_value(core_name) is None and _check_for_cores():
        success = True
        for name in __opts__['solr.cores']:
            resp = set_is_polling(cmd, host=host, core_name=name)
            if not resp['success']:
                success = False
            data = {name: {'data': resp['data']}}
            ret = _update_return_dict(ret, success, data, resp['errors'], resp['warnings'])
        return ret
    else:
        resp = _replication_request(cmd, host=host, core_name=core_name)
        return resp

def set_replication_enabled(status, host=None, core_name=None):
    if False:
        return 10
    "\n    MASTER ONLY\n    Sets the master to ignore poll requests from the slaves. Useful when you\n    don't want the slaves replicating during indexing or when clearing the\n    index.\n\n    status : boolean\n        Sets the replication status to the specified state.\n    host : str (None)\n        The solr host to query. __opts__['host'] is default.\n\n    core_name : str (None)\n        The name of the solr core if using cores. Leave this blank if you are\n        not using cores or if you want to set the status on all cores.\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.set_replication_enabled false, None, music\n    "
    if not _is_master() and _get_none_or_value(host) is None:
        return _get_return_dict(False, errors=['Only minions configured as master can run this'])
    cmd = 'enablereplication' if status else 'disablereplication'
    if _get_none_or_value(core_name) is None and _check_for_cores():
        ret = _get_return_dict()
        success = True
        for name in __opts__['solr.cores']:
            resp = set_replication_enabled(status, host, name)
            if not resp['success']:
                success = False
            data = {name: {'data': resp['data']}}
            ret = _update_return_dict(ret, success, data, resp['errors'], resp['warnings'])
        return ret
    elif status:
        return _replication_request(cmd, host=host, core_name=core_name)
    else:
        return _replication_request(cmd, host=host, core_name=core_name)

def signal(signal=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Signals Apache Solr to start, stop, or restart. Obviously this is only\n    going to work if the minion resides on the solr host. Additionally Solr\n    doesn't ship with an init script so one must be created.\n\n    signal : str (None)\n        The command to pass to the apache solr init valid values are 'start',\n        'stop', and 'restart'\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.signal restart\n    "
    valid_signals = ('start', 'stop', 'restart')
    if signal not in valid_signals:
        return '{} is an invalid signal. Try: one of: {} or {}'.format(signal, ', '.join(valid_signals[:-1]), valid_signals[-1])
    cmd = '{} {}'.format(__opts__['solr.init_script'], signal)
    __salt__['cmd.run'](cmd, python_shell=False)

def reload_core(host=None, core_name=None):
    if False:
        i = 10
        return i + 15
    '\n    MULTI-CORE HOSTS ONLY\n    Load a new core from the same configuration as an existing registered core.\n    While the "new" core is initializing, the "old" one will continue to accept\n    requests. Once it has finished, all new request will go to the "new" core,\n    and the "old" core will be unloaded.\n\n    host : str (None)\n        The solr host to query. __opts__[\'host\'] is default.\n    core_name : str\n        The name of the core to reload\n\n    Return : dict<str,obj>::\n\n        {\'success\':boolean, \'data\':dict, \'errors\':list, \'warnings\':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' solr.reload_core None music\n\n    Return data is in the following format::\n\n        {\'success\':bool, \'data\':dict, \'errors\':list, \'warnings\':list}\n    '
    ret = _get_return_dict()
    if not _check_for_cores():
        err = ['solr.reload_core can only be called by "multi-core" minions']
        return ret.update({'success': False, 'errors': err})
    if _get_none_or_value(core_name) is None and _check_for_cores():
        success = True
        for name in __opts__['solr.cores']:
            resp = reload_core(host, name)
            if not resp['success']:
                success = False
            data = {name: {'data': resp['data']}}
            ret = _update_return_dict(ret, success, data, resp['errors'], resp['warnings'])
        return ret
    extra = ['action=RELOAD', 'core={}'.format(core_name)]
    url = _format_url('admin/cores', host=host, core_name=None, extra=extra)
    return _http_request(url)

def core_status(host=None, core_name=None):
    if False:
        print('Hello World!')
    "\n    MULTI-CORE HOSTS ONLY\n    Get the status for a given core or all cores if no core is specified\n\n    host : str (None)\n        The solr host to query. __opts__['host'] is default.\n    core_name : str\n        The name of the core to reload\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.core_status None music\n    "
    ret = _get_return_dict()
    if not _check_for_cores():
        err = ['solr.reload_core can only be called by "multi-core" minions']
        return ret.update({'success': False, 'errors': err})
    if _get_none_or_value(core_name) is None and _check_for_cores():
        success = True
        for name in __opts__['solr.cores']:
            resp = reload_core(host, name)
            if not resp['success']:
                success = False
            data = {name: {'data': resp['data']}}
            ret = _update_return_dict(ret, success, data, resp['errors'], resp['warnings'])
        return ret
    extra = ['action=STATUS', 'core={}'.format(core_name)]
    url = _format_url('admin/cores', host=host, core_name=None, extra=extra)
    return _http_request(url)

def reload_import_config(handler, host=None, core_name=None, verbose=False):
    if False:
        print('Hello World!')
    "\n    MASTER ONLY\n    re-loads the handler config XML file.\n    This command can only be run if the minion is a 'master' type\n\n    handler : str\n        The name of the data import handler.\n    host : str (None)\n        The solr host to query. __opts__['host'] is default.\n    core : str (None)\n        The core the handler belongs to.\n    verbose : boolean (False)\n        Run the command with verbose output.\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.reload_import_config dataimport None music {'clean':True}\n    "
    if not _is_master() and _get_none_or_value(host) is None:
        err = ['solr.pre_indexing_check can only be called by "master" minions']
        return _get_return_dict(False, err)
    if _get_none_or_value(core_name) is None and _check_for_cores():
        err = ['No core specified when minion is configured as "multi-core".']
        return _get_return_dict(False, err)
    params = ['command=reload-config']
    if verbose:
        params.append('verbose=true')
    url = _format_url(handler, host=host, core_name=core_name, extra=params)
    return _http_request(url)

def abort_import(handler, host=None, core_name=None, verbose=False):
    if False:
        return 10
    "\n    MASTER ONLY\n    Aborts an existing import command to the specified handler.\n    This command can only be run if the minion is configured with\n    solr.type=master\n\n    handler : str\n        The name of the data import handler.\n    host : str (None)\n        The solr host to query. __opts__['host'] is default.\n    core : str (None)\n        The core the handler belongs to.\n    verbose : boolean (False)\n        Run the command with verbose output.\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.abort_import dataimport None music {'clean':True}\n    "
    if not _is_master() and _get_none_or_value(host) is None:
        err = ['solr.abort_import can only be called on "master" minions']
        return _get_return_dict(False, errors=err)
    if _get_none_or_value(core_name) is None and _check_for_cores():
        err = ['No core specified when minion is configured as "multi-core".']
        return _get_return_dict(False, err)
    params = ['command=abort']
    if verbose:
        params.append('verbose=true')
    url = _format_url(handler, host=host, core_name=core_name, extra=params)
    return _http_request(url)

def full_import(handler, host=None, core_name=None, options=None, extra=None):
    if False:
        return 10
    '\n    MASTER ONLY\n    Submits an import command to the specified handler using specified options.\n    This command can only be run if the minion is configured with\n    solr.type=master\n\n    handler : str\n        The name of the data import handler.\n    host : str (None)\n        The solr host to query. __opts__[\'host\'] is default.\n    core : str (None)\n        The core the handler belongs to.\n    options : dict (__opts__)\n        A list of options such as clean, optimize commit, verbose, and\n        pause_replication. leave blank to use __opts__ defaults. options will\n        be merged with __opts__\n    extra : dict ([])\n        Extra name value pairs to pass to the handler. e.g. ["name=value"]\n\n    Return : dict<str,obj>::\n\n        {\'success\':boolean, \'data\':dict, \'errors\':list, \'warnings\':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' solr.full_import dataimport None music {\'clean\':True}\n    '
    options = {} if options is None else options
    extra = [] if extra is None else extra
    if not _is_master():
        err = ['solr.full_import can only be called on "master" minions']
        return _get_return_dict(False, errors=err)
    if _get_none_or_value(core_name) is None and _check_for_cores():
        err = ['No core specified when minion is configured as "multi-core".']
        return _get_return_dict(False, err)
    resp = _pre_index_check(handler, host, core_name)
    if not resp['success']:
        return resp
    options = _merge_options(options)
    if options['clean']:
        resp = set_replication_enabled(False, host=host, core_name=core_name)
        if not resp['success']:
            errors = ['Failed to set the replication status on the master.']
            return _get_return_dict(False, errors=errors)
    params = ['command=full-import']
    for (key, val) in options.items():
        params.append('&{}={}'.format(key, val))
    url = _format_url(handler, host=host, core_name=core_name, extra=params + extra)
    return _http_request(url)

def delta_import(handler, host=None, core_name=None, options=None, extra=None):
    if False:
        return 10
    '\n    Submits an import command to the specified handler using specified options.\n    This command can only be run if the minion is configured with\n    solr.type=master\n\n    handler : str\n        The name of the data import handler.\n    host : str (None)\n        The solr host to query. __opts__[\'host\'] is default.\n    core : str (None)\n        The core the handler belongs to.\n    options : dict (__opts__)\n        A list of options such as clean, optimize commit, verbose, and\n        pause_replication. leave blank to use __opts__ defaults. options will\n        be merged with __opts__\n\n    extra : dict ([])\n        Extra name value pairs to pass to the handler. e.g. ["name=value"]\n\n    Return : dict<str,obj>::\n\n        {\'success\':boolean, \'data\':dict, \'errors\':list, \'warnings\':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt \'*\' solr.delta_import dataimport None music {\'clean\':True}\n    '
    options = {} if options is None else options
    extra = [] if extra is None else extra
    if not _is_master() and _get_none_or_value(host) is None:
        err = ['solr.delta_import can only be called on "master" minions']
        return _get_return_dict(False, errors=err)
    resp = _pre_index_check(handler, host=host, core_name=core_name)
    if not resp['success']:
        return resp
    options = _merge_options(options)
    if options['clean'] and _check_for_cores():
        resp = set_replication_enabled(False, host=host, core_name=core_name)
        if not resp['success']:
            errors = ['Failed to set the replication status on the master.']
            return _get_return_dict(False, errors=errors)
    params = ['command=delta-import']
    for (key, val) in options.items():
        params.append('{}={}'.format(key, val))
    url = _format_url(handler, host=host, core_name=core_name, extra=params + extra)
    return _http_request(url)

def import_status(handler, host=None, core_name=None, verbose=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Submits an import command to the specified handler using specified options.\n    This command can only be run if the minion is configured with\n    solr.type: 'master'\n\n    handler : str\n        The name of the data import handler.\n    host : str (None)\n        The solr host to query. __opts__['host'] is default.\n    core : str (None)\n        The core the handler belongs to.\n    verbose : boolean (False)\n        Specifies verbose output\n\n    Return : dict<str,obj>::\n\n        {'success':boolean, 'data':dict, 'errors':list, 'warnings':list}\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' solr.import_status dataimport None music False\n    "
    if not _is_master() and _get_none_or_value(host) is None:
        errors = ['solr.import_status can only be called by "master" minions']
        return _get_return_dict(False, errors=errors)
    extra = ['command=status']
    if verbose:
        extra.append('verbose=true')
    url = _format_url(handler, host=host, core_name=core_name, extra=extra)
    return _http_request(url)