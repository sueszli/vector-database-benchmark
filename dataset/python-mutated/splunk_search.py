"""
Module for interop with the Splunk API

.. versionadded:: 2015.5.0

:depends:   - splunk-sdk python module
:configuration: Configure this module by specifying the name of a configuration
    profile in the minion config, minion pillar, or master config. The module
    will use the 'splunk' key by default, if defined.

    For example:

    .. code-block:: yaml

        splunk:
            username: alice
            password: abc123
            host: example.splunkcloud.com
            port: 8080
"""
import logging
import urllib.parse
import salt.utils.yaml
from salt.utils.odict import OrderedDict
HAS_LIBS = False
try:
    import requests
    import splunklib.client
    HAS_LIBS = True
except ImportError:
    pass
log = logging.getLogger(__name__)
__func_alias__ = {'list_': 'list'}
__virtualname__ = 'splunk_search'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load this module if splunk is installed on this minion.\n    '
    if HAS_LIBS:
        return __virtualname__
    return (False, 'The splunk_search execution module failed to load: requires both the requests and the splunk-sdk python library to be installed.')

def _get_splunk(profile):
    if False:
        i = 10
        return i + 15
    '\n    Return the splunk client, cached into __context__ for performance\n    '
    config = __salt__['config.option'](profile)
    key = 'splunk_search.{}:{}:{}:{}'.format(config.get('host'), config.get('port'), config.get('username'), config.get('password'))
    if key not in __context__:
        __context__[key] = splunklib.client.connect(host=config.get('host'), port=config.get('port'), username=config.get('username'), password=config.get('password'))
    return __context__[key]

def _get_splunk_search_props(search):
    if False:
        print('Hello World!')
    '\n    Get splunk search properties from an object\n    '
    props = search.content
    props['app'] = search.access.app
    props['sharing'] = search.access.sharing
    return props

def get(name, profile='splunk'):
    if False:
        print('Hello World!')
    "\n    Get a splunk search\n\n    CLI Example:\n\n        splunk_search.get 'my search name'\n    "
    client = _get_splunk(profile)
    search = None
    try:
        search = client.saved_searches[name]
    except KeyError:
        pass
    return search

def update(name, profile='splunk', **kwargs):
    if False:
        return 10
    "\n    Update a splunk search\n\n    CLI Example:\n\n        splunk_search.update 'my search name' sharing=app\n    "
    client = _get_splunk(profile)
    search = client.saved_searches[name]
    props = _get_splunk_search_props(search)
    updates = kwargs
    update_needed = False
    update_set = dict()
    diffs = []
    for key in sorted(kwargs):
        old_value = props.get(key, None)
        new_value = updates.get(key, None)
        if isinstance(old_value, str):
            old_value = old_value.strip()
        if isinstance(new_value, str):
            new_value = new_value.strip()
        if old_value != new_value:
            update_set[key] = new_value
            update_needed = True
            diffs.append("{}: '{}' => '{}'".format(key, old_value, new_value))
    if update_needed:
        search.update(**update_set).refresh()
        return (update_set, diffs)
    return False

def create(name, profile='splunk', **kwargs):
    if False:
        print('Hello World!')
    "\n    Create a splunk search\n\n    CLI Example:\n\n        splunk_search.create 'my search name' search='error msg'\n    "
    client = _get_splunk(profile)
    search = client.saved_searches.create(name, **kwargs)
    config = __salt__['config.option'](profile)
    url = 'https://{}:{}'.format(config.get('host'), config.get('port'))
    auth = (config.get('username'), config.get('password'))
    data = {'owner': config.get('username'), 'sharing': 'app', 'perms.read': '*'}
    _req_url = '{}/servicesNS/{}/search/saved/searches/{}/acl'.format(url, config.get('username'), urllib.parse.quote(name))
    requests.post(_req_url, auth=auth, verify=True, data=data)
    return _get_splunk_search_props(search)

def delete(name, profile='splunk'):
    if False:
        return 10
    "\n    Delete a splunk search\n\n    CLI Example:\n\n       splunk_search.delete 'my search name'\n    "
    client = _get_splunk(profile)
    try:
        client.saved_searches.delete(name)
        return True
    except KeyError:
        return None

def list_(profile='splunk'):
    if False:
        print('Hello World!')
    '\n    List splunk searches (names only)\n\n    CLI Example:\n\n        splunk_search.list\n    '
    client = _get_splunk(profile)
    searches = [x['name'] for x in client.saved_searches]
    return searches

def list_all(prefix=None, app=None, owner=None, description_contains=None, name_not_contains=None, profile='splunk'):
    if False:
        i = 10
        return i + 15
    '\n    Get all splunk search details. Produces results that can be used to create\n    an sls file.\n\n    if app or owner are specified, results will be limited to matching saved\n    searches.\n\n    if description_contains is specified, results will be limited to those\n    where "description_contains in description" is true if name_not_contains is\n    specified, results will be limited to those where "name_not_contains not in\n    name" is true.\n\n    If prefix parameter is given, alarm names in the output will be prepended\n    with the prefix; alarms that have the prefix will be skipped. This can be\n    used to convert existing alarms to be managed by salt, as follows:\n\n    CLI Example:\n\n            1. Make a "backup" of all existing searches\n                $ salt-call splunk_search.list_all --out=txt | sed "s/local: //" > legacy_searches.sls\n\n            2. Get all searches with new prefixed names\n                $ salt-call splunk_search.list_all "prefix=**MANAGED BY SALT** " --out=txt | sed "s/local: //" > managed_searches.sls\n\n            3. Insert the managed searches into splunk\n                $ salt-call state.sls managed_searches.sls\n\n            4.  Manually verify that the new searches look right\n\n            5.  Delete the original searches\n                $ sed s/present/absent/ legacy_searches.sls > remove_legacy_searches.sls\n                $ salt-call state.sls remove_legacy_searches.sls\n\n            6.  Get all searches again, verify no changes\n                $ salt-call splunk_search.list_all --out=txt | sed "s/local: //" > final_searches.sls\n                $ diff final_searches.sls managed_searches.sls\n    '
    client = _get_splunk(profile)
    name = 'splunk_search.list_all get defaults'
    try:
        client.saved_searches.delete(name)
    except Exception:
        pass
    search = client.saved_searches.create(name, search='nothing')
    defaults = dict(search.content)
    client.saved_searches.delete(name)
    readonly_keys = ('triggered_alert_count', 'action.email', 'action.populate_lookup', 'action.rss', 'action.script', 'action.summary_index', 'qualifiedSearch', 'next_scheduled_time')
    results = OrderedDict()
    searches = sorted(((s.name, s) for s in client.saved_searches))
    for (name, search) in searches:
        if app and search.access.app != app:
            continue
        if owner and search.access.owner != owner:
            continue
        if name_not_contains and name_not_contains in name:
            continue
        if prefix:
            if name.startswith(prefix):
                continue
            name = prefix + name
        d = [{'name': name}]
        description = ''
        for (k, v) in sorted(search.content.items()):
            if k in readonly_keys:
                continue
            if k.startswith('display.'):
                continue
            if not v:
                continue
            if k in defaults and defaults[k] == v:
                continue
            d.append({k: v})
            if k == 'description':
                description = v
        if description_contains and description_contains not in description:
            continue
        results['manage splunk search ' + name] = {'splunk_search.present': d}
    return salt.utils.yaml.safe_dump(results, default_flow_style=False, width=120)