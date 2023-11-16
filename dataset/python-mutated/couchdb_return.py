"""
Simple returner for CouchDB. Optional configuration
settings are listed below, along with sane defaults:

.. code-block:: yaml

    couchdb.db: 'salt'
    couchdb.url: 'http://salt:5984/'

Alternative configuration values can be used by prefacing the configuration.
Any values not found in the alternative configuration will be pulled from
the default location:

.. code-block:: yaml

    alternative.couchdb.db: 'salt'
    alternative.couchdb.url: 'http://salt:5984/'

To use the couchdb returner, append ``--return couchdb`` to the salt command. Example:

.. code-block:: bash

    salt '*' test.ping --return couchdb

To use the alternative configuration, append ``--return_config alternative`` to the salt command.

.. versionadded:: 2015.5.0

.. code-block:: bash

    salt '*' test.ping --return couchdb --return_config alternative

To override individual configuration items, append --return_kwargs '{"key:": "value"}' to the salt command.

.. versionadded:: 2016.3.0

.. code-block:: bash

    salt '*' test.ping --return couchdb --return_kwargs '{"db": "another-salt"}'

On concurrent database access
==============================

As this returner creates a couchdb document with the salt job id as document id
and as only one document with a given id can exist in a given couchdb database,
it is advised for most setups that every minion be configured to write to it own
database (the value of ``couchdb.db`` may be suffixed with the minion id),
otherwise multi-minion targeting can lead to losing output:

* the first returning minion is able to create a document in the database
* other minions fail with ``{'error': 'HTTP Error 409: Conflict'}``
"""
import logging
import time
from urllib.error import HTTPError
from urllib.request import HTTPHandler as _HTTPHandler
from urllib.request import Request as _Request
from urllib.request import build_opener as _build_opener
import salt.returners
import salt.utils.jid
import salt.utils.json
log = logging.getLogger(__name__)
__virtualname__ = 'couchdb'

def __virtual__():
    if False:
        i = 10
        return i + 15
    return __virtualname__

def _get_options(ret=None):
    if False:
        i = 10
        return i + 15
    '\n    Get the couchdb options from salt.\n    '
    attrs = {'url': 'url', 'db': 'db'}
    _options = salt.returners.get_returner_options(__virtualname__, ret, attrs, __salt__=__salt__, __opts__=__opts__)
    if 'url' not in _options:
        log.debug('Using default url.')
        _options['url'] = 'http://salt:5984/'
    if 'db' not in _options:
        log.debug('Using default database.')
        _options['db'] = 'salt'
    return _options

def _generate_doc(ret):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a object that will be saved into the database based on\n    options.\n    '
    retc = ret.copy()
    retc['_id'] = ret['jid']
    retc['timestamp'] = time.time()
    return retc

def _request(method, url, content_type=None, _data=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Makes a HTTP request. Returns the JSON parse, or an obj with an error.\n    '
    opener = _build_opener(_HTTPHandler)
    request = _Request(url, data=_data)
    if content_type:
        request.add_header('Content-Type', content_type)
    request.get_method = lambda : method
    try:
        handler = opener.open(request)
    except HTTPError as exc:
        return {'error': '{}'.format(exc)}
    return salt.utils.json.loads(handler.read())

def returner(ret):
    if False:
        i = 10
        return i + 15
    '\n    Take in the return and shove it into the couchdb database.\n    '
    options = _get_options(ret)
    _response = _request('GET', options['url'] + '_all_dbs')
    if options['db'] not in _response:
        _response = _request('PUT', options['url'] + options['db'])
        if 'ok' not in _response or _response['ok'] is not True:
            log.error("Unable to create database '%s'", options['db'])
            log.error('Nothing logged! Lost data.')
            return
        log.info("Created database '%s'", options['db'])
    doc = _generate_doc(ret)
    _response = _request('PUT', options['url'] + options['db'] + '/' + doc['_id'], 'application/json', salt.utils.json.dumps(doc))
    if 'ok' not in _response or _response['ok'] is not True:
        log.error("Unable to create document: '%s'", _response)
        log.error('Nothing logged! Lost data.')

def get_jid(jid):
    if False:
        return 10
    '\n    Get the document with a given JID.\n    '
    options = _get_options(ret=None)
    _response = _request('GET', options['url'] + options['db'] + '/' + jid)
    if 'error' in _response:
        log.error("Unable to get JID '%s' : '%s'", jid, _response)
        return {}
    return {_response['id']: _response}

def get_jids():
    if False:
        while True:
            i = 10
    '\n    List all the jobs that we have..\n    '
    options = _get_options(ret=None)
    _response = _request('GET', options['url'] + options['db'] + '/_all_docs?include_docs=true')
    if 'total_rows' not in _response:
        log.error("Didn't get valid response from requesting all docs: %s", _response)
        return {}
    ret = {}
    for row in _response['rows']:
        jid = row['id']
        if not salt.utils.jid.is_jid(jid):
            continue
        ret[jid] = salt.utils.jid.format_jid_instance(jid, row['doc'])
    return ret

def get_fun(fun):
    if False:
        return 10
    "\n    Return a dict with key being minion and value\n    being the job details of the last run of function 'fun'.\n    "
    options = _get_options(ret=None)
    _ret = {}
    for minion in get_minions():
        _response = _request('GET', options['url'] + options['db'] + '/_design/salt/_view/by-minion-fun-timestamp?descending=true&endkey=["{0}","{1}",0]&startkey=["{0}","{1}",9999999999]&limit=1'.format(minion, fun))
        if 'error' in _response:
            log.warning('Got an error when querying for last command by a minion: %s', _response['error'])
            continue
        if len(_response['rows']) < 1:
            continue
        _ret[minion] = _response['rows'][0]['value']
    return _ret

def get_minions():
    if False:
        print('Hello World!')
    '\n    Return a list of minion identifiers from a request of the view.\n    '
    options = _get_options(ret=None)
    if not ensure_views():
        return []
    _response = _request('GET', options['url'] + options['db'] + '/_design/salt/_view/minions?group=true')
    if 'rows' not in _response:
        log.error('Unable to get available minions: %s', _response)
        return []
    _ret = []
    for row in _response['rows']:
        _ret.append(row['key'])
    return _ret

def ensure_views():
    if False:
        while True:
            i = 10
    '\n    This function makes sure that all the views that should\n    exist in the design document do exist.\n    '
    options = _get_options(ret=None)
    _response = _request('GET', options['url'] + options['db'] + '/_design/salt')
    if 'error' in _response:
        return set_salt_view()
    for view in get_valid_salt_views():
        if view not in _response['views']:
            return set_salt_view()
    return True

def get_valid_salt_views():
    if False:
        while True:
            i = 10
    '\n    Returns a dict object of views that should be\n    part of the salt design document.\n    '
    ret = {}
    ret['minions'] = {}
    ret['minions']['map'] = 'function( doc ){ emit( doc.id, null ); }'
    ret['minions']['reduce'] = 'function( keys,values,rereduce ){ return key[0]; }'
    ret['by-minion-fun-timestamp'] = {}
    ret['by-minion-fun-timestamp']['map'] = 'function( doc ){ emit( [doc.id,doc.fun,doc.timestamp], doc ); }'
    return ret

def set_salt_view():
    if False:
        print('Hello World!')
    '\n    Helper function that sets the salt design\n    document. Uses get_valid_salt_views and some hardcoded values.\n    '
    options = _get_options(ret=None)
    new_doc = {}
    new_doc['views'] = get_valid_salt_views()
    new_doc['language'] = 'javascript'
    _response = _request('PUT', options['url'] + options['db'] + '/_design/salt', 'application/json', salt.utils.json.dumps(new_doc))
    if 'error' in _response:
        log.warning('Unable to set the salt design document: %s', _response['error'])
        return False
    return True

def prep_jid(nocache=False, passed_jid=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Do any work necessary to prepare a JID, including sending a custom id\n    '
    return passed_jid if passed_jid is not None else salt.utils.jid.gen_jid(__opts__)

def save_minions(jid, minions, syndic_id=None):
    if False:
        return 10
    '\n    Included for API consistency\n    '