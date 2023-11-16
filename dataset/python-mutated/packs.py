from __future__ import absolute_import
import copy
import itertools
import os
import requests
from requests.utils import should_bypass_proxies
import six
from six.moves import range
from oslo_config import cfg
import shutil
import yaml
from st2common import log as logging
from st2common.content.utils import get_pack_base_path
from st2common.exceptions.content import ResourceDiskFilesRemovalError
from st2common.models.db.stormbase import UIDFieldMixin
from st2common.persistence.pack import Pack
from st2common.util.misc import lowercase_value
from st2common.util.jsonify import json_encode
__all__ = ['get_pack_by_ref', 'fetch_pack_index', 'get_pack_from_index', 'search_pack_index', 'delete_action_files_from_pack', 'clone_action_files', 'clone_action_db', 'temp_backup_action_files', 'restore_temp_action_files', 'remove_temp_action_files']
EXCLUDE_FIELDS = ['repo_url', 'email']
SEARCH_PRIORITY = ['name', 'keywords']
LOG = logging.getLogger(__name__)

def _build_index_list(index_url):
    if False:
        i = 10
        return i + 15
    if not index_url:
        index_urls = cfg.CONF.content.index_url[::-1]
    elif isinstance(index_url, str):
        index_urls = [index_url]
    elif hasattr(index_url, '__iter__'):
        index_urls = index_url
    else:
        raise TypeError('"index_url" should either be a string or an iterable object.')
    return index_urls

def _fetch_and_compile_index(index_urls, logger=None, proxy_config=None):
    if False:
        i = 10
        return i + 15
    '\n    Go through the index list and compile results into a single object.\n    '
    status = []
    index = {}
    proxies_dict = {}
    verify = True
    if proxy_config:
        https_proxy = proxy_config.get('https_proxy', None)
        http_proxy = proxy_config.get('http_proxy', None)
        no_proxy = proxy_config.get('no_proxy', None)
        ca_bundle_path = proxy_config.get('proxy_ca_bundle_path', None)
        if https_proxy:
            proxies_dict['https'] = https_proxy
            verify = ca_bundle_path or True
        if http_proxy:
            proxies_dict['http'] = http_proxy
        if no_proxy:
            proxies_dict['no'] = no_proxy
    for index_url in index_urls:
        bypass_proxy = should_bypass_proxies(index_url, proxies_dict.get('no'))
        index_status = {'url': index_url, 'packs': 0, 'message': None, 'error': None}
        index_json = None
        try:
            request = requests.get(index_url, proxies=proxies_dict if not bypass_proxy else None, verify=verify if not bypass_proxy else True)
            request.raise_for_status()
            index_json = request.json()
        except ValueError as e:
            index_status['error'] = 'malformed'
            index_status['message'] = repr(e)
        except requests.exceptions.RequestException as e:
            index_status['error'] = 'unresponsive'
            index_status['message'] = repr(e)
        except Exception as e:
            index_status['error'] = 'other errors'
            index_status['message'] = repr(e)
        if index_json == {}:
            index_status['error'] = 'empty'
            index_status['message'] = 'The index URL returned an empty object.'
        elif type(index_json) is list:
            index_status['error'] = 'malformed'
            index_status['message'] = 'Expected an index object, got a list instead.'
        elif index_json and 'packs' not in index_json:
            index_status['error'] = 'malformed'
            index_status['message'] = 'Index object is missing "packs" attribute.'
        if index_status['error']:
            logger.error('Index parsing error: %s' % json_encode(index_status, indent=4))
        else:
            packs_data = index_json['packs']
            index_status['message'] = 'Success.'
            index_status['packs'] = len(packs_data)
            index.update(packs_data)
        status.append(index_status)
    return (index, status)

def get_pack_by_ref(pack_ref):
    if False:
        for i in range(10):
            print('nop')
    '\n    Retrieve PackDB by the provided reference.\n    '
    pack_db = Pack.get_by_ref(pack_ref)
    return pack_db

def fetch_pack_index(index_url=None, logger=None, allow_empty=False, proxy_config=None):
    if False:
        i = 10
        return i + 15
    '\n    Fetch the pack indexes (either from the config or provided as an argument)\n    and return the object.\n    '
    logger = logger or LOG
    index_urls = _build_index_list(index_url)
    (index, status) = _fetch_and_compile_index(index_urls=index_urls, logger=logger, proxy_config=proxy_config)
    if not index and (not allow_empty):
        raise ValueError('No results from the %s: tried %s.\nStatus: %s' % ('index' if len(index_urls) == 1 else 'indexes', ', '.join(index_urls), json_encode(status, indent=4)))
    return (index, status)

def get_pack_from_index(pack, proxy_config=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Search index by pack name.\n    Returns a pack.\n    '
    if not pack:
        raise ValueError('Pack name must be specified.')
    (index, _) = fetch_pack_index(proxy_config=proxy_config)
    return index.get(pack)

def search_pack_index(query, exclude=None, priority=None, case_sensitive=True, proxy_config=None):
    if False:
        while True:
            i = 10
    '\n    Search the pack index by query.\n    Returns a list of matches for a query.\n    '
    if not query:
        raise ValueError('Query must be specified.')
    if not exclude:
        exclude = EXCLUDE_FIELDS
    if not priority:
        priority = SEARCH_PRIORITY
    if not case_sensitive:
        query = str(query).lower()
    (index, _) = fetch_pack_index(proxy_config=proxy_config)
    matches = [[] for i in range(len(priority) + 1)]
    for pack in six.itervalues(index):
        for (key, value) in six.iteritems(pack):
            if not hasattr(value, '__contains__'):
                value = str(value)
            if not case_sensitive:
                value = lowercase_value(value=value)
            if key not in exclude and query in value:
                if key in priority:
                    matches[priority.index(key)].append(pack)
                else:
                    matches[-1].append(pack)
                break
    return list(itertools.chain.from_iterable(matches))

def delete_action_files_from_pack(pack_name, entry_point, metadata_file):
    if False:
        return 10
    '\n    Prepares the path for entry_point file and metadata file of action and\n    deletes them from disk.\n    '
    pack_base_path = get_pack_base_path(pack_name=pack_name)
    action_entrypoint_file_path = os.path.join(pack_base_path, 'actions', entry_point)
    action_metadata_file_path = os.path.join(pack_base_path, metadata_file)
    if os.path.isfile(action_entrypoint_file_path):
        try:
            os.remove(action_entrypoint_file_path)
        except PermissionError:
            LOG.error('No permission to delete the "%s" file', action_entrypoint_file_path)
            msg = 'No permission to delete "%s" file from disk' % action_entrypoint_file_path
            raise PermissionError(msg)
        except Exception as e:
            LOG.error('Unable to delete "%s" file. Exception was "%s"', action_entrypoint_file_path, e)
            msg = 'The action file "%s" could not be removed from disk, please check the logs or ask your StackStorm administrator to check and delete the actions files manually' % action_entrypoint_file_path
            raise ResourceDiskFilesRemovalError(msg)
    else:
        LOG.warning('The action entry point file "%s" does not exists on disk.', action_entrypoint_file_path)
    if os.path.isfile(action_metadata_file_path):
        try:
            os.remove(action_metadata_file_path)
        except PermissionError:
            LOG.error('No permission to delete the "%s" file', action_metadata_file_path)
            msg = 'No permission to delete "%s" file from disk' % action_metadata_file_path
            raise PermissionError(msg)
        except Exception as e:
            LOG.error('Could not delete "%s" file. Exception was "%s"', action_metadata_file_path, e)
            msg = 'The action file "%s" could not be removed from disk, please check the logs or ask your StackStorm administrator to check and delete the actions files manually' % action_metadata_file_path
            raise ResourceDiskFilesRemovalError(msg)
    else:
        LOG.warning('The action metadata file "%s" does not exists on disk.', action_metadata_file_path)

def _clone_content_to_destination_file(source_file, destination_file):
    if False:
        i = 10
        return i + 15
    try:
        shutil.copy(src=source_file, dst=destination_file)
    except PermissionError:
        LOG.error('Unable to copy file to "%s" due to permission error.', destination_file)
        msg = 'Unable to copy file to "%s".' % destination_file
        raise PermissionError(msg)
    except Exception as e:
        LOG.error('Unable to copy file to "%s". Exception was "%s".', destination_file, e)
        msg = 'Unable to copy file to "%s". Please check the logs or ask your administrator to clone the files manually.' % destination_file
        raise Exception(msg)

def clone_action_files(source_action_db, dest_action_db, dest_pack_base_path):
    if False:
        i = 10
        return i + 15
    '\n    Prepares the path for entry point and metadata files for source and destination.\n    Clones the content from source action files to destination action files.\n    '
    source_pack = source_action_db['pack']
    source_entry_point = source_action_db['entry_point']
    source_metadata_file = source_action_db['metadata_file']
    source_pack_base_path = get_pack_base_path(pack_name=source_pack)
    source_metadata_file_path = os.path.join(source_pack_base_path, source_metadata_file)
    dest_metadata_file_name = dest_action_db['metadata_file']
    dest_metadata_file_path = os.path.join(dest_pack_base_path, dest_metadata_file_name)
    ac_dir_path = os.path.join(dest_pack_base_path, 'actions')
    if not os.path.isdir(ac_dir_path):
        os.mkdir(path=ac_dir_path)
    _clone_content_to_destination_file(source_file=source_metadata_file_path, destination_file=dest_metadata_file_path)
    dest_entry_point = dest_action_db['entry_point']
    dest_runner_type = dest_action_db['runner_type']['name']
    if dest_entry_point:
        if dest_runner_type in ['orquesta', 'action-chain']:
            wf_dir_path = os.path.join(dest_pack_base_path, 'actions', 'workflows')
            if not os.path.isdir(wf_dir_path):
                os.mkdir(path=wf_dir_path)
        source_entry_point_file_path = os.path.join(source_pack_base_path, 'actions', source_entry_point)
        dest_entrypoint_file_path = os.path.join(dest_pack_base_path, 'actions', dest_entry_point)
        _clone_content_to_destination_file(source_file=source_entry_point_file_path, destination_file=dest_entrypoint_file_path)
    with open(dest_metadata_file_path) as df:
        doc = yaml.load(df, Loader=yaml.FullLoader)
    doc['name'] = dest_action_db['name']
    if 'pack' in doc:
        doc['pack'] = dest_action_db['pack']
    doc['entry_point'] = dest_entry_point
    with open(dest_metadata_file_path, 'w') as df:
        yaml.dump(doc, df, default_flow_style=False, sort_keys=False)

def clone_action_db(source_action_db, dest_pack, dest_action):
    if False:
        print('Hello World!')
    dest_action_db = copy.deepcopy(source_action_db)
    source_runner_type = source_action_db['runner_type']['name']
    if source_action_db['entry_point']:
        if source_runner_type in ['orquesta', 'action-chain']:
            dest_entry_point_file_name = 'workflows/%s.yaml' % dest_action
        else:
            old_ext = os.path.splitext(source_action_db['entry_point'])[1]
            dest_entry_point_file_name = dest_action + old_ext
    else:
        dest_entry_point_file_name = ''
    dest_action_db['entry_point'] = dest_entry_point_file_name
    dest_action_db['metadata_file'] = 'actions/%s.yaml' % dest_action
    dest_action_db['name'] = dest_action
    dest_ref = '.'.join([dest_pack, dest_action])
    dest_action_db['ref'] = dest_ref
    dest_action_db['uid'] = UIDFieldMixin.UID_SEPARATOR.join(['action', dest_pack, dest_action])
    if 'pack' in dest_action_db:
        dest_action_db['pack'] = dest_pack
    dest_action_db['id'] = None
    return dest_action_db

def temp_backup_action_files(pack_base_path, metadata_file, entry_point, temp_sub_dir):
    if False:
        while True:
            i = 10
    temp_dir_path = '/tmp/%s' % temp_sub_dir
    os.mkdir(temp_dir_path)
    actions_dir = os.path.join(temp_dir_path, 'actions')
    os.mkdir(actions_dir)
    temp_metadata_file_path = os.path.join(temp_dir_path, metadata_file)
    dest_metadata_file_path = os.path.join(pack_base_path, metadata_file)
    _clone_content_to_destination_file(source_file=dest_metadata_file_path, destination_file=temp_metadata_file_path)
    if entry_point:
        entry_point_dir = str(os.path.split(entry_point)[0])
        if entry_point_dir != '':
            os.makedirs(os.path.join(actions_dir, entry_point_dir))
        temp_entry_point_file_path = os.path.join(actions_dir, entry_point)
        dest_entry_point_file_path = os.path.join(pack_base_path, 'actions', entry_point)
        _clone_content_to_destination_file(source_file=dest_entry_point_file_path, destination_file=temp_entry_point_file_path)

def restore_temp_action_files(pack_base_path, metadata_file, entry_point, temp_sub_dir):
    if False:
        while True:
            i = 10
    temp_dir_path = '/tmp/%s' % temp_sub_dir
    temp_metadata_file_path = os.path.join(temp_dir_path, metadata_file)
    dest_metadata_file_path = os.path.join(pack_base_path, metadata_file)
    _clone_content_to_destination_file(source_file=temp_metadata_file_path, destination_file=dest_metadata_file_path)
    if entry_point:
        temp_entry_point_file_path = os.path.join(temp_dir_path, 'actions', entry_point)
        dest_entry_point_file_path = os.path.join(pack_base_path, 'actions', entry_point)
        _clone_content_to_destination_file(source_file=temp_entry_point_file_path, destination_file=dest_entry_point_file_path)

def remove_temp_action_files(temp_sub_dir):
    if False:
        return 10
    temp_dir_path = '/tmp/%s' % temp_sub_dir
    if os.path.isdir(temp_dir_path):
        try:
            shutil.rmtree(temp_dir_path)
        except PermissionError:
            LOG.error('No permission to delete the "%s" directory', temp_dir_path)
            msg = 'No permission to delete the "%s" directory' % temp_dir_path
            raise PermissionError(msg)
        except Exception as e:
            LOG.error('Unable to delete "%s" directory. Exception was "%s"', temp_dir_path, e)
            msg = 'The temporary directory "%s" could not be removed from disk, please check the logs or ask your StackStorm administrator to check and delete the temporary directory manually' % temp_dir_path
            raise Exception(msg)