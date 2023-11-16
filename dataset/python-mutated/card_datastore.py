""" 

"""
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
import os
import shutil
from metaflow.plugins.datastores.local_storage import LocalStorage
from metaflow.metaflow_config import CARD_S3ROOT, CARD_LOCALROOT, DATASTORE_LOCAL_DIR, CARD_SUFFIX, CARD_AZUREROOT, CARD_GSROOT, SKIP_CARD_DUALWRITE
import metaflow.metaflow_config as metaflow_config
from .exception import CardNotPresentException
TEMP_DIR_NAME = 'metaflow_card_cache'
NUM_SHORT_HASH_CHARS = 5
CardInfo = namedtuple('CardInfo', ['type', 'hash', 'id', 'filename'])

def path_spec_resolver(pathspec):
    if False:
        return 10
    splits = pathspec.split('/')
    splits.extend([None] * (4 - len(splits)))
    return tuple(splits)

def is_file_present(path):
    if False:
        print('Hello World!')
    try:
        os.stat(path)
        return True
    except FileNotFoundError:
        return False
    except:
        raise

class CardDatastore(object):

    @classmethod
    def get_storage_root(cls, storage_type):
        if False:
            for i in range(10):
                print('nop')
        if storage_type == 's3':
            return CARD_S3ROOT
        elif storage_type == 'azure':
            return CARD_AZUREROOT
        elif storage_type == 'gs':
            return CARD_GSROOT
        elif storage_type == 'local':
            result = CARD_LOCALROOT
            if result is None:
                current_path = os.getcwd()
                check_dir = os.path.join(current_path, DATASTORE_LOCAL_DIR, CARD_SUFFIX)
                check_dir = os.path.realpath(check_dir)
                orig_path = check_dir
                while not os.path.isdir(check_dir):
                    new_path = os.path.dirname(current_path)
                    if new_path == current_path:
                        break
                    current_path = new_path
                    check_dir = os.path.join(current_path, DATASTORE_LOCAL_DIR, CARD_SUFFIX)
                result = orig_path
            return result
        else:
            raise NotImplementedError('Card datastore does not support backend %s' % (storage_type,))

    def __init__(self, flow_datastore, pathspec=None):
        if False:
            print('Hello World!')
        self._backend = flow_datastore._storage_impl
        self._flow_name = flow_datastore.flow_name
        (_, run_id, step_name, _) = pathspec.split('/')
        self._run_id = run_id
        self._step_name = step_name
        self._pathspec = pathspec
        self._temp_card_save_path = self._get_write_path(base_pth=TEMP_DIR_NAME)

    @classmethod
    def get_card_location(cls, base_path, card_name, card_html, card_id=None):
        if False:
            print('Hello World!')
        chash = sha1(bytes(card_html, 'utf-8')).hexdigest()
        if card_id is None:
            card_file_name = '%s-%s.html' % (card_name, chash)
        else:
            card_file_name = '%s-%s-%s.html' % (card_name, card_id, chash)
        return os.path.join(base_path, card_file_name)

    def _make_path(self, base_pth, pathspec=None, with_steps=False):
        if False:
            print('Hello World!')
        sysroot = base_pth
        if pathspec is not None:
            (flow_name, run_id, step_name, task_id) = path_spec_resolver(pathspec)
        if with_steps:
            pth_arr = [sysroot, flow_name, 'runs', run_id, 'steps', step_name, 'tasks', task_id, 'cards']
        else:
            pth_arr = [sysroot, flow_name, 'runs', run_id, 'tasks', task_id, 'cards']
        if sysroot == '' or sysroot is None:
            pth_arr.pop(0)
        return os.path.join(*pth_arr)

    def _get_write_path(self, base_pth=''):
        if False:
            while True:
                i = 10
        return self._make_path(base_pth, pathspec=self._pathspec, with_steps=True)

    def _get_read_path(self, base_pth='', with_steps=False):
        if False:
            for i in range(10):
                print('nop')
        return self._make_path(base_pth, pathspec=self._pathspec, with_steps=with_steps)

    @staticmethod
    def card_info_from_path(path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            path (str): The path to the card\n\n        Raises:\n            Exception: When the card_path is invalid\n\n        Returns:\n            CardInfo\n        '
        card_file_name = path.split('/')[-1]
        file_split = card_file_name.split('-')
        if len(file_split) not in [2, 3]:
            raise Exception('Invalid card file name %s. Card file names should be of form TYPE-HASH.html or TYPE-ID-HASH.html' % card_file_name)
        (card_type, card_hash, card_id) = (None, None, None)
        if len(file_split) == 2:
            (card_type, card_hash) = file_split
        else:
            (card_type, card_id, card_hash) = file_split
        card_hash = card_hash.split('.html')[0]
        return CardInfo(card_type, card_hash, card_id, card_file_name)

    def save_card(self, card_type, card_html, card_id=None, overwrite=True):
        if False:
            print('Hello World!')
        card_file_name = card_type
        card_path_with_steps = self.get_card_location(self._get_write_path(), card_file_name, card_html, card_id=card_id)
        if SKIP_CARD_DUALWRITE:
            self._backend.save_bytes([(card_path_with_steps, BytesIO(bytes(card_html, 'utf-8')))], overwrite=overwrite)
        else:
            card_path_without_steps = self.get_card_location(self._get_read_path(with_steps=False), card_file_name, card_html, card_id=card_id)
            for cp in [card_path_with_steps, card_path_without_steps]:
                self._backend.save_bytes([(cp, BytesIO(bytes(card_html, 'utf-8')))], overwrite=overwrite)
        return self.card_info_from_path(card_path_with_steps)

    def _list_card_paths(self, card_type=None, card_hash=None, card_id=None):
        if False:
            return 10
        card_paths = []
        card_paths_with_steps = self._backend.list_content([self._get_read_path(with_steps=True)])
        if len(card_paths_with_steps) == 0:
            card_paths_without_steps = self._backend.list_content([self._get_read_path(with_steps=False)])
            if len(card_paths_without_steps) == 0:
                raise CardNotPresentException(self._pathspec, card_hash=card_hash, card_type=card_type)
            else:
                card_paths = card_paths_without_steps
        else:
            card_paths = card_paths_with_steps
        cards_found = []
        for task_card_path in card_paths:
            card_path = task_card_path.path
            card_info = self.card_info_from_path(card_path)
            if card_type is not None and card_info.type != card_type:
                continue
            elif card_hash is not None:
                if not card_info.hash.startswith(card_hash):
                    continue
            elif card_id is not None and card_info.id != card_id:
                continue
            if task_card_path.is_file:
                cards_found.append(card_path)
        return cards_found

    def create_full_path(self, card_path):
        if False:
            return 10
        return os.path.join(self._backend.datastore_root, card_path)

    def get_card_names(self, card_paths):
        if False:
            for i in range(10):
                print('nop')
        return [self.card_info_from_path(path) for path in card_paths]

    def get_card_html(self, path):
        if False:
            return 10
        with self._backend.load_bytes([path]) as get_results:
            for (_, path, _) in get_results:
                if path is not None:
                    with open(path, 'r') as f:
                        return f.read()

    def cache_locally(self, path, save_path=None):
        if False:
            print('Hello World!')
        '\n        Saves the data present in the `path` the `metaflow_card_cache` directory or to the `save_path`.\n        '
        if save_path is None:
            if not is_file_present(self._temp_card_save_path):
                LocalStorage._makedirs(self._temp_card_save_path)
        else:
            save_dir = os.path.dirname(save_path)
            if save_dir != '' and (not is_file_present(save_dir)):
                LocalStorage._makedirs(os.path.dirname(save_path))
        with self._backend.load_bytes([path]) as get_results:
            for (key, path, meta) in get_results:
                if path is not None:
                    main_path = path
                    if save_path is None:
                        file_name = key.split('/')[-1]
                        main_path = os.path.join(self._temp_card_save_path, file_name)
                    else:
                        main_path = save_path
                    shutil.copy(path, main_path)
                    return main_path

    def extract_card_paths(self, card_type=None, card_hash=None, card_id=None):
        if False:
            print('Hello World!')
        return self._list_card_paths(card_type=card_type, card_hash=card_hash, card_id=card_id)