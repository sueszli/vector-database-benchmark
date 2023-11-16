from __future__ import print_function
import six
from st2common.constants.pack import PACK_VERSION_SEPARATOR
from st2common.content.utils import get_pack_base_path
from st2common.runners.base_action import Action
from st2common.util.pack import get_pack_metadata

class GetPackDependencies(Action):

    def run(self, packs_status, nested):
        if False:
            print('Hello World!')
        '\n        :param packs_status: Name of the pack in Exchange or a git repo URL and download status.\n        :type: packs_status: ``dict``\n\n        :param nested: Nested level of dependencies to prevent infinite or really\n        long download loops.\n        :type nested: ``integer``\n        '
        result = {}
        dependency_list = []
        conflict_list = []
        if not packs_status or nested == 0:
            return result
        for (pack, status) in six.iteritems(packs_status):
            if 'success' not in status.lower():
                continue
            dependency_packs = get_dependency_list(pack)
            if not dependency_packs:
                continue
            for dep_pack in dependency_packs:
                (name_or_url, pack_version) = self.get_name_and_version(dep_pack)
                if len(name_or_url.split('/')) == 1:
                    pack_name = name_or_url
                else:
                    name_or_git = name_or_url.split('/')[-1]
                    pack_name = name_or_git if '.git' not in name_or_git else name_or_git.split('.')[0]
                existing_pack_version = get_pack_version(pack_name)
                if not existing_pack_version and 'stackstorm-' in pack_name.lower():
                    existing_pack_version = get_pack_version(pack_name.split('stackstorm-')[-1])
                if existing_pack_version:
                    if existing_pack_version and (not existing_pack_version.startswith('v')):
                        existing_pack_version = 'v' + existing_pack_version
                    if pack_version and (not pack_version.startswith('v')):
                        pack_version = 'v' + pack_version
                    if pack_version and existing_pack_version != pack_version and (dep_pack not in conflict_list):
                        conflict_list.append(dep_pack)
                else:
                    conflict = self.check_dependency_list_for_conflict(name_or_url, pack_version, dependency_list)
                    if conflict:
                        conflict_list.append(dep_pack)
                    elif dep_pack not in dependency_list:
                        dependency_list.append(dep_pack)
        result['dependency_list'] = dependency_list
        result['conflict_list'] = conflict_list
        result['nested'] = nested - 1
        return result

    def check_dependency_list_for_conflict(self, name, version, dependency_list):
        if False:
            for i in range(10):
                print('nop')
        conflict = False
        for pack in dependency_list:
            (name_or_url, pack_version) = self.get_name_and_version(pack)
            if name == name_or_url:
                if version != pack_version:
                    conflict = True
                break
        return conflict

    @staticmethod
    def get_name_and_version(pack):
        if False:
            print('Hello World!')
        pack_and_version = pack.split(PACK_VERSION_SEPARATOR)
        name_or_url = pack_and_version[0]
        pack_version = pack_and_version[1] if len(pack_and_version) > 1 else None
        return (name_or_url, pack_version)

def get_pack_version(pack=None):
    if False:
        i = 10
        return i + 15
    pack_path = get_pack_base_path(pack)
    try:
        pack_metadata = get_pack_metadata(pack_dir=pack_path)
        result = pack_metadata.get('version', None)
    except Exception:
        result = None
    finally:
        return result

def get_dependency_list(pack=None):
    if False:
        while True:
            i = 10
    pack_path = get_pack_base_path(pack)
    try:
        pack_metadata = get_pack_metadata(pack_dir=pack_path)
        result = pack_metadata.get('dependencies', None)
    except Exception:
        print('Could not open pack.yaml at location %s' % pack_path)
        result = None
    finally:
        return result