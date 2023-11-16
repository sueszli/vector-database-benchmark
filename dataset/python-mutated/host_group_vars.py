from __future__ import annotations
DOCUMENTATION = '\n    name: host_group_vars\n    version_added: "2.4"\n    short_description: In charge of loading group_vars and host_vars\n    requirements:\n        - Enabled in configuration\n    description:\n        - Loads YAML vars into corresponding groups/hosts in group_vars/ and host_vars/ directories.\n        - Files are restricted by extension to one of .yaml, .json, .yml or no extension.\n        - Hidden (starting with \'.\') and backup (ending with \'~\') files and directories are ignored.\n        - Only applies to inventory sources that are existing paths.\n        - Starting in 2.10, this plugin requires enabling and is enabled by default.\n    options:\n      stage:\n        ini:\n          - key: stage\n            section: vars_host_group_vars\n        env:\n          - name: ANSIBLE_VARS_PLUGIN_STAGE\n      _valid_extensions:\n        default: [".yml", ".yaml", ".json"]\n        description:\n          - "Check all of these extensions when looking for \'variable\' files which should be YAML or JSON or vaulted versions of these."\n          - \'This affects vars_files, include_vars, inventory and vars plugins among others.\'\n        env:\n          - name: ANSIBLE_YAML_FILENAME_EXT\n        ini:\n          - key: yaml_valid_extensions\n            section: defaults\n        type: list\n        elements: string\n    extends_documentation_fragment:\n      - vars_plugin_staging\n'
import os
from ansible.errors import AnsibleParserError
from ansible.module_utils.common.text.converters import to_native
from ansible.plugins.vars import BaseVarsPlugin
from ansible.utils.path import basedir
from ansible.inventory.group import InventoryObjectType
from ansible.utils.vars import combine_vars
CANONICAL_PATHS = {}
FOUND = {}
NAK = set()
PATH_CACHE = {}

class VarsModule(BaseVarsPlugin):
    REQUIRES_ENABLED = True
    is_stateless = True

    def load_found_files(self, loader, data, found_files):
        if False:
            i = 10
            return i + 15
        for found in found_files:
            new_data = loader.load_from_file(found, cache=True, unsafe=True)
            if new_data:
                data = combine_vars(data, new_data)
        return data

    def get_vars(self, loader, path, entities, cache=True):
        if False:
            while True:
                i = 10
        ' parses the inventory file '
        if not isinstance(entities, list):
            entities = [entities]
        try:
            realpath_basedir = CANONICAL_PATHS[path]
        except KeyError:
            CANONICAL_PATHS[path] = realpath_basedir = os.path.realpath(basedir(path))
        data = {}
        for entity in entities:
            try:
                entity_name = entity.name
            except AttributeError:
                raise AnsibleParserError('Supplied entity must be Host or Group, got %s instead' % type(entity))
            try:
                first_char = entity_name[0]
            except (TypeError, IndexError, KeyError):
                raise AnsibleParserError('Supplied entity must be Host or Group, got %s instead' % type(entity))
            if first_char != os.path.sep:
                try:
                    found_files = []
                    try:
                        entity_type = entity.base_type
                    except AttributeError:
                        raise AnsibleParserError('Supplied entity must be Host or Group, got %s instead' % type(entity))
                    if entity_type is InventoryObjectType.HOST:
                        subdir = 'host_vars'
                    elif entity_type is InventoryObjectType.GROUP:
                        subdir = 'group_vars'
                    else:
                        raise AnsibleParserError('Supplied entity must be Host or Group, got %s instead' % type(entity))
                    if cache:
                        try:
                            opath = PATH_CACHE[realpath_basedir, subdir]
                        except KeyError:
                            opath = PATH_CACHE[realpath_basedir, subdir] = os.path.join(realpath_basedir, subdir)
                        if opath in NAK:
                            continue
                        key = '%s.%s' % (entity_name, opath)
                        if key in FOUND:
                            data = self.load_found_files(loader, data, FOUND[key])
                            continue
                    else:
                        opath = PATH_CACHE[realpath_basedir, subdir] = os.path.join(realpath_basedir, subdir)
                    if os.path.isdir(opath):
                        self._display.debug('\tprocessing dir %s' % opath)
                        FOUND[key] = found_files = loader.find_vars_files(opath, entity_name)
                    elif not os.path.exists(opath):
                        NAK.add(opath)
                    else:
                        self._display.warning('Found %s that is not a directory, skipping: %s' % (subdir, opath))
                        NAK.add(opath)
                    data = self.load_found_files(loader, data, found_files)
                except Exception as e:
                    raise AnsibleParserError(to_native(e))
        return data