from __future__ import annotations
import os
import ansible.constants as C
from ansible.errors import AnsibleParserError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import string_types
from ansible.parsing.splitter import split_args
from ansible.parsing.yaml.objects import AnsibleBaseYAMLObject, AnsibleMapping
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.conditional import Conditional
from ansible.playbook.taggable import Taggable
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path, _get_collection_playbook_path
from ansible.template import Templar
from ansible.utils.display import Display
display = Display()

class PlaybookInclude(Base, Conditional, Taggable):
    import_playbook = NonInheritableFieldAttribute(isa='string')
    vars_val = NonInheritableFieldAttribute(isa='dict', default=dict, alias='vars')

    @staticmethod
    def load(data, basedir, variable_manager=None, loader=None):
        if False:
            print('Hello World!')
        return PlaybookInclude().load_data(ds=data, basedir=basedir, variable_manager=variable_manager, loader=loader)

    def load_data(self, ds, variable_manager=None, loader=None, basedir=None):
        if False:
            print('Hello World!')
        "\n        Overrides the base load_data(), as we're actually going to return a new\n        Playbook() object rather than a PlaybookInclude object\n        "
        from ansible.playbook import Playbook
        from ansible.playbook.play import Play
        new_obj = super(PlaybookInclude, self).load_data(ds, variable_manager, loader)
        all_vars = self.vars.copy()
        if variable_manager:
            all_vars |= variable_manager.get_vars()
        templar = Templar(loader=loader, variables=all_vars)
        pb = Playbook(loader=loader)
        file_name = templar.template(new_obj.import_playbook)
        resource = _get_collection_playbook_path(file_name)
        if resource is not None:
            playbook = resource[1]
            playbook_collection = resource[2]
        else:
            playbook = file_name
            if not os.path.isabs(playbook):
                playbook = os.path.join(basedir, playbook)
            playbook_collection = _get_collection_name_from_path(playbook)
        if playbook_collection:
            AnsibleCollectionConfig.default_collection = playbook_collection
        else:
            AnsibleCollectionConfig.playbook_paths.append(os.path.dirname(os.path.abspath(to_bytes(playbook, errors='surrogate_or_strict'))))
        pb._load_playbook_data(file_name=playbook, variable_manager=variable_manager, vars=self.vars.copy())
        for entry in pb._entries:
            if new_obj.when and isinstance(entry, Play):
                entry._included_conditional = new_obj.when[:]
            temp_vars = entry.vars | new_obj.vars
            param_tags = temp_vars.pop('tags', None)
            if param_tags is not None:
                entry.tags.extend(param_tags.split(','))
            entry.vars = temp_vars
            entry.tags = list(set(entry.tags).union(new_obj.tags))
            if entry._included_path is None:
                entry._included_path = os.path.dirname(playbook)
            if new_obj.when:
                for task_block in entry.pre_tasks + entry.roles + entry.tasks + entry.post_tasks:
                    task_block._when = new_obj.when[:] + task_block.when[:]
        return pb

    def preprocess_data(self, ds):
        if False:
            while True:
                i = 10
        '\n        Regorganizes the data for a PlaybookInclude datastructure to line\n        up with what we expect the proper attributes to be\n        '
        if not isinstance(ds, dict):
            raise AnsibleAssertionError('ds (%s) should be a dict but was a %s' % (ds, type(ds)))
        new_ds = AnsibleMapping()
        if isinstance(ds, AnsibleBaseYAMLObject):
            new_ds.ansible_pos = ds.ansible_pos
        for (k, v) in ds.items():
            if k in C._ACTION_IMPORT_PLAYBOOK:
                self._preprocess_import(ds, new_ds, k, v)
            else:
                if k == 'vars':
                    if 'vars' in new_ds:
                        raise AnsibleParserError("import_playbook parameters cannot be mixed with 'vars' entries for import statements", obj=ds)
                    elif not isinstance(v, dict):
                        raise AnsibleParserError('vars for import_playbook statements must be specified as a dictionary', obj=ds)
                new_ds[k] = v
        return super(PlaybookInclude, self).preprocess_data(new_ds)

    def _preprocess_import(self, ds, new_ds, k, v):
        if False:
            i = 10
            return i + 15
        '\n        Splits the playbook import line up into filename and parameters\n        '
        if v is None:
            raise AnsibleParserError('playbook import parameter is missing', obj=ds)
        elif not isinstance(v, string_types):
            raise AnsibleParserError('playbook import parameter must be a string indicating a file path, got %s instead' % type(v), obj=ds)
        items = split_args(v)
        if len(items) == 0:
            raise AnsibleParserError('import_playbook statements must specify the file name to import', obj=ds)
        new_ds['import_playbook'] = items[0].strip()