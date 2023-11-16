from __future__ import annotations
from ansible.errors import AnsibleError
from ansible.module_utils.six import string_types
from ansible.playbook.attribute import FieldAttribute
from ansible.template import Templar
from ansible.utils.sentinel import Sentinel

def _flatten_tags(tags: list) -> list:
    if False:
        return 10
    rv = set()
    for tag in tags:
        if isinstance(tag, list):
            rv.update(tag)
        else:
            rv.add(tag)
    return list(rv)

class Taggable:
    untagged = frozenset(['untagged'])
    tags = FieldAttribute(isa='list', default=list, listof=(string_types, int), extend=True)

    def _load_tags(self, attr, ds):
        if False:
            i = 10
            return i + 15
        if isinstance(ds, list):
            return ds
        elif isinstance(ds, string_types):
            return [x.strip() for x in ds.split(',')]
        else:
            raise AnsibleError('tags must be specified as a list', obj=ds)

    def evaluate_tags(self, only_tags, skip_tags, all_vars):
        if False:
            for i in range(10):
                print('nop')
        ' this checks if the current item should be executed depending on tag options '
        if self.tags:
            templar = Templar(loader=self._loader, variables=all_vars)
            obj = self
            while obj is not None:
                if (_tags := getattr(obj, '_tags', Sentinel)) is not Sentinel:
                    obj._tags = _flatten_tags(templar.template(_tags))
                obj = obj._parent
            tags = set(self.tags)
        else:
            tags = self.untagged
        should_run = True
        if only_tags:
            if 'always' in tags:
                should_run = True
            elif 'all' in only_tags and 'never' not in tags:
                should_run = True
            elif not tags.isdisjoint(only_tags):
                should_run = True
            elif 'tagged' in only_tags and tags != self.untagged and ('never' not in tags):
                should_run = True
            else:
                should_run = False
        if should_run and skip_tags:
            if 'all' in skip_tags:
                if 'always' not in tags or 'always' in skip_tags:
                    should_run = False
            elif not tags.isdisjoint(skip_tags):
                should_run = False
            elif 'tagged' in skip_tags and tags != self.untagged:
                should_run = False
        return should_run