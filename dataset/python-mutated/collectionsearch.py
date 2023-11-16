from __future__ import annotations
from ansible.module_utils.six import string_types
from ansible.playbook.attribute import FieldAttribute
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.template import is_template
from ansible.utils.display import Display
from jinja2.nativetypes import NativeEnvironment
display = Display()

def _ensure_default_collection(collection_list=None):
    if False:
        i = 10
        return i + 15
    default_collection = AnsibleCollectionConfig.default_collection
    if collection_list is None:
        collection_list = []
    if default_collection and default_collection not in collection_list:
        collection_list.insert(0, default_collection)
    if collection_list and 'ansible.builtin' not in collection_list and ('ansible.legacy' not in collection_list):
        collection_list.append('ansible.legacy')
    return collection_list

class CollectionSearch:
    collections = FieldAttribute(isa='list', listof=string_types, priority=100, default=_ensure_default_collection, always_post_validate=True, static=True)

    def _load_collections(self, attr, ds):
        if False:
            for i in range(10):
                print('nop')
        ds = self.get_validated_value('collections', self.fattributes.get('collections'), ds, None)
        _ensure_default_collection(collection_list=ds)
        if not ds:
            return None
        env = NativeEnvironment()
        for collection_name in ds:
            if is_template(collection_name, env):
                display.warning('"collections" is not templatable, but we found: %s, it will not be templated and will be used "as is".' % collection_name)
        return ds