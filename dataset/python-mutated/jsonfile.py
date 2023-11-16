from __future__ import annotations
DOCUMENTATION = '\n    name: jsonfile\n    short_description: JSON formatted files.\n    description:\n        - This cache uses JSON formatted, per host, files saved to the filesystem.\n    version_added: "1.9"\n    author: Ansible Core (@ansible-core)\n    options:\n      _uri:\n        required: True\n        description:\n          - Path in which the cache plugin will save the JSON files\n        env:\n          - name: ANSIBLE_CACHE_PLUGIN_CONNECTION\n        ini:\n          - key: fact_caching_connection\n            section: defaults\n        type: path\n      _prefix:\n        description: User defined prefix to use when creating the JSON files\n        env:\n          - name: ANSIBLE_CACHE_PLUGIN_PREFIX\n        ini:\n          - key: fact_caching_prefix\n            section: defaults\n      _timeout:\n        default: 86400\n        description: Expiration timeout for the cache plugin data\n        env:\n          - name: ANSIBLE_CACHE_PLUGIN_TIMEOUT\n        ini:\n          - key: fact_caching_timeout\n            section: defaults\n        type: integer\n'
import codecs
import json
from ansible.parsing.ajson import AnsibleJSONEncoder, AnsibleJSONDecoder
from ansible.plugins.cache import BaseFileCacheModule

class CacheModule(BaseFileCacheModule):
    """
    A caching module backed by json files.
    """

    def _load(self, filepath):
        if False:
            i = 10
            return i + 15
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f, cls=AnsibleJSONDecoder)

    def _dump(self, value, filepath):
        if False:
            while True:
                i = 10
        with codecs.open(filepath, 'w', encoding='utf-8') as f:
            f.write(json.dumps(value, cls=AnsibleJSONEncoder, sort_keys=True, indent=4))