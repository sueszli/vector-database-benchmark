from __future__ import annotations
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
from ansible.module_utils.common.yaml import yaml_load

def _meta_yml_to_dict(yaml_string_data, content_id):
    if False:
        return 10
    '\n    Converts string YAML dictionary to a Python dictionary. This function may be monkeypatched to another implementation\n    by some tools (eg the import sanity test).\n    :param yaml_string_data: a bytes-ish YAML dictionary\n    :param content_id: a unique ID representing the content to allow other implementations to cache the output\n    :return: a Python dictionary representing the YAML dictionary content\n    '
    routing_dict = yaml_load(yaml_string_data)
    if not routing_dict:
        routing_dict = {}
    if not isinstance(routing_dict, Mapping):
        raise ValueError('collection metadata must be an instance of Python Mapping')
    return routing_dict