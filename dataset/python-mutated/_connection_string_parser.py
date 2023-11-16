from typing import Mapping

def parse_connection_string(conn_str: str, case_sensitive_keys: bool=False) -> Mapping[str, str]:
    if False:
        while True:
            i = 10
    'Parses the connection string into a dict of its component parts, with the option of preserving case\n    of keys, and validates that each key in the connection string has a provided value. If case of keys\n    is not preserved (ie. `case_sensitive_keys=False`), then a dict with LOWERCASE KEYS will be returned.\n\n    :param str conn_str: String with connection details provided by Azure services.\n    :param bool case_sensitive_keys: Indicates whether the casing of the keys will be preserved. When `False`(the\n        default), all keys will be lower-cased. If set to `True`, the original casing of the keys will be preserved.\n    :rtype: Mapping\n    :returns: Dict of connection string key/value pairs.\n    :raises:\n        ValueError: if each key in conn_str does not have a corresponding value and\n            for other bad formatting of connection strings - including duplicate\n            args, bad syntax, etc.\n    '
    cs_args = [s.split('=', 1) for s in conn_str.strip().rstrip(';').split(';')]
    if any((len(tup) != 2 or not all(tup) for tup in cs_args)):
        raise ValueError('Connection string is either blank or malformed.')
    args_dict = dict(cs_args)
    if len(cs_args) != len(args_dict):
        raise ValueError('Connection string is either blank or malformed.')
    if not case_sensitive_keys:
        new_args_dict = {}
        for key in args_dict.keys():
            new_key = key.lower()
            if new_key in new_args_dict:
                raise ValueError('Duplicate key in connection string: {}'.format(new_key))
            new_args_dict[new_key] = args_dict[key]
        return new_args_dict
    return args_dict