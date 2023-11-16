"""Contain common functions on dictionaries."""
__all__ = ['get_dict_entry_by_value', 'sort_dict']

def get_dict_entry_by_value(x: dict, value_select_fn=max):
    if False:
        while True:
            i = 10
    'Get from dictionary the entry with value that returned from value_select_fn.\n\n    Returns\n    -------\n    Tuple: key, value\n    '
    if not x:
        return (None, None)
    value = value_select_fn(x.values())
    index = list(x.values()).index(value)
    return (list(x.keys())[index], value)

def sort_dict(x: dict, reverse=True):
    if False:
        for i in range(10):
            print('nop')
    'Sort dictionary by values.\n\n    Returns\n    -------\n    Dict: sorted dictionary\n    '
    return dict(sorted(x.items(), key=lambda item: item[1], reverse=reverse))