import sys
import salt.payload

def _trim_dict_in_dict(data, max_val_size, replace_with):
    if False:
        for i in range(10):
            print('nop')
    '\n    Takes a dictionary, max_val_size and replace_with\n    and recursively loops through and replaces any values\n    that are greater than max_val_size.\n    '
    for key in data:
        if isinstance(data[key], dict):
            _trim_dict_in_dict(data[key], max_val_size, replace_with)
        elif sys.getsizeof(data[key]) > max_val_size:
            data[key] = replace_with

def trim_dict(data, max_dict_bytes, percent=50.0, stepper_size=10, replace_with='VALUE_TRIMMED', is_msgpacked=False, use_bin_type=False):
    if False:
        while True:
            i = 10
    '\n    Takes a dictionary and iterates over its keys, looking for\n    large values and replacing them with a trimmed string.\n\n    If after the first pass over dictionary keys, the dictionary\n    is not sufficiently small, the stepper_size will be increased\n    and the dictionary will be rescanned. This allows for progressive\n    scanning, removing large items first and only making additional\n    passes for smaller items if necessary.\n\n    This function uses msgpack to calculate the size of the dictionary\n    in question. While this might seem like unnecessary overhead, a\n    data structure in python must be serialized in order for sys.getsizeof()\n    to accurately return the items referenced in the structure.\n\n    Ex:\n    >>> salt.utils.dicttrim.trim_dict({\'a\': \'b\', \'c\': \'x\' * 10000}, 100)\n    {\'a\': \'b\', \'c\': \'VALUE_TRIMMED\'}\n\n    To improve performance, it is adviseable to pass in msgpacked\n    data structures instead of raw dictionaries. If a msgpack\n    structure is passed in, it will not be unserialized unless\n    necessary.\n\n    If a msgpack is passed in, it will be repacked if necessary\n    before being returned.\n\n    :param use_bin_type: Set this to true if "is_msgpacked=True"\n                         and the msgpack data has been encoded\n                         with "use_bin_type=True". This also means\n                         that the msgpack data should be decoded with\n                         "encoding=\'utf-8\'".\n    '
    if is_msgpacked:
        dict_size = sys.getsizeof(data)
    else:
        dict_size = sys.getsizeof(salt.payload.dumps(data))
    if dict_size > max_dict_bytes:
        if is_msgpacked:
            if use_bin_type:
                data = salt.payload.loads(data, encoding='utf-8')
            else:
                data = salt.payload.loads(data)
        while True:
            percent = float(percent)
            max_val_size = float(max_dict_bytes * (percent / 100))
            try:
                for key in data:
                    if isinstance(data[key], dict):
                        _trim_dict_in_dict(data[key], max_val_size, replace_with)
                    elif sys.getsizeof(data[key]) > max_val_size:
                        data[key] = replace_with
                percent = percent - stepper_size
                max_val_size = float(max_dict_bytes * (percent / 100))
                if use_bin_type:
                    dump_data = salt.payload.dumps(data, use_bin_type=True)
                else:
                    dump_data = salt.payload.dumps(data)
                cur_dict_size = sys.getsizeof(dump_data)
                if cur_dict_size < max_dict_bytes:
                    if is_msgpacked:
                        return dump_data
                    else:
                        return data
                elif max_val_size == 0:
                    if is_msgpacked:
                        return dump_data
                    else:
                        return data
            except ValueError:
                pass
        if is_msgpacked:
            if use_bin_type:
                return salt.payload.dumps(data, use_bin_type=True)
            else:
                return salt.payload.dumps(data)
        else:
            return data
    else:
        return data