from collections import defaultdict, Iterable, OrderedDict

def inverse_dicts(*dicts):
    if False:
        return 10
    '\n    Inverts the dicts, e.g. {1: 2, 3: 4} and {2: 3, 4: 4} will be inverted\n    {2: [1], 3: [2], 4: [3, 4]}. This also handles dictionaries\n    with Iterable items as values e.g. {1: [1, 2, 3], 2: [3, 4, 5]} and\n    {2: [1], 3: [2], 4: [3, 4]} will be inverted to\n    {1: [1, 2], 2: [1, 3], 3: [1, 2, 4], 4: [2, 4], 5: [2]}.\n    No order is preserved.\n\n    :param dicts: The dictionaries to invert.\n    :type dicts:  dict\n    :return:      The inversed dictionary which merges all dictionaries into\n                  one.\n    :rtype: defaultdict\n    '
    inverse = defaultdict(list)
    for dictionary in dicts:
        for (key, value) in dictionary.items():
            if isinstance(value, Iterable):
                for item in value:
                    inverse[item].append(key)
            else:
                inverse[value].append(key)
    return inverse

def update_ordered_dict_key(dictionary, old_key, new_key):
    if False:
        print('Hello World!')
    return OrderedDict(((new_key if k == old_key else k, v) for (k, v) in dictionary.items()))