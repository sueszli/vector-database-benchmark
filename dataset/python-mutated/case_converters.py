import re

def to_upper_camel_case(snake_case_str):
    if False:
        while True:
            i = 10
    'Converts snake_case to UpperCamelCase.\n\n    Example\n    -------\n        foo_bar -> FooBar\n\n    '
    return ''.join(map(str.title, snake_case_str.split('_')))

def to_lower_camel_case(snake_case_str):
    if False:
        return 10
    'Converts snake_case to lowerCamelCase.\n\n    Example\n    -------\n        foo_bar -> fooBar\n        fooBar -> foobar\n\n    '
    words = snake_case_str.split('_')
    if len(words) > 1:
        capitalized = [w.title() for w in words]
        capitalized[0] = words[0]
        return ''.join(capitalized)
    else:
        return snake_case_str

def to_snake_case(camel_case_str):
    if False:
        print('Hello World!')
    'Converts UpperCamelCase and lowerCamelCase to snake_case.\n\n    Examples\n    --------\n        fooBar -> foo_bar\n        BazBang -> baz_bang\n\n    '
    s1 = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', camel_case_str)
    return re.sub('([a-z0-9])([A-Z])', '\\1_\\2', s1).lower()

def convert_dict_keys(func, in_dict):
    if False:
        return 10
    'Apply a conversion function to all keys in a dict.\n\n    Parameters\n    ----------\n    func : callable\n        The function to apply. Takes a str and returns a str.\n    in_dict : dict\n        The dictionary to convert. If some value in this dict is itself a dict,\n        it also gets recursively converted.\n\n    Returns\n    -------\n    dict\n        A new dict with all the contents of `in_dict`, but with the keys\n        converted by `func`.\n\n    '
    out_dict = dict()
    for (k, v) in in_dict.items():
        converted_key = func(k)
        if type(v) is dict:
            out_dict[converted_key] = convert_dict_keys(func, v)
        else:
            out_dict[converted_key] = v
    return out_dict