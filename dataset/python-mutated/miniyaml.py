from collections import OrderedDict
import cimodel.lib.miniutils as miniutils
LIST_MARKER = '- '
INDENTATION_WIDTH = 2

def is_dict(data):
    if False:
        for i in range(10):
            print('nop')
    return type(data) in [dict, OrderedDict]

def is_collection(data):
    if False:
        while True:
            i = 10
    return is_dict(data) or type(data) is list

def render(fh, data, depth, is_list_member=False):
    if False:
        while True:
            i = 10
    '\n    PyYaml does not allow precise control over the quoting\n    behavior, especially for merge references.\n    Therefore, we use this custom YAML renderer.\n    '
    indentation = ' ' * INDENTATION_WIDTH * depth
    if is_dict(data):
        tuples = list(data.items())
        if type(data) is not OrderedDict:
            tuples.sort()
        for (i, (k, v)) in enumerate(tuples):
            if not v:
                continue
            list_marker_prefix = LIST_MARKER if is_list_member and (not i) else ''
            trailing_whitespace = '\n' if is_collection(v) else ' '
            fh.write(indentation + list_marker_prefix + k + ':' + trailing_whitespace)
            render(fh, v, depth + 1 + int(is_list_member))
    elif type(data) is list:
        for v in data:
            render(fh, v, depth, True)
    else:
        modified_data = miniutils.quote(data) if data == '' else data
        list_member_prefix = indentation + LIST_MARKER if is_list_member else ''
        fh.write(list_member_prefix + str(modified_data) + '\n')