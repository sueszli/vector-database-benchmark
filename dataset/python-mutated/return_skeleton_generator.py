from __future__ import annotations
from collections import OrderedDict
import json
import sys
import yaml

def represent_ordereddict(dumper, data):
    if False:
        while True:
            i = 10
    value = []
    for (item_key, item_value) in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)
        value.append((node_key, node_value))
    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)

def get_return_data(key, value):
    if False:
        i = 10
        return i + 15
    returns_info = {key: OrderedDict()}
    returns_info[key]['description'] = 'FIXME *** add description for %s' % key
    returns_info[key]['returned'] = 'always'
    if isinstance(value, dict):
        returns_info[key]['type'] = 'complex'
        returns_info[key]['contains'] = get_all_items(value)
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        returns_info[key]['type'] = 'complex'
        returns_info[key]['contains'] = get_all_items(value[0])
    else:
        returns_info[key]['type'] = type(value).__name__
        returns_info[key]['sample'] = value
        if returns_info[key]['type'] == 'unicode':
            returns_info[key]['type'] = 'str'
    return returns_info

def get_all_items(data):
    if False:
        for i in range(10):
            print('nop')
    items = sorted([get_return_data(key, value) for (key, value) in data.items()])
    result = OrderedDict()
    for item in items:
        (key, value) = item.items()[0]
        result[key] = value
    return result

def main(args):
    if False:
        for i in range(10):
            print('nop')
    yaml.representer.SafeRepresenter.add_representer(OrderedDict, represent_ordereddict)
    if args:
        src = open(args[0])
    else:
        src = sys.stdin
    data = json.load(src, strict=False)
    docs = get_all_items(data)
    if 'invocation' in docs:
        del docs['invocation']
    print(yaml.safe_dump(docs, default_flow_style=False))
if __name__ == '__main__':
    main(sys.argv[1:])