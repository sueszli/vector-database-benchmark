import copy
import json
import os
from collections import OrderedDict
import yaml

class RamlLoader(yaml.SafeLoader):
    pass

def construct_include(loader, node):
    if False:
        i = 10
        return i + 15
    path = os.path.join(os.path.dirname(loader.stream.name), node.value)
    with open(path, encoding='utf-8') as f:
        return yaml.load(f, Loader=RamlLoader)

def construct_mapping(loader, node):
    if False:
        for i in range(10):
            print('nop')
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))
RamlLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
RamlLoader.add_constructor('!include', construct_include)

class RamlSpec:
    """
    This class loads the raml specification, and expose useful
    aspects of the spec

    Main usage for now is for the doc, but it can be extended to make sure
    raml spec matches other spec implemented in the tests
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        fn = os.path.join(os.path.dirname(__file__), os.pardir, 'spec', 'api.raml')
        with open(fn, encoding='utf-8') as f:
            self.api = yaml.load(f, Loader=RamlLoader)
        with open(fn, encoding='utf-8') as f:
            self.rawraml = f.read()
        endpoints = {}
        self.endpoints_by_type = {}
        self.rawendpoints = {}
        self.endpoints = self.parse_endpoints(endpoints, '', self.api)
        self.types = self.parse_types()

    def parse_endpoints(self, endpoints, base, api, uriParameters=None):
        if False:
            i = 10
            return i + 15
        if uriParameters is None:
            uriParameters = OrderedDict()
        for (k, v) in api.items():
            if k.startswith('/'):
                ep = base + k
                p = copy.deepcopy(uriParameters)
                if v is not None:
                    p.update(v.get('uriParameters', {}))
                    v['uriParameters'] = p
                    endpoints[ep] = v
                self.parse_endpoints(endpoints, ep, v, p)
            elif k in ['get', 'post']:
                if 'is' not in v:
                    continue
                for _is in v['is']:
                    if not isinstance(_is, dict):
                        raise RuntimeError(f'Unexpected "is" target {type(_is)}: {_is}')
                    if 'bbget' in _is:
                        try:
                            v['eptype'] = _is['bbget']['bbtype']
                        except TypeError as e:
                            raise RuntimeError(f"Unexpected 'is' target {_is['bbget']}") from e
                        self.endpoints_by_type.setdefault(v['eptype'], {})
                        self.endpoints_by_type[v['eptype']][base] = api
                    if 'bbgetraw' in _is:
                        self.rawendpoints.setdefault(base, {})
                        self.rawendpoints[base] = api
        return endpoints

    def reindent(self, s, indent):
        if False:
            while True:
                i = 10
        return s.replace('\n', '\n' + ' ' * indent)

    def format_json(self, j, indent):
        if False:
            i = 10
            return i + 15
        j = json.dumps(j, indent=4).replace(', \n', ',\n')
        return self.reindent(j, indent)

    def parse_types(self):
        if False:
            for i in range(10):
                print('nop')
        types = self.api['types']
        return types

    def iter_actions(self, endpoint):
        if False:
            i = 10
            return i + 15
        ACTIONS_MAGIC = '/actions/'
        for (k, v) in endpoint.items():
            if k.startswith(ACTIONS_MAGIC):
                k = k[len(ACTIONS_MAGIC):]
                v = v['post']
                v['body'] = v['body']['application/json'].get('properties', {})
                yield (k, v)