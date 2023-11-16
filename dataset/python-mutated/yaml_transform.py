import collections
import json
import logging
import os
import pprint
import re
import uuid
from typing import Any
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Set
import yaml
from yaml.loader import SafeLoader
import apache_beam as beam
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.transforms.fully_qualified_named_transform import FullyQualifiedNamedTransform
from apache_beam.yaml import yaml_provider
from apache_beam.yaml.yaml_combine import normalize_combine
__all__ = ['YamlTransform']
_LOGGER = logging.getLogger(__name__)
yaml_provider.fix_pycallable()
try:
    import jsonschema
except ImportError:
    jsonschema = None
if jsonschema is not None:
    with open(os.path.join(os.path.dirname(__file__), 'pipeline.schema.yaml')) as yaml_file:
        pipeline_schema = yaml.safe_load(yaml_file)

def validate_against_schema(pipeline):
    if False:
        for i in range(10):
            print('nop')
    try:
        jsonschema.validate(pipeline, pipeline_schema)
    except jsonschema.ValidationError as exn:
        exn.message += f' at line {SafeLineLoader.get_line(exn.instance)}'
        raise exn

def memoize_method(func):
    if False:
        print('Hello World!')

    def wrapper(self, *args):
        if False:
            while True:
                i = 10
        if not hasattr(self, '_cache'):
            self._cache = {}
        key = (func.__name__, args)
        if key not in self._cache:
            self._cache[key] = func(self, *args)
        return self._cache[key]
    return wrapper

def only_element(xs):
    if False:
        i = 10
        return i + 15
    (x,) = xs
    return x

def explicitly_empty():
    if False:
        for i in range(10):
            print('nop')
    return {'__explicitly_empty__': None}

def is_explicitly_empty(io):
    if False:
        i = 10
        return i + 15
    return io == explicitly_empty()

def is_empty(io):
    if False:
        i = 10
        return i + 15
    return not io or is_explicitly_empty(io)

def empty_if_explicitly_empty(io):
    if False:
        return 10
    if is_explicitly_empty(io):
        return {}
    else:
        return io

class SafeLineLoader(SafeLoader):
    """A yaml loader that attaches line information to mappings and strings."""

    class TaggedString(str):
        """A string class to which we can attach metadata.

    This is primarily used to trace a string's origin back to its place in a
    yaml file.
    """

        def __reduce__(self):
            if False:
                print('Hello World!')
            return (str, (str(self),))

    def construct_scalar(self, node):
        if False:
            i = 10
            return i + 15
        value = super().construct_scalar(node)
        if isinstance(value, str):
            value = SafeLineLoader.TaggedString(value)
            value._line_ = node.start_mark.line + 1
        return value

    def construct_mapping(self, node, deep=False):
        if False:
            i = 10
            return i + 15
        mapping = super().construct_mapping(node, deep=deep)
        mapping['__line__'] = node.start_mark.line + 1
        mapping['__uuid__'] = self.create_uuid()
        return mapping

    @classmethod
    def create_uuid(cls):
        if False:
            for i in range(10):
                print('nop')
        return str(uuid.uuid4())

    @classmethod
    def strip_metadata(cls, spec, tagged_str=True):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(spec, Mapping):
            return {key: cls.strip_metadata(value, tagged_str) for (key, value) in spec.items() if key not in ('__line__', '__uuid__')}
        elif isinstance(spec, Iterable) and (not isinstance(spec, (str, bytes))):
            return [cls.strip_metadata(value, tagged_str) for value in spec]
        elif isinstance(spec, SafeLineLoader.TaggedString) and tagged_str:
            return str(spec)
        else:
            return spec

    @staticmethod
    def get_line(obj):
        if False:
            return 10
        if isinstance(obj, dict):
            return obj.get('__line__', 'unknown')
        else:
            return getattr(obj, '_line_', 'unknown')

class LightweightScope(object):

    def __init__(self, transforms):
        if False:
            return 10
        self._transforms = transforms
        self._transforms_by_uuid = {t['__uuid__']: t for t in self._transforms}
        self._uuid_by_name = collections.defaultdict(set)
        for spec in self._transforms:
            if 'name' in spec:
                self._uuid_by_name[spec['name']].add(spec['__uuid__'])
            if 'type' in spec:
                self._uuid_by_name[spec['type']].add(spec['__uuid__'])

    def get_transform_id_and_output_name(self, name):
        if False:
            return 10
        if '.' in name:
            (transform_name, output) = name.rsplit('.', 1)
        else:
            (transform_name, output) = (name, None)
        return (self.get_transform_id(transform_name), output)

    def get_transform_id(self, transform_name):
        if False:
            i = 10
            return i + 15
        if transform_name in self._transforms_by_uuid:
            return transform_name
        else:
            candidates = self._uuid_by_name[transform_name]
            if not candidates:
                raise ValueError(f'Unknown transform at line {SafeLineLoader.get_line(transform_name)}: {transform_name}')
            elif len(candidates) > 1:
                raise ValueError(f'Ambiguous transform at line {SafeLineLoader.get_line(transform_name)}: {transform_name}')
            else:
                return only_element(candidates)

class Scope(LightweightScope):
    """To look up PCollections (typically outputs of prior transforms) by name."""

    def __init__(self, root, inputs: Mapping[str, Any], transforms: Iterable[dict], providers: Mapping[str, Iterable[yaml_provider.Provider]], input_providers: Iterable[yaml_provider.Provider]):
        if False:
            print('Hello World!')
        super().__init__(transforms)
        self.root = root
        self._inputs = inputs
        self.providers = providers
        self._seen_names: Set[str] = set()
        self.input_providers = input_providers
        self._all_followers = None

    def followers(self, transform_name):
        if False:
            while True:
                i = 10
        if self._all_followers is None:
            self._all_followers = collections.defaultdict(list)
            for transform in self._transforms:
                if transform['type'] != 'composite':
                    for input in empty_if_explicitly_empty(transform['input']).values():
                        (transform_id, _) = self.get_transform_id_and_output_name(input)
                        self._all_followers[transform_id].append(transform['__uuid__'])
        return self._all_followers[self.get_transform_id(transform_name)]

    def compute_all(self):
        if False:
            for i in range(10):
                print('nop')
        for transform_id in self._transforms_by_uuid.keys():
            self.compute_outputs(transform_id)

    def get_pcollection(self, name):
        if False:
            i = 10
            return i + 15
        if name in self._inputs:
            return self._inputs[name]
        elif '.' in name:
            (transform, output) = name.rsplit('.', 1)
            outputs = self.get_outputs(transform)
            if output in outputs:
                return outputs[output]
            elif len(outputs) == 1 and outputs[next(iter(outputs))].tag == output:
                return outputs[next(iter(outputs))]
            else:
                raise ValueError(f'Unknown output {repr(output)} at line {SafeLineLoader.get_line(name)}: {transform} only has outputs {list(outputs.keys())}')
        else:
            outputs = self.get_outputs(name)
            if len(outputs) == 1:
                return only_element(outputs.values())
            else:
                error_output = self._transforms_by_uuid[self.get_transform_id(name)]['config'].get('error_handling', {}).get('output')
                if error_output and error_output in outputs and (len(outputs) == 2):
                    return next((output for (tag, output) in outputs.items() if tag != error_output))
                raise ValueError(f'Ambiguous output at line {SafeLineLoader.get_line(name)}: {name} has outputs {list(outputs.keys())}')

    def get_outputs(self, transform_name):
        if False:
            while True:
                i = 10
        return self.compute_outputs(self.get_transform_id(transform_name))

    @memoize_method
    def compute_outputs(self, transform_id):
        if False:
            print('Hello World!')
        return expand_transform(self._transforms_by_uuid[transform_id], self)

    def best_provider(self, t, input_providers: yaml_provider.Iterable[yaml_provider.Provider]):
        if False:
            while True:
                i = 10
        if isinstance(t, dict):
            spec = t
        else:
            spec = self._transforms_by_uuid[self.get_transform_id(t)]
        possible_providers = [p for p in self.providers[spec['type']] if p.available()]
        if not possible_providers:
            raise ValueError('No available provider for type %r at %s' % (spec['type'], identify_object(spec)))
        if len(possible_providers) == 1:
            return possible_providers[0]

        def best_matches(possible_providers: Iterable[yaml_provider.Provider], adjacent_provider_options: Iterable[Iterable[yaml_provider.Provider]]) -> List[yaml_provider.Provider]:
            if False:
                print('Hello World!')
            "Given a set of possible providers, and a set of providers for each\n      adjacent transform, returns the top possible providers as ranked by\n      affinity to the adjacent transforms' providers.\n      "
            providers_by_score = collections.defaultdict(list)
            for p in possible_providers:
                providers_by_score[sum((max((p.affinity(ap) for ap in apo)) for apo in adjacent_provider_options))].append(p)
            return providers_by_score[max(providers_by_score.keys())]
        if input_providers:
            possible_providers = best_matches(possible_providers, [[p] for p in input_providers])
        if '__uuid__' not in spec:
            return possible_providers[0]
        if len(possible_providers) > 1:
            adjacent_transforms = list(self.followers(spec['__uuid__']))
            while adjacent_transforms:
                adjacent_provider_options = [[p for p in self.providers[self._transforms_by_uuid[t]['type']] if p.available()] for t in adjacent_transforms]
                if any((not apo for apo in adjacent_provider_options)):
                    break
                possible_providers = best_matches(possible_providers, adjacent_provider_options)
                if len(possible_providers) == 1:
                    break
                adjacent_transforms = sum([list(self.followers(t)) for t in adjacent_transforms], [])
        return possible_providers[0]

    def create_ptransform(self, spec, input_pcolls):
        if False:
            print('Hello World!')
        if 'type' not in spec:
            raise ValueError(f'Missing transform type: {identify_object(spec)}')
        if spec['type'] not in self.providers:
            raise ValueError('Unknown transform type %r at %s' % (spec['type'], identify_object(spec)))
        providers_by_input = {k: v for (k, v) in self.input_providers.items()}
        input_providers = [providers_by_input[pcoll] for pcoll in input_pcolls if pcoll in providers_by_input]
        provider = self.best_provider(spec, input_providers)
        config = SafeLineLoader.strip_metadata(spec.get('config', {}))
        if not isinstance(config, dict):
            raise ValueError('Config for transform at %s must be a mapping.' % identify_object(spec))
        if not input_pcolls and (not is_explicitly_empty(spec.get('input', {}))) and provider.requires_inputs(spec['type'], config):
            raise ValueError(f'Missing inputs for transform at {identify_object(spec)}')
        try:
            ptransform = provider.create_transform(spec['type'], config, self.create_ptransform)
            annotations = dict(yaml_type=spec['type'], yaml_args=json.dumps(config), yaml_provider=json.dumps(provider.to_json()), **ptransform.annotations())
            ptransform.annotations = lambda : annotations
            original_expand = ptransform.expand

            def recording_expand(pvalue):
                if False:
                    for i in range(10):
                        print('nop')
                result = original_expand(pvalue)

                def record_providers(pvalueish):
                    if False:
                        print('Hello World!')
                    if isinstance(pvalueish, (tuple, list)):
                        for p in pvalueish:
                            record_providers(p)
                    elif isinstance(pvalueish, dict):
                        for p in pvalueish.values():
                            record_providers(p)
                    elif isinstance(pvalueish, beam.PCollection):
                        if pvalueish not in self.input_providers:
                            self.input_providers[pvalueish] = provider
                record_providers(result)
                return result
            ptransform.expand = recording_expand
            return ptransform
        except Exception as exn:
            if isinstance(exn, TypeError):
                msg = str(exn).replace('positional', '').replace('keyword', '')
                msg = re.sub('\\S+lambda\\S+', '', msg)
                msg = re.sub('  +', ' ', msg).strip()
            else:
                msg = str(exn)
            raise ValueError(f'Invalid transform specification at {identify_object(spec)}: {msg}') from exn

    def unique_name(self, spec, ptransform, strictness=0):
        if False:
            return 10
        if 'name' in spec:
            name = spec['name']
            strictness += 1
        elif 'ExternalTransform' not in ptransform.label:
            name = ptransform.label
        else:
            name = spec['type']
        if name in self._seen_names:
            if strictness >= 2:
                raise ValueError(f'Duplicate name at {identify_object(spec)}: {name}')
            else:
                name = f'{name}@{SafeLineLoader.get_line(spec)}'
        self._seen_names.add(name)
        return name

def expand_transform(spec, scope):
    if False:
        return 10
    if 'type' not in spec:
        raise TypeError(f'Missing type parameter for transform at {identify_object(spec)}')
    type = spec['type']
    if type == 'composite':
        return expand_composite_transform(spec, scope)
    else:
        return expand_leaf_transform(spec, scope)

def expand_leaf_transform(spec, scope):
    if False:
        i = 10
        return i + 15
    spec = normalize_inputs_outputs(spec)
    inputs_dict = {key: scope.get_pcollection(value) for (key, value) in empty_if_explicitly_empty(spec['input']).items()}
    input_type = spec.get('input_type', 'default')
    if input_type == 'list':
        inputs = tuple(inputs_dict.values())
    elif input_type == 'map':
        inputs = inputs_dict
    elif len(inputs_dict) == 0:
        inputs = scope.root
    elif len(inputs_dict) == 1:
        inputs = next(iter(inputs_dict.values()))
    else:
        inputs = inputs_dict
    _LOGGER.info('Expanding %s ', identify_object(spec))
    ptransform = scope.create_ptransform(spec, inputs_dict.values())
    try:
        with FullyQualifiedNamedTransform.with_filter('*'):
            outputs = inputs | scope.unique_name(spec, ptransform) >> ptransform
    except Exception as exn:
        raise ValueError(f'Error apply transform {identify_object(spec)}: {exn}') from exn
    if isinstance(outputs, dict):
        return outputs
    elif isinstance(outputs, (tuple, list)):
        return {f'out{ix}': pcoll for (ix, pcoll) in enumerate(outputs)}
    elif isinstance(outputs, beam.PCollection):
        return {'out': outputs}
    elif outputs is None:
        return {}
    else:
        raise ValueError(f'Transform {identify_object(spec)} returned an unexpected type {type(outputs)}')

def expand_composite_transform(spec, scope):
    if False:
        while True:
            i = 10
    spec = normalize_inputs_outputs(normalize_source_sink(spec))
    inner_scope = Scope(scope.root, {key: scope.get_pcollection(value) for (key, value) in empty_if_explicitly_empty(spec['input']).items()}, spec['transforms'], yaml_provider.merge_providers(yaml_provider.parse_providers(spec.get('providers', [])), scope.providers), scope.input_providers)

    class CompositePTransform(beam.PTransform):

        @staticmethod
        def expand(inputs):
            if False:
                for i in range(10):
                    print('nop')
            inner_scope.compute_all()
            return {key: inner_scope.get_pcollection(value) for (key, value) in spec['output'].items()}
    if 'name' not in spec:
        spec['name'] = 'Composite'
    if spec['name'] is None:
        return CompositePTransform.expand(None)
    else:
        _LOGGER.info('Expanding %s ', identify_object(spec))
        return ({key: scope.get_pcollection(value) for (key, value) in empty_if_explicitly_empty(spec['input']).items()} or scope.root) | scope.unique_name(spec, None) >> CompositePTransform()

def expand_chain_transform(spec, scope):
    if False:
        i = 10
        return i + 15
    return expand_composite_transform(chain_as_composite(spec), scope)

def chain_as_composite(spec):
    if False:
        i = 10
        return i + 15

    def is_not_output_of_last_transform(new_transforms, value):
        if False:
            for i in range(10):
                print('nop')
        return 'name' in new_transforms[-1] and value != new_transforms[-1]['name'] or ('type' in new_transforms[-1] and value != new_transforms[-1]['type'])
    spec = normalize_source_sink(spec)
    if 'transforms' not in spec:
        raise TypeError(f'Chain at {identify_object(spec)} missing transforms property.')
    has_explicit_outputs = 'output' in spec
    composite_spec = normalize_inputs_outputs(tag_explicit_inputs(spec))
    new_transforms = []
    for (ix, transform) in enumerate(composite_spec['transforms']):
        if any((io in transform for io in ('input', 'output'))):
            if ix == 0 and 'input' in transform and ('output' not in transform) and is_explicitly_empty(transform['input']):
                pass
            else:
                raise ValueError(f'Transform {identify_object(transform)} is part of a chain, must have implicit inputs and outputs.')
        if ix == 0:
            if is_explicitly_empty(transform.get('input', None)):
                pass
            elif is_explicitly_empty(composite_spec['input']):
                transform['input'] = composite_spec['input']
            elif is_empty(composite_spec['input']):
                del composite_spec['input']
            else:
                transform['input'] = {key: key for key in composite_spec['input'].keys()}
        else:
            transform['input'] = new_transforms[-1]['__uuid__']
        new_transforms.append(transform)
    new_transforms.extend(spec.get('extra_transforms', []))
    composite_spec['transforms'] = new_transforms
    last_transform = new_transforms[-1]['__uuid__']
    if has_explicit_outputs:
        for (key, value) in composite_spec['output'].items():
            if is_not_output_of_last_transform(new_transforms, value):
                raise ValueError(f'Explicit output {identify_object(value)} of the chain transform is not an output of the last transform.')
        composite_spec['output'] = {key: f'{last_transform}.{value}' for (key, value) in composite_spec['output'].items()}
    else:
        composite_spec['output'] = last_transform
    if 'name' not in composite_spec:
        composite_spec['name'] = 'Chain'
    composite_spec['type'] = 'composite'
    return composite_spec

def preprocess_chain(spec):
    if False:
        while True:
            i = 10
    if spec['type'] == 'chain':
        return chain_as_composite(spec)
    else:
        return spec

def pipeline_as_composite(spec):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(spec, list):
        return {'type': 'composite', 'name': None, 'transforms': spec, '__line__': spec[0]['__line__'], '__uuid__': SafeLineLoader.create_uuid()}
    else:
        return dict(spec, name=None, type=spec.get('type', 'composite'))

def normalize_source_sink(spec):
    if False:
        i = 10
        return i + 15
    if 'source' not in spec and 'sink' not in spec:
        return spec
    spec = dict(spec)
    spec['transforms'] = list(spec.get('transforms', []))
    if 'source' in spec:
        if 'input' not in spec['source']:
            spec['source']['input'] = explicitly_empty()
        spec['transforms'].insert(0, spec.pop('source'))
    if 'sink' in spec:
        spec['transforms'].append(spec.pop('sink'))
    return spec

def preprocess_source_sink(spec):
    if False:
        i = 10
        return i + 15
    if spec['type'] in ('chain', 'composite'):
        return normalize_source_sink(spec)
    else:
        return spec

def tag_explicit_inputs(spec):
    if False:
        print('Hello World!')
    if 'input' in spec and (not SafeLineLoader.strip_metadata(spec['input'])):
        return dict(spec, input=explicitly_empty())
    else:
        return spec

def normalize_inputs_outputs(spec):
    if False:
        for i in range(10):
            print('nop')
    spec = dict(spec)

    def normalize_io(tag):
        if False:
            while True:
                i = 10
        io = spec.get(tag, {})
        if isinstance(io, (str, list)):
            return {tag: io}
        else:
            return SafeLineLoader.strip_metadata(io, tagged_str=False)
    return dict(spec, input=normalize_io('input'), output=normalize_io('output'))

def identify_object(spec):
    if False:
        return 10
    line = SafeLineLoader.get_line(spec)
    name = extract_name(spec)
    if name:
        return f'"{name}" at line {line}'
    else:
        return f'at line {line}'

def extract_name(spec):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(spec, dict):
        if 'name' in spec:
            return spec['name']
        elif 'id' in spec:
            return spec['id']
        elif 'type' in spec:
            return spec['type']
        elif len(spec) == 1:
            return extract_name(next(iter(spec.values())))
        else:
            return ''
    elif isinstance(spec, str):
        return spec
    else:
        return ''

def push_windowing_to_roots(spec):
    if False:
        print('Hello World!')
    scope = LightweightScope(spec['transforms'])
    consumed_outputs_by_transform = collections.defaultdict(set)
    for transform in spec['transforms']:
        for (_, input_ref) in empty_if_explicitly_empty(transform['input']).items():
            try:
                (transform_id, output) = scope.get_transform_id_and_output_name(input_ref)
                consumed_outputs_by_transform[transform_id].add(output)
            except ValueError:
                pass
    for transform in spec['transforms']:
        if is_empty(transform['input']) and 'windowing' not in transform:
            transform['windowing'] = spec['windowing']
            transform['__consumed_outputs'] = consumed_outputs_by_transform[transform['__uuid__']]
    return spec

def preprocess_windowing(spec):
    if False:
        for i in range(10):
            print('nop')
    if spec['type'] == 'WindowInto':
        if 'windowing' in spec:
            spec['config'] = spec.get('config', {})
            spec['config']['windowing'] = spec.pop('windowing')
        return spec
    elif 'windowing' not in spec:
        return spec
    if spec['type'] == 'composite':
        spec = push_windowing_to_roots(spec)
    windowing = spec.pop('windowing')
    if not is_empty(spec['input']):
        original_inputs = spec['input']
        windowing_transforms = [{'type': 'WindowInto', 'name': f'WindowInto[{key}]', 'windowing': windowing, 'input': key, '__line__': spec['__line__'], '__uuid__': SafeLineLoader.create_uuid()} for key in original_inputs.keys()]
        windowed_inputs = {key: t['__uuid__'] for (key, t) in zip(original_inputs.keys(), windowing_transforms)}
        modified_spec = dict(spec, input=windowed_inputs, __uuid__=SafeLineLoader.create_uuid())
        return {'type': 'composite', 'name': spec.get('name', None) or spec['type'], 'transforms': [modified_spec] + windowing_transforms, 'input': spec['input'], 'output': modified_spec['__uuid__'], '__line__': spec['__line__'], '__uuid__': spec['__uuid__']}
    elif spec['type'] == 'composite':
        return spec
    else:
        consumed_outputs = list(spec.pop('__consumed_outputs', {None}))
        modified_spec = dict(spec, __uuid__=SafeLineLoader.create_uuid())
        windowing_transforms = [{'type': 'WindowInto', 'name': f'WindowInto[{out}]', 'windowing': windowing, 'input': {'input': modified_spec['__uuid__'] + ('.' + out if out else '')}, '__line__': spec['__line__'], '__uuid__': SafeLineLoader.create_uuid()} for out in consumed_outputs]
        if consumed_outputs == [None]:
            windowed_outputs = only_element(windowing_transforms)['__uuid__']
        else:
            windowed_outputs = {out: t['__uuid__'] for (out, t) in zip(consumed_outputs, windowing_transforms)}
        return {'type': 'composite', 'name': spec.get('name', None) or spec['type'], 'transforms': [modified_spec] + windowing_transforms, 'output': windowed_outputs, '__line__': spec['__line__'], '__uuid__': spec['__uuid__']}

def preprocess_flattened_inputs(spec):
    if False:
        i = 10
        return i + 15
    if spec['type'] != 'composite':
        return spec
    new_transforms = []
    for t in spec['transforms']:
        if t['type'] == 'Flatten':

            def all_inputs(t):
                if False:
                    i = 10
                    return i + 15
                for (key, values) in t.get('input', {}).items():
                    if isinstance(values, list):
                        for (ix, values) in enumerate(values):
                            yield (f'{key}{ix}', values)
                    else:
                        yield (key, values)
            inputs_dict = {}
            for (key, value) in all_inputs(t):
                while key in inputs_dict:
                    key += '_'
                inputs_dict[key] = value
            t = dict(t, input=inputs_dict)
        else:
            replaced_inputs = {}
            for (key, values) in t.get('input', {}).items():
                if isinstance(values, list):
                    flatten_id = SafeLineLoader.create_uuid()
                    new_transforms.append({'type': 'Flatten', 'name': '%s-Flatten[%s]' % (t.get('name', t['type']), key), 'input': {f'input{ix}': value for (ix, value) in enumerate(values)}, '__line__': spec['__line__'], '__uuid__': flatten_id})
                    replaced_inputs[key] = flatten_id
            if replaced_inputs:
                t = dict(t, input={**t['input'], **replaced_inputs})
        new_transforms.append(t)
    return dict(spec, transforms=new_transforms)

def ensure_transforms_have_types(spec):
    if False:
        for i in range(10):
            print('nop')
    if 'type' not in spec:
        raise ValueError(f'Missing type specification in {identify_object(spec)}')
    return spec

def ensure_errors_consumed(spec):
    if False:
        while True:
            i = 10
    if spec['type'] == 'composite':
        scope = LightweightScope(spec['transforms'])
        to_handle = {}
        consumed = set((scope.get_transform_id_and_output_name(output) for output in spec['output'].values()))
        for t in spec['transforms']:
            config = t.get('config', t)
            if 'error_handling' in config:
                if 'output' not in config['error_handling']:
                    raise ValueError(f'Missing output in error_handling of {identify_object(t)}')
                to_handle[t['__uuid__'], config['error_handling']['output']] = t
            for (_, input) in empty_if_explicitly_empty(t['input']).items():
                if input not in spec['input']:
                    consumed.add(scope.get_transform_id_and_output_name(input))
        for (error_pcoll, t) in to_handle.items():
            if error_pcoll not in consumed:
                raise ValueError(f'Unconsumed error output for {identify_object(t)}.')
    return spec

def lift_config(spec):
    if False:
        print('Hello World!')
    if 'config' not in spec:
        common_params = ('name', 'type', 'input', 'output', 'transforms')
        return {'config': {k: v for (k, v) in spec.items() if k not in common_params}, **{k: v for (k, v) in spec.items() if k in common_params or k in ('__line__', '__uuid__')}}
    else:
        return spec

def ensure_config(spec):
    if False:
        while True:
            i = 10
    if 'config' not in spec:
        spec['config'] = {}
    return spec

def preprocess(spec, verbose=False, known_transforms=None):
    if False:
        return 10
    if verbose:
        pprint.pprint(spec)

    def apply(phase, spec):
        if False:
            for i in range(10):
                print('nop')
        spec = phase(spec)
        if spec['type'] in {'composite', 'chain'} and 'transforms' in spec:
            spec = dict(spec, transforms=[apply(phase, t) for t in spec['transforms']])
        return spec
    if known_transforms:
        known_transforms = set(known_transforms).union(['chain', 'composite'])

    def ensure_transforms_have_providers(spec):
        if False:
            print('Hello World!')
        if known_transforms:
            if spec['type'] not in known_transforms:
                raise ValueError(f"Unknown type or missing provider for type {spec['type']} for {identify_object(spec)}")
        return spec

    def preprocess_langauges(spec):
        if False:
            for i in range(10):
                print('nop')
        if spec['type'] in ('Filter', 'MapToFields', 'Combine'):
            language = spec.get('config', {}).get('language', 'generic')
            new_type = spec['type'] + '-' + language
            if known_transforms and new_type not in known_transforms:
                if language == 'generic':
                    raise ValueError(f'Missing language for {identify_object(spec)}')
                else:
                    raise ValueError(f'Unknown language {language} for {identify_object(spec)}')
            return dict(spec, type=new_type, name=spec.get('name', spec['type']))
        else:
            return spec
    for phase in [ensure_transforms_have_types, normalize_combine, preprocess_langauges, ensure_transforms_have_providers, preprocess_source_sink, preprocess_chain, tag_explicit_inputs, normalize_inputs_outputs, preprocess_flattened_inputs, ensure_errors_consumed, preprocess_windowing, ensure_config]:
        spec = apply(phase, spec)
        if verbose:
            print('=' * 20, phase, '=' * 20)
            pprint.pprint(spec)
    return spec

class YamlTransform(beam.PTransform):

    def __init__(self, spec, providers={}):
        if False:
            return 10
        if isinstance(spec, str):
            spec = yaml.load(spec, Loader=SafeLineLoader)
        if isinstance(providers, dict):
            providers = {key: yaml_provider.as_provider_list(key, value) for (key, value) in providers.items()}
        self._providers = yaml_provider.merge_providers(providers, yaml_provider.standard_providers())
        self._spec = preprocess(spec, known_transforms=self._providers.keys())
        self._was_chain = spec['type'] == 'chain'

    def expand(self, pcolls):
        if False:
            return 10
        if isinstance(pcolls, beam.pvalue.PBegin):
            root = pcolls
            pipeline = root.pipeline
            pcolls = {}
        elif isinstance(pcolls, beam.PCollection):
            root = pcolls.pipeline
            pipeline = root
            pcolls = {'input': pcolls}
            if not self._spec['input']:
                self._spec['input'] = {'input': 'input'}
                if self._was_chain and self._spec['transforms']:
                    self._spec['transforms'][0]['input'] = self._spec['input']
        else:
            root = next(iter(pcolls.values())).pipeline
            pipeline = root
            if not self._spec['input']:
                self._spec['input'] = {name: name for name in pcolls.keys()}
        python_provider = yaml_provider.InlineProvider({})
        options = pipeline.options.view_as(GoogleCloudOptions)
        options.labels = ['yaml=true']
        result = expand_transform(self._spec, Scope(root, pcolls, transforms=[self._spec], providers=self._providers, input_providers={pcoll: python_provider for pcoll in pcolls.values()}))
        if len(result) == 1:
            return only_element(result.values())
        else:
            return result

def expand_pipeline(pipeline, pipeline_spec, providers=None, validate_schema=jsonschema is not None):
    if False:
        i = 10
        return i + 15
    if isinstance(pipeline_spec, str):
        pipeline_spec = yaml.load(pipeline_spec, Loader=SafeLineLoader)
    if validate_schema:
        validate_against_schema(pipeline_spec)
    return YamlTransform(pipeline_as_composite(pipeline_spec['pipeline']), {**yaml_provider.parse_providers(pipeline_spec.get('providers', [])), **{key: yaml_provider.as_provider_list(key, value) for (key, value) in (providers or {}).items()}}).expand(beam.pvalue.PBegin(pipeline))