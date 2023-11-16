"""This module defines the basic Combine operation."""
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Optional
import apache_beam as beam
from apache_beam import typehints
from apache_beam.typehints import row_type
from apache_beam.typehints import trivial_inference
from apache_beam.typehints.decorators import get_type_hints
from apache_beam.typehints.schemas import named_fields_from_element_type
from apache_beam.utils import python_callable
from apache_beam.yaml import options
from apache_beam.yaml import yaml_mapping
from apache_beam.yaml import yaml_provider
BUILTIN_COMBINE_FNS = {'sum': sum, 'max': max, 'min': min, 'all': all, 'any': any, 'mean': beam.transforms.combiners.MeanCombineFn(), 'count': beam.transforms.combiners.CountCombineFn()}

def normalize_combine(spec):
    if False:
        for i in range(10):
            print('nop')
    "Expands various shorthand specs for combine (which can otherwise be quite\n  verbose for simple cases.)  We do this here so that it doesn't need to be done\n  per language.  The following are all equivalent::\n\n      dest: fn_type\n\n      dest:\n        value: dest\n        fn: fn_type\n\n      dest:\n        value: dest\n        fn:\n          type: fn_type\n  "
    from apache_beam.yaml.yaml_transform import SafeLineLoader
    if spec['type'] == 'Combine':
        config = spec.get('config')
        if isinstance(config.get('group_by'), str):
            config['group_by'] = [config['group_by']]

        def normalize_agg(dest, agg):
            if False:
                while True:
                    i = 10
            if isinstance(agg, str):
                agg = {'fn': agg}
            if 'value' not in agg and spec.get('language') != 'sql':
                agg['value'] = dest
            if isinstance(agg['fn'], str):
                agg['fn'] = {'type': agg['fn']}
            return agg
        if 'combine' not in config:
            raise ValueError('Missing combine parameter in Combine config.')
        config['combine'] = {dest: normalize_agg(dest, agg) for (dest, agg) in SafeLineLoader.strip_metadata(config['combine']).items()}
    return spec

class PyJsYamlCombine(beam.PTransform):

    def __init__(self, group_by: Iterable[str], combine: Mapping[str, Mapping[str, Any]], language: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        self._group_by = group_by
        self._combine = combine
        self._language = language

    def expand(self, pcoll):
        if False:
            for i in range(10):
                print('nop')
        options.YamlOptions.check_enabled(pcoll.pipeline, 'Combine')
        input_types = dict(named_fields_from_element_type(pcoll.element_type))
        all_fields = list(input_types.keys())
        unknown_keys = set(self._group_by) - set(all_fields)
        if unknown_keys:
            raise ValueError(f'Unknown grouping columns: {list(unknown_keys)}')

        def create_combine_fn(fn_spec):
            if False:
                return 10
            if 'type' not in fn_spec:
                raise ValueError(f'CombineFn spec missing type: {fn_spec}')
            elif fn_spec['type'] in BUILTIN_COMBINE_FNS:
                return BUILTIN_COMBINE_FNS[fn_spec['type']]
            elif self._language == 'python':
                fn = python_callable.PythonCallableWithSource.load_from_source(fn_spec['type'])
                if 'config' in fn_spec:
                    fn = fn(**fn_spec['config'])
                return fn
            else:
                raise TypeError('Unknown CombineFn: {fn_spec}')

        def extract_return_type(expr):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(expr, str) and expr in input_types:
                return input_types[expr]
            expr_hints = get_type_hints(expr)
            if expr_hints and expr_hints.has_simple_output_type() and (expr_hints.simple_output_type(None) != typehints.Any):
                return expr_hints.simple_output_type(None)
            elif callable(expr):
                return trivial_inference.infer_return_type(expr, [pcoll.element_type])
            else:
                return Any
        transform = beam.GroupBy(*self._group_by)
        output_types = [(k, input_types[k]) for k in self._group_by]
        for (output, agg) in self._combine.items():
            expr = yaml_mapping._as_callable(all_fields, agg['value'], 'Combine', self._language)
            fn = create_combine_fn(agg['fn'])
            transform = transform.aggregate_field(expr, fn, output)
            expr_type = extract_return_type(expr)
            print('expr', expr, 'expr_type', expr_type)
            if isinstance(fn, beam.CombineFn):
                combined_type = extract_return_type(fn)
            elif fn in (sum, min, max):
                combined_type = expr_type
            elif fn in (any, all):
                combined_type = bool
            else:
                combined_type = Any
            output_types.append((output, combined_type))
        return pcoll | transform.with_output_types(row_type.RowTypeConstraint.from_fields(output_types))

@beam.ptransform.ptransform_fn
def _SqlCombineTransform(pcoll, sql_transform_constructor, group_by, combine, language=None):
    if False:
        for i in range(10):
            print('nop')
    options.YamlOptions.check_enabled(pcoll.pipeline, 'Combine')
    all_fields = [x for (x, _) in named_fields_from_element_type(pcoll.element_type)]
    unknown_keys = set(group_by) - set(all_fields)
    if unknown_keys:
        raise ValueError(f'Unknown grouping columns: {list(unknown_keys)}')

    def combine_col(dest, fn_spec):
        if False:
            return 10
        if 'value' in fn_spec or 'config' in fn_spec['fn']:
            expr = '%s(%s)' % (fn_spec['fn']['type'], ', '.join([fn_spec['value']] + list(fn_spec['fn'].get('config', {}).values())))
        else:
            expr = fn_spec['fn']['type']
        return f'{expr} as {dest}'
    return pcoll | sql_transform_constructor('SELECT %s FROM PCOLLECTION GROUP BY %s' % (', '.join(list(group_by) + [combine_col(dest, fn_spec) for (dest, fn_spec) in combine.items()]), ', '.join(group_by)))

def create_combine_providers():
    if False:
        print('Hello World!')
    return [yaml_provider.InlineProvider({'Combine-generic': PyJsYamlCombine, 'Combine-python': PyJsYamlCombine, 'Combine-javascript': PyJsYamlCombine}), yaml_provider.SqlBackedProvider({'Combine-generic': _SqlCombineTransform, 'Combine-sql': _SqlCombineTransform, 'Combine-calcite': _SqlCombineTransform})]