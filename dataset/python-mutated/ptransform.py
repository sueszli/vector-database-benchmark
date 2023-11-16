"""PTransform and descendants.

A PTransform is an object describing (not executing) a computation. The actual
execution semantics for a transform is captured by a runner object. A transform
object always belongs to a pipeline object.

A PTransform derived class needs to define the expand() method that describes
how one or more PValues are created by the transform.

The module defines a few standard transforms: FlatMap (parallel do),
GroupByKey (group by key), etc. Note that the expand() methods for these
classes contain code that will add nodes to the processing graph associated
with a pipeline.

As support for the FlatMap transform, the module also defines a DoFn
class and wrapper class that allows lambda functions to be used as
FlatMap processing functions.
"""
import copy
import itertools
import json
import logging
import operator
import os
import sys
import threading
import warnings
from functools import reduce
from functools import wraps
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import overload
from google.protobuf import message
from apache_beam import error
from apache_beam import pvalue
from apache_beam.internal import pickler
from apache_beam.internal import util
from apache_beam.portability import python_urns
from apache_beam.pvalue import DoOutputsTuple
from apache_beam.transforms import resources
from apache_beam.transforms.display import DisplayDataItem
from apache_beam.transforms.display import HasDisplayData
from apache_beam.transforms.sideinputs import SIDE_INPUT_PREFIX
from apache_beam.typehints import native_type_compatibility
from apache_beam.typehints import typehints
from apache_beam.typehints.decorators import IOTypeHints
from apache_beam.typehints.decorators import TypeCheckError
from apache_beam.typehints.decorators import WithTypeHints
from apache_beam.typehints.decorators import get_signature
from apache_beam.typehints.decorators import get_type_hints
from apache_beam.typehints.decorators import getcallargs_forhints
from apache_beam.typehints.trivial_inference import instance_to_type
from apache_beam.typehints.typehints import validate_composite_type_param
from apache_beam.utils import proto_utils
from apache_beam.utils import python_callable
if TYPE_CHECKING:
    from apache_beam import coders
    from apache_beam.pipeline import Pipeline
    from apache_beam.runners.pipeline_context import PipelineContext
    from apache_beam.transforms.core import Windowing
    from apache_beam.portability.api import beam_runner_api_pb2
__all__ = ['PTransform', 'ptransform_fn', 'label_from_callable', 'annotate_yaml']
_LOGGER = logging.getLogger(__name__)
T = TypeVar('T')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')
PTransformT = TypeVar('PTransformT', bound='PTransform')
ConstructorFn = Callable[['beam_runner_api_pb2.PTransform', Optional[Any], 'PipelineContext'], Any]
ptransform_fn_typehints_enabled = False

class _PValueishTransform(object):
    """Visitor for PValueish objects.

  A PValueish is a PValue, or list, tuple, dict of PValuesish objects.

  This visits a PValueish, contstructing a (possibly mutated) copy.
  """

    def visit_nested(self, node, *args):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(node, (tuple, list)):
            args = [self.visit(x, *args) for x in node]
            if isinstance(node, tuple) and hasattr(node.__class__, '_make'):
                return node.__class__(*args)
            else:
                return node.__class__(args)
        elif isinstance(node, dict):
            return node.__class__({key: self.visit(value, *args) for (key, value) in node.items()})
        else:
            return node

class _SetInputPValues(_PValueishTransform):

    def visit(self, node, replacements):
        if False:
            i = 10
            return i + 15
        if id(node) in replacements:
            return replacements[id(node)]
        else:
            return self.visit_nested(node, replacements)
_pipeline_materialization_cache = {}
_pipeline_materialization_lock = threading.Lock()

def _allocate_materialized_pipeline(pipeline):
    if False:
        for i in range(10):
            print('nop')
    pid = os.getpid()
    with _pipeline_materialization_lock:
        pipeline_id = id(pipeline)
        _pipeline_materialization_cache[pid, pipeline_id] = {}

def _allocate_materialized_result(pipeline):
    if False:
        for i in range(10):
            print('nop')
    pid = os.getpid()
    with _pipeline_materialization_lock:
        pipeline_id = id(pipeline)
        if (pid, pipeline_id) not in _pipeline_materialization_cache:
            raise ValueError('Materialized pipeline is not allocated for result cache.')
        result_id = len(_pipeline_materialization_cache[pid, pipeline_id])
        result = _MaterializedResult(pipeline_id, result_id)
        _pipeline_materialization_cache[pid, pipeline_id][result_id] = result
        return result

def _get_materialized_result(pipeline_id, result_id):
    if False:
        i = 10
        return i + 15
    pid = os.getpid()
    with _pipeline_materialization_lock:
        if (pid, pipeline_id) not in _pipeline_materialization_cache:
            raise Exception('Materialization in out-of-process and remote runners is not yet supported.')
        return _pipeline_materialization_cache[pid, pipeline_id][result_id]

def _release_materialized_pipeline(pipeline):
    if False:
        return 10
    pid = os.getpid()
    with _pipeline_materialization_lock:
        pipeline_id = id(pipeline)
        del _pipeline_materialization_cache[pid, pipeline_id]

class _MaterializedResult(object):

    def __init__(self, pipeline_id, result_id):
        if False:
            while True:
                i = 10
        self._pipeline_id = pipeline_id
        self._result_id = result_id
        self.elements = []

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        return (_get_materialized_result, (self._pipeline_id, self._result_id))

class _MaterializedDoOutputsTuple(pvalue.DoOutputsTuple):

    def __init__(self, deferred, results_by_tag):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(None, None, deferred._tags, deferred._main_tag)
        self._deferred = deferred
        self._results_by_tag = results_by_tag

    def __getitem__(self, tag):
        if False:
            print('Hello World!')
        if tag not in self._results_by_tag:
            raise KeyError('Tag %r is not a defined output tag of %s.' % (tag, self._deferred))
        return self._results_by_tag[tag].elements

class _AddMaterializationTransforms(_PValueishTransform):

    def _materialize_transform(self, pipeline):
        if False:
            return 10
        result = _allocate_materialized_result(pipeline)
        from apache_beam import DoFn
        from apache_beam import ParDo

        class _MaterializeValuesDoFn(DoFn):

            def process(self, element):
                if False:
                    while True:
                        i = 10
                result.elements.append(element)
        materialization_label = '_MaterializeValues%d' % result._result_id
        return (materialization_label >> ParDo(_MaterializeValuesDoFn()), result)

    def visit(self, node):
        if False:
            while True:
                i = 10
        if isinstance(node, pvalue.PValue):
            (transform, result) = self._materialize_transform(node.pipeline)
            node | transform
            return result
        elif isinstance(node, pvalue.DoOutputsTuple):
            results_by_tag = {}
            for tag in itertools.chain([node._main_tag], node._tags):
                results_by_tag[tag] = self.visit(node[tag])
            return _MaterializedDoOutputsTuple(node, results_by_tag)
        else:
            return self.visit_nested(node)

class _FinalizeMaterialization(_PValueishTransform):

    def visit(self, node):
        if False:
            return 10
        if isinstance(node, _MaterializedResult):
            return node.elements
        elif isinstance(node, _MaterializedDoOutputsTuple):
            return node
        else:
            return self.visit_nested(node)

def get_named_nested_pvalues(pvalueish, as_inputs=False):
    if False:
        return 10
    if isinstance(pvalueish, tuple):
        fields = getattr(pvalueish, '_fields', None)
        if fields and len(fields) == len(pvalueish):
            tagged_values = zip(fields, pvalueish)
        else:
            tagged_values = enumerate(pvalueish)
    elif isinstance(pvalueish, list):
        if as_inputs:
            yield (None, pvalueish)
            return
        tagged_values = enumerate(pvalueish)
    elif isinstance(pvalueish, dict):
        tagged_values = pvalueish.items()
    else:
        if as_inputs or isinstance(pvalueish, (pvalue.PValue, pvalue.DoOutputsTuple)):
            yield (None, pvalueish)
        return
    for (tag, subvalue) in tagged_values:
        for (subtag, subsubvalue) in get_named_nested_pvalues(subvalue, as_inputs=as_inputs):
            if subtag is None:
                yield (tag, subsubvalue)
            else:
                yield ('%s.%s' % (tag, subtag), subsubvalue)

class _ZipPValues(object):
    """Pairs each PValue in a pvalueish with a value in a parallel out sibling.

  Sibling should have the same nested structure as pvalueish.  Leaves in
  sibling are expanded across nested pvalueish lists, tuples, and dicts.
  For example

      ZipPValues().visit({'a': pc1, 'b': (pc2, pc3)},
                         {'a': 'A', 'b', 'B'})

  will return

      [('a', pc1, 'A'), ('b', pc2, 'B'), ('b', pc3, 'B')]
  """

    def visit(self, pvalueish, sibling, pairs=None, context=None):
        if False:
            for i in range(10):
                print('nop')
        if pairs is None:
            pairs = []
            self.visit(pvalueish, sibling, pairs, context)
            return pairs
        elif isinstance(pvalueish, (pvalue.PValue, pvalue.DoOutputsTuple)):
            pairs.append((context, pvalueish, sibling))
        elif isinstance(pvalueish, (list, tuple)):
            self.visit_sequence(pvalueish, sibling, pairs, context)
        elif isinstance(pvalueish, dict):
            self.visit_dict(pvalueish, sibling, pairs, context)

    def visit_sequence(self, pvalueish, sibling, pairs, context):
        if False:
            i = 10
            return i + 15
        if isinstance(sibling, (list, tuple)):
            for (ix, (p, s)) in enumerate(zip(pvalueish, list(sibling) + [None] * len(pvalueish))):
                self.visit(p, s, pairs, 'position %s' % ix)
        else:
            for p in pvalueish:
                self.visit(p, sibling, pairs, context)

    def visit_dict(self, pvalueish, sibling, pairs, context):
        if False:
            print('Hello World!')
        if isinstance(sibling, dict):
            for (key, p) in pvalueish.items():
                self.visit(p, sibling.get(key), pairs, key)
        else:
            for p in pvalueish.values():
                self.visit(p, sibling, pairs, context)

class PTransform(WithTypeHints, HasDisplayData, Generic[InputT, OutputT]):
    """A transform object used to modify one or more PCollections.

  Subclasses must define an expand() method that will be used when the transform
  is applied to some arguments. Typical usage pattern will be:

    input | CustomTransform(...)

  The expand() method of the CustomTransform object passed in will be called
  with input as an argument.
  """
    side_inputs = ()
    pipeline = None
    _user_label = None

    def __init__(self, label=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.label = label

    @property
    def label(self):
        if False:
            print('Hello World!')
        return self._user_label or self.default_label()

    @label.setter
    def label(self, value):
        if False:
            i = 10
            return i + 15
        self._user_label = value

    def default_label(self):
        if False:
            i = 10
            return i + 15
        return self.__class__.__name__

    def annotations(self) -> Dict[str, Union[bytes, str, message.Message]]:
        if False:
            while True:
                i = 10
        return {'python_type': f'{self.__class__.__module__}.{self.__class__.__qualname__}'}

    def default_type_hints(self):
        if False:
            while True:
                i = 10
        fn_type_hints = IOTypeHints.from_callable(self.expand)
        if fn_type_hints is not None:
            fn_type_hints = fn_type_hints.strip_pcoll()
        return get_type_hints(self.__class__).with_defaults(fn_type_hints)

    def with_input_types(self, input_type_hint):
        if False:
            print('Hello World!')
        'Annotates the input type of a :class:`PTransform` with a type-hint.\n\n    Args:\n      input_type_hint (type): An instance of an allowed built-in type, a custom\n        class, or an instance of a\n        :class:`~apache_beam.typehints.typehints.TypeConstraint`.\n\n    Raises:\n      TypeError: If **input_type_hint** is not a valid type-hint.\n        See\n        :obj:`apache_beam.typehints.typehints.validate_composite_type_param()`\n        for further details.\n\n    Returns:\n      PTransform: A reference to the instance of this particular\n      :class:`PTransform` object. This allows chaining type-hinting related\n      methods.\n    '
        input_type_hint = native_type_compatibility.convert_to_beam_type(input_type_hint)
        validate_composite_type_param(input_type_hint, 'Type hints for a PTransform')
        return super().with_input_types(input_type_hint)

    def with_output_types(self, type_hint):
        if False:
            return 10
        'Annotates the output type of a :class:`PTransform` with a type-hint.\n\n    Args:\n      type_hint (type): An instance of an allowed built-in type, a custom class,\n        or a :class:`~apache_beam.typehints.typehints.TypeConstraint`.\n\n    Raises:\n      TypeError: If **type_hint** is not a valid type-hint. See\n        :obj:`~apache_beam.typehints.typehints.validate_composite_type_param()`\n        for further details.\n\n    Returns:\n      PTransform: A reference to the instance of this particular\n      :class:`PTransform` object. This allows chaining type-hinting related\n      methods.\n    '
        type_hint = native_type_compatibility.convert_to_beam_type(type_hint)
        validate_composite_type_param(type_hint, 'Type hints for a PTransform')
        return super().with_output_types(type_hint)

    def with_resource_hints(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Adds resource hints to the :class:`PTransform`.\n\n    Resource hints allow users to express constraints on the environment where\n    the transform should be executed.  Interpretation of the resource hints is\n    defined by Beam Runners. Runners may ignore the unsupported hints.\n\n    Args:\n      **kwargs: key-value pairs describing hints and their values.\n\n    Raises:\n      ValueError: if provided hints are unknown to the SDK. See\n        :mod:`apache_beam.transforms.resources` for a list of known hints.\n\n    Returns:\n      PTransform: A reference to the instance of this particular\n      :class:`PTransform` object.\n    '
        self.get_resource_hints().update(resources.parse_resource_hints(kwargs))
        return self

    def get_resource_hints(self):
        if False:
            while True:
                i = 10
        if '_resource_hints' not in self.__dict__:
            self._resource_hints = {}
        return self._resource_hints

    def type_check_inputs(self, pvalueish):
        if False:
            for i in range(10):
                print('nop')
        self.type_check_inputs_or_outputs(pvalueish, 'input')

    def infer_output_type(self, unused_input_type):
        if False:
            return 10
        return self.get_type_hints().simple_output_type(self.label) or typehints.Any

    def type_check_outputs(self, pvalueish):
        if False:
            i = 10
            return i + 15
        self.type_check_inputs_or_outputs(pvalueish, 'output')

    def type_check_inputs_or_outputs(self, pvalueish, input_or_output):
        if False:
            return 10
        type_hints = self.get_type_hints()
        hints = getattr(type_hints, input_or_output + '_types')
        if hints is None or not any(hints):
            return
        (arg_hints, kwarg_hints) = hints
        if arg_hints and kwarg_hints:
            raise TypeCheckError('PTransform cannot have both positional and keyword type hints without overriding %s._type_check_%s()' % (self.__class__, input_or_output))
        root_hint = arg_hints[0] if len(arg_hints) == 1 else arg_hints or kwarg_hints
        for (context, pvalue_, hint) in _ZipPValues().visit(pvalueish, root_hint):
            if isinstance(pvalue_, DoOutputsTuple):
                continue
            if pvalue_.element_type is None:
                continue
            if hint and (not typehints.is_consistent_with(pvalue_.element_type, hint)):
                at_context = ' %s %s' % (input_or_output, context) if context else ''
                raise TypeCheckError('{type} type hint violation at {label}{context}: expected {hint}, got {actual_type}\nFull type hint:\n{debug_str}'.format(type=input_or_output.title(), label=self.label, context=at_context, hint=hint, actual_type=pvalue_.element_type, debug_str=type_hints.debug_str()))

    def _infer_output_coder(self, input_type=None, input_coder=None):
        if False:
            i = 10
            return i + 15
        'Returns the output coder to use for output of this transform.\n\n    The Coder returned here should not be wrapped in a WindowedValueCoder\n    wrapper.\n\n    Args:\n      input_type: An instance of an allowed built-in type, a custom class, or a\n        typehints.TypeConstraint for the input type, or None if not available.\n      input_coder: Coder object for encoding input to this PTransform, or None\n        if not available.\n\n    Returns:\n      Coder object for encoding output of this PTransform or None if unknown.\n    '
        return None

    def _clone(self, new_label):
        if False:
            for i in range(10):
                print('nop')
        'Clones the current transform instance under a new label.'
        transform = copy.copy(self)
        transform.label = new_label
        return transform

    def expand(self, input_or_inputs: InputT) -> OutputT:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s>' % self._str_internal()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s at %s>' % (self._str_internal(), hex(id(self)))

    def _str_internal(self):
        if False:
            i = 10
            return i + 15
        return '%s(PTransform)%s%s%s' % (self.__class__.__name__, ' label=[%s]' % self.label if hasattr(self, 'label') and self.label else '', ' inputs=%s' % str(self.inputs) if hasattr(self, 'inputs') and self.inputs else '', ' side_inputs=%s' % str(self.side_inputs) if self.side_inputs else '')

    def _check_pcollection(self, pcoll):
        if False:
            while True:
                i = 10
        if not isinstance(pcoll, pvalue.PCollection):
            raise error.TransformError('Expecting a PCollection argument.')
        if not pcoll.pipeline:
            raise error.TransformError('PCollection not part of a pipeline.')

    def get_windowing(self, inputs):
        if False:
            return 10
        "Returns the window function to be associated with transform's output.\n\n    By default most transforms just return the windowing function associated\n    with the input PCollection (or the first input if several).\n    "
        if inputs:
            return inputs[0].windowing
        else:
            from apache_beam.transforms.core import Windowing
            from apache_beam.transforms.window import GlobalWindows
            return Windowing(GlobalWindows())

    def __rrshift__(self, label):
        if False:
            print('Hello World!')
        return _NamedPTransform(self, label)

    def __or__(self, right):
        if False:
            while True:
                i = 10
        'Used to compose PTransforms, e.g., ptransform1 | ptransform2.'
        if isinstance(right, PTransform):
            return _ChainedPTransform(self, right)
        return NotImplemented

    def __ror__(self, left, label=None):
        if False:
            print('Hello World!')
        'Used to apply this PTransform to non-PValues, e.g., a tuple.'
        (pvalueish, pvalues) = self._extract_input_pvalues(left)
        if isinstance(pvalues, dict):
            pvalues = tuple(pvalues.values())
        pipelines = [v.pipeline for v in pvalues if isinstance(v, pvalue.PValue)]
        if pvalues and (not pipelines):
            deferred = False
            from apache_beam import pipeline
            from apache_beam.options.pipeline_options import PipelineOptions
            p = pipeline.Pipeline('DirectRunner', PipelineOptions(sys.argv))
        else:
            if not pipelines:
                if self.pipeline is not None:
                    p = self.pipeline
                else:
                    raise ValueError('"%s" requires a pipeline to be specified as there are no deferred inputs.' % self.label)
            else:
                p = self.pipeline or pipelines[0]
                for pp in pipelines:
                    if p != pp:
                        raise ValueError('Mixing values in different pipelines is not allowed.\n{%r} != {%r}' % (p, pp))
            deferred = not getattr(p.runner, 'is_eager', False)
        from apache_beam.transforms.core import Create
        replacements = {id(v): p | 'CreatePInput%s' % ix >> Create(v, reshuffle=False) for (ix, v) in enumerate(pvalues) if not isinstance(v, pvalue.PValue) and v is not None}
        pvalueish = _SetInputPValues().visit(pvalueish, replacements)
        self.pipeline = p
        result = p.apply(self, pvalueish, label)
        if deferred:
            return result
        _allocate_materialized_pipeline(p)
        materialized_result = _AddMaterializationTransforms().visit(result)
        p.run().wait_until_finish()
        _release_materialized_pipeline(p)
        return _FinalizeMaterialization().visit(materialized_result)

    def _extract_input_pvalues(self, pvalueish):
        if False:
            for i in range(10):
                print('nop')
        'Extract all the pvalues contained in the input pvalueish.\n\n    Returns pvalueish as well as the flat inputs list as the input may have to\n    be copied as inspection may be destructive.\n\n    By default, recursively extracts tuple components and dict values.\n\n    Generally only needs to be overriden for multi-input PTransforms.\n    '
        from apache_beam import pipeline
        if isinstance(pvalueish, pipeline.Pipeline):
            pvalueish = pvalue.PBegin(pvalueish)
        return (pvalueish, {str(tag): value for (tag, value) in get_named_nested_pvalues(pvalueish, as_inputs=True)})

    def _pvaluish_from_dict(self, input_dict):
        if False:
            i = 10
            return i + 15
        if len(input_dict) == 1:
            return next(iter(input_dict.values()))
        else:
            return input_dict

    def _named_inputs(self, main_inputs, side_inputs):
        if False:
            i = 10
            return i + 15
        'Returns the dictionary of named inputs (including side inputs) as they\n    should be named in the beam proto.\n    '
        main_inputs = {tag: input for (tag, input) in main_inputs.items() if isinstance(input, pvalue.PCollection)}
        named_side_inputs = {(SIDE_INPUT_PREFIX + '%s') % ix: si.pvalue for (ix, si) in enumerate(side_inputs)}
        return dict(main_inputs, **named_side_inputs)

    def _named_outputs(self, outputs):
        if False:
            for i in range(10):
                print('nop')
        'Returns the dictionary of named outputs as they should be named in the\n    beam proto.\n    '
        return {str(tag): output for (tag, output) in outputs.items() if isinstance(output, pvalue.PCollection)}
    _known_urns = {}

    @classmethod
    @overload
    def register_urn(cls, urn, parameter_type):
        if False:
            while True:
                i = 10
        pass

    @classmethod
    @overload
    def register_urn(cls, urn, parameter_type):
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    @overload
    def register_urn(cls, urn, parameter_type, constructor):
        if False:
            print('Hello World!')
        pass

    @classmethod
    @overload
    def register_urn(cls, urn, parameter_type, constructor):
        if False:
            return 10
        pass

    @classmethod
    def register_urn(cls, urn, parameter_type, constructor=None):
        if False:
            for i in range(10):
                print('nop')

        def register(constructor):
            if False:
                print('Hello World!')
            if isinstance(constructor, type):
                constructor.from_runner_api_parameter = register(constructor.from_runner_api_parameter)
            else:
                cls._known_urns[urn] = (parameter_type, constructor)
            return constructor
        if constructor:
            register(constructor)
        else:
            return register

    def to_runner_api(self, context, has_parts=False, **extra_kwargs):
        if False:
            print('Hello World!')
        from apache_beam.portability.api import beam_runner_api_pb2
        (urn, typed_param) = self.to_runner_api_parameter(context, **extra_kwargs)
        if urn == python_urns.GENERIC_COMPOSITE_TRANSFORM and (not has_parts):
            (urn, typed_param) = self.to_runner_api_pickled(context)
        return beam_runner_api_pb2.FunctionSpec(urn=urn, payload=typed_param.SerializeToString() if isinstance(typed_param, message.Message) else typed_param.encode('utf-8') if isinstance(typed_param, str) else typed_param)

    @classmethod
    def from_runner_api(cls, proto, context):
        if False:
            while True:
                i = 10
        if proto is None or proto.spec is None or (not proto.spec.urn):
            return None
        (parameter_type, constructor) = cls._known_urns[proto.spec.urn]
        return constructor(proto, proto_utils.parse_Bytes(proto.spec.payload, parameter_type), context)

    def to_runner_api_parameter(self, unused_context):
        if False:
            while True:
                i = 10
        return (python_urns.GENERIC_COMPOSITE_TRANSFORM, getattr(self, '_fn_api_payload', str(self)))

    def to_runner_api_pickled(self, unused_context):
        if False:
            for i in range(10):
                print('nop')
        return (python_urns.PICKLED_TRANSFORM, pickler.dumps(self))

    def runner_api_requires_keyed_input(self):
        if False:
            i = 10
            return i + 15
        return False

    def _add_type_constraint_from_consumer(self, full_label, input_type_hints):
        if False:
            print('Hello World!')
        "Adds a consumer transform's input type hints to our output type\n    constraints, which is used during performance runtime type-checking.\n    "
        pass

@PTransform.register_urn(python_urns.GENERIC_COMPOSITE_TRANSFORM, None)
def _create_transform(unused_ptransform, payload, unused_context):
    if False:
        return 10
    empty_transform = PTransform()
    empty_transform._fn_api_payload = payload
    return empty_transform

@PTransform.register_urn(python_urns.PICKLED_TRANSFORM, None)
def _unpickle_transform(unused_ptransform, pickled_bytes, unused_context):
    if False:
        for i in range(10):
            print('nop')
    return pickler.loads(pickled_bytes)

class _ChainedPTransform(PTransform):

    def __init__(self, *parts):
        if False:
            return 10
        super().__init__(label=self._chain_label(parts))
        self._parts = parts

    def _chain_label(self, parts):
        if False:
            print('Hello World!')
        return '|'.join((p.label for p in parts))

    def __or__(self, right):
        if False:
            while True:
                i = 10
        if isinstance(right, PTransform):
            return _ChainedPTransform(*self._parts + (right,))
        return NotImplemented

    def expand(self, pval):
        if False:
            print('Hello World!')
        return reduce(operator.or_, self._parts, pval)

class PTransformWithSideInputs(PTransform):
    """A superclass for any :class:`PTransform` (e.g.
  :func:`~apache_beam.transforms.core.FlatMap` or
  :class:`~apache_beam.transforms.core.CombineFn`)
  invoking user code.

  :class:`PTransform` s like :func:`~apache_beam.transforms.core.FlatMap`
  invoke user-supplied code in some kind of package (e.g. a
  :class:`~apache_beam.transforms.core.DoFn`) and optionally provide arguments
  and side inputs to that code. This internal-use-only class contains common
  functionality for :class:`PTransform` s that fit this model.
  """

    def __init__(self, fn, *args, **kwargs):
        if False:
            while True:
                i = 10
        if isinstance(fn, type) and issubclass(fn, WithTypeHints):
            raise ValueError('Use %s() not %s.' % (fn.__name__, fn.__name__))
        self.fn = self.make_fn(fn, bool(args or kwargs))
        super().__init__()
        if any((isinstance(v, pvalue.PCollection) for v in args)) or any((isinstance(v, pvalue.PCollection) for v in kwargs.values())):
            raise error.SideInputError('PCollection used directly as side input argument. Specify AsIter(pcollection) or AsSingleton(pcollection) to indicate how the PCollection is to be used.')
        (self.args, self.kwargs, self.side_inputs) = util.remove_objects_from_args(args, kwargs, pvalue.AsSideInput)
        self.raw_side_inputs = (args, kwargs)
        self._cached_fn = self.fn
        try:
            self.fn = pickler.loads(pickler.dumps(self.fn))
        except RuntimeError as e:
            raise RuntimeError('Unable to pickle fn %s: %s' % (self.fn, e))
        self.args = pickler.loads(pickler.dumps(self.args))
        self.kwargs = pickler.loads(pickler.dumps(self.kwargs))
        self.fn = self._cached_fn

    def with_input_types(self, input_type_hint, *side_inputs_arg_hints, **side_input_kwarg_hints):
        if False:
            for i in range(10):
                print('nop')
        'Annotates the types of main inputs and side inputs for the PTransform.\n\n    Args:\n      input_type_hint: An instance of an allowed built-in type, a custom class,\n        or an instance of a typehints.TypeConstraint.\n      *side_inputs_arg_hints: A variable length argument composed of\n        of an allowed built-in type, a custom class, or a\n        typehints.TypeConstraint.\n      **side_input_kwarg_hints: A dictionary argument composed of\n        of an allowed built-in type, a custom class, or a\n        typehints.TypeConstraint.\n\n    Example of annotating the types of side-inputs::\n\n      FlatMap().with_input_types(int, int, bool)\n\n    Raises:\n      :class:`TypeError`: If **type_hint** is not a valid type-hint.\n        See\n        :func:`~apache_beam.typehints.typehints.validate_composite_type_param`\n        for further details.\n\n    Returns:\n      :class:`PTransform`: A reference to the instance of this particular\n      :class:`PTransform` object. This allows chaining type-hinting related\n      methods.\n    '
        super().with_input_types(input_type_hint)
        side_inputs_arg_hints = native_type_compatibility.convert_to_beam_types(side_inputs_arg_hints)
        side_input_kwarg_hints = native_type_compatibility.convert_to_beam_types(side_input_kwarg_hints)
        for si in side_inputs_arg_hints:
            validate_composite_type_param(si, 'Type hints for a PTransform')
        for si in side_input_kwarg_hints.values():
            validate_composite_type_param(si, 'Type hints for a PTransform')
        self.side_inputs_types = side_inputs_arg_hints
        return WithTypeHints.with_input_types(self, input_type_hint, *side_inputs_arg_hints, **side_input_kwarg_hints)

    def type_check_inputs(self, pvalueish):
        if False:
            return 10
        type_hints = self.get_type_hints()
        input_types = type_hints.input_types
        if input_types:
            (args, kwargs) = self.raw_side_inputs

            def element_type(side_input):
                if False:
                    return 10
                if isinstance(side_input, pvalue.AsSideInput):
                    return side_input.element_type
                return instance_to_type(side_input)
            arg_types = [pvalueish.element_type] + [element_type(v) for v in args]
            kwargs_types = {k: element_type(v) for (k, v) in kwargs.items()}
            argspec_fn = self._process_argspec_fn()
            bindings = getcallargs_forhints(argspec_fn, *arg_types, **kwargs_types)
            hints = getcallargs_forhints(argspec_fn, *input_types[0], **input_types[1])
            for (arg, hint) in hints.items():
                if arg.startswith('__unknown__'):
                    continue
                if hint is None:
                    continue
                if not typehints.is_consistent_with(bindings.get(arg, typehints.Any), hint):
                    raise TypeCheckError("Type hint violation for '{label}': requires {hint} but got {actual_type} for {arg}\nFull type hint:\n{debug_str}".format(label=self.label, hint=hint, actual_type=bindings[arg], arg=arg, debug_str=type_hints.debug_str()))

    def _process_argspec_fn(self):
        if False:
            while True:
                i = 10
        'Returns an argspec of the function actually consuming the data.\n    '
        raise NotImplementedError

    def make_fn(self, fn, has_side_inputs):
        if False:
            i = 10
            return i + 15
        return fn

    def default_label(self):
        if False:
            i = 10
            return i + 15
        return '%s(%s)' % (self.__class__.__name__, self.fn.default_label())

class _PTransformFnPTransform(PTransform):
    """A class wrapper for a function-based transform."""

    def __init__(self, fn, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def display_data(self):
        if False:
            return 10
        res = {'fn': self._fn.__name__ if hasattr(self._fn, '__name__') else self._fn.__class__, 'args': DisplayDataItem(str(self._args)).drop_if_default('()'), 'kwargs': DisplayDataItem(str(self._kwargs)).drop_if_default('{}')}
        return res

    def expand(self, pcoll):
        if False:
            i = 10
            return i + 15
        kwargs = dict(self._kwargs)
        args = tuple(self._args)
        try:
            if 'type_hints' in get_signature(self._fn).parameters:
                args = (self.get_type_hints(),) + args
        except TypeError:
            pass
        return self._fn(pcoll, *args, **kwargs)

    def default_label(self):
        if False:
            return 10
        if self._args:
            return '%s(%s)' % (label_from_callable(self._fn), label_from_callable(self._args[0]))
        return label_from_callable(self._fn)

def ptransform_fn(fn):
    if False:
        return 10
    'A decorator for a function-based PTransform.\n\n  Args:\n    fn: A function implementing a custom PTransform.\n\n  Returns:\n    A CallablePTransform instance wrapping the function-based PTransform.\n\n  This wrapper provides an alternative, simpler way to define a PTransform.\n  The standard method is to subclass from PTransform and override the expand()\n  method. An equivalent effect can be obtained by defining a function that\n  accepts an input PCollection and additional optional arguments and returns a\n  resulting PCollection. For example::\n\n    @ptransform_fn\n    @beam.typehints.with_input_types(..)\n    @beam.typehints.with_output_types(..)\n    def CustomMapper(pcoll, mapfn):\n      return pcoll | ParDo(mapfn)\n\n  The equivalent approach using PTransform subclassing::\n\n    @beam.typehints.with_input_types(..)\n    @beam.typehints.with_output_types(..)\n    class CustomMapper(PTransform):\n\n      def __init__(self, mapfn):\n        super().__init__()\n        self.mapfn = mapfn\n\n      def expand(self, pcoll):\n        return pcoll | ParDo(self.mapfn)\n\n  With either method the custom PTransform can be used in pipelines as if\n  it were one of the "native" PTransforms::\n\n    result_pcoll = input_pcoll | \'Label\' >> CustomMapper(somefn)\n\n  Note that for both solutions the underlying implementation of the pipe\n  operator (i.e., `|`) will inject the pcoll argument in its proper place\n  (first argument if no label was specified and second argument otherwise).\n\n  Type hint support needs to be enabled via the\n  --type_check_additional=ptransform_fn flag in Beam 2.\n  If CustomMapper is a Cython function, you can still specify input and output\n  types provided the decorators appear before @ptransform_fn.\n  '

    @wraps(fn)
    def callable_ptransform_factory(*args, **kwargs):
        if False:
            print('Hello World!')
        res = _PTransformFnPTransform(fn, *args, **kwargs)
        if ptransform_fn_typehints_enabled:
            type_hints = get_type_hints(callable_ptransform_factory)
            res._set_type_hints(type_hints.with_defaults(res.get_type_hints()))
            _LOGGER.debug('type hints for %s: %s', res.default_label(), res.get_type_hints())
        return res
    return callable_ptransform_factory

def label_from_callable(fn):
    if False:
        print('Hello World!')
    if hasattr(fn, 'default_label'):
        return fn.default_label()
    elif hasattr(fn, '__name__'):
        if fn.__name__ == '<lambda>':
            return '<lambda at %s:%s>' % (os.path.basename(fn.__code__.co_filename), fn.__code__.co_firstlineno)
        return fn.__name__
    return str(fn)

class _NamedPTransform(PTransform):

    def __init__(self, transform, label):
        if False:
            i = 10
            return i + 15
        super().__init__(label)
        self.transform = transform

    def __ror__(self, pvalueish, _unused=None):
        if False:
            i = 10
            return i + 15
        return self.transform.__ror__(pvalueish, self.label)

    def expand(self, pvalue):
        if False:
            print('Hello World!')
        raise RuntimeError('Should never be expanded directly.')

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        transform_attr = getattr(self.transform, attr)
        if callable(transform_attr):

            @wraps(transform_attr)
            def wrapper(*args, **kwargs):
                if False:
                    return 10
                result = transform_attr(*args, **kwargs)
                if isinstance(result, PTransform):
                    return _NamedPTransform(result, self.label)
                else:
                    return result
            return wrapper
        else:
            return transform_attr

def annotate_yaml(constructor):
    if False:
        i = 10
        return i + 15
    'Causes instances of this transform to be annotated with their yaml syntax.\n\n  Should only be used for transforms that are fully defined by their constructor\n  arguments.\n  '

    @wraps(constructor)
    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        transform = constructor(*args, **kwargs)
        fully_qualified_name = f'{constructor.__module__}.{constructor.__qualname__}'
        try:
            imported_constructor = python_callable.PythonCallableWithSource.load_from_fully_qualified_name(fully_qualified_name)
            if imported_constructor != wrapper:
                raise ImportError('Different object.')
        except ImportError:
            warnings.warn(f'Cannot import {constructor} as {fully_qualified_name}.')
            return transform
        try:
            config = json.dumps({'constructor': fully_qualified_name, 'args': args, 'kwargs': kwargs})
        except TypeError as exn:
            warnings.warn(f'Cannot serialize arguments for {constructor} as json: {exn}')
            return transform
        original_annotations = transform.annotations
        transform.annotations = lambda : {**original_annotations(), 'yaml_provider': 'python', 'yaml_type': 'PyTransform', 'yaml_args': config}
        return transform
    return wrapper