from functools import reduce
from itertools import chain
import attr
from hypothesis import strategies as st
from hypothesis.errors import ResolutionFailed
from hypothesis.internal.compat import get_type_hints
from hypothesis.strategies._internal.core import BuildsStrategy
from hypothesis.strategies._internal.types import is_a_type, type_sorting_key
from hypothesis.utils.conventions import infer

def from_attrs(target, args, kwargs, to_infer):
    if False:
        i = 10
        return i + 15
    'An internal version of builds(), specialised for Attrs classes.'
    fields = attr.fields(target)
    kwargs = {k: v for (k, v) in kwargs.items() if v is not infer}
    for name in to_infer:
        kwargs[name] = from_attrs_attribute(getattr(fields, name), target)
    return BuildsStrategy(target, args, kwargs)

def from_attrs_attribute(attrib, target):
    if False:
        i = 10
        return i + 15
    'Infer a strategy from the metadata on an attr.Attribute object.'
    default = st.nothing()
    if isinstance(attrib.default, attr.Factory):
        if not attrib.default.takes_self:
            default = st.builds(attrib.default.factory)
    elif attrib.default is not attr.NOTHING:
        default = st.just(attrib.default)
    null = st.nothing()
    in_collections = []
    validator_types = set()
    if attrib.validator is not None:
        validator = attrib.validator
        if isinstance(validator, attr.validators._OptionalValidator):
            null = st.none()
            validator = validator.validator
        if isinstance(validator, attr.validators._AndValidator):
            vs = validator._validators
        else:
            vs = [validator]
        for v in vs:
            if isinstance(v, attr.validators._InValidator):
                if isinstance(v.options, str):
                    in_collections.append(list(all_substrings(v.options)))
                else:
                    in_collections.append(v.options)
            elif isinstance(v, attr.validators._InstanceOfValidator):
                validator_types.add(v.type)
    if in_collections:
        sample = st.sampled_from(list(ordered_intersection(in_collections)))
        strat = default | null | sample
    else:
        strat = default | null | types_to_strategy(attrib, validator_types)
    if strat.is_empty:
        raise ResolutionFailed(f'Cannot infer a strategy from the default, validator, type, or converter for attribute={attrib!r} of class={target!r}')
    return strat

def types_to_strategy(attrib, types):
    if False:
        for i in range(10):
            print('nop')
    'Find all the type metadata for this attribute, reconcile it, and infer a\n    strategy from the mess.'
    if len(types) == 1:
        (typ,) = types
        if isinstance(typ, tuple):
            return st.one_of(*map(st.from_type, typ))
        return st.from_type(typ)
    elif types:
        type_tuples = [k if isinstance(k, tuple) else (k,) for k in types]
        allowed = [t for t in set(sum(type_tuples, ())) if all((issubclass(t, tup) for tup in type_tuples))]
        allowed.sort(key=type_sorting_key)
        return st.one_of([st.from_type(t) for t in allowed])
    if is_a_type(getattr(attrib, 'type', None)):
        return st.from_type(attrib.type)
    converter = getattr(attrib, 'converter', None)
    if isinstance(converter, type):
        return st.from_type(converter)
    elif callable(converter):
        hints = get_type_hints(converter)
        if 'return' in hints:
            return st.from_type(hints['return'])
    return st.nothing()

def ordered_intersection(in_):
    if False:
        return 10
    'Set union of n sequences, ordered for reproducibility across runs.'
    intersection = reduce(set.intersection, in_, set(in_[0]))
    for x in chain.from_iterable(in_):
        if x in intersection:
            yield x
            intersection.remove(x)

def all_substrings(s):
    if False:
        return 10
    "Generate all substrings of `s`, in order of length then occurrence.\n    Includes the empty string (first), and any duplicates that are present.\n\n    >>> list(all_substrings('010'))\n    ['', '0', '1', '0', '01', '10', '010']\n    "
    yield s[:0]
    for (n, _) in enumerate(s):
        for i in range(len(s) - n):
            yield s[i:i + n + 1]