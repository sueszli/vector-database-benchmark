"""Specialized instance representations."""
import contextlib
import logging
from typing import Dict as _Dict, Tuple as _Tuple, Union
from pytype.abstract import _base
from pytype.abstract import _instance_base
from pytype.abstract import abstract_utils
from pytype.abstract import function
from pytype.abstract import mixin
from pytype.pytd import pytd
from pytype.typegraph import cfg
from pytype.typegraph import cfg_utils
log = logging.getLogger(__name__)
_make = abstract_utils._make

def _var_map(func, var):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(var, cfg.Variable):
        return (func(v) for v in var.data)
    else:
        return func(var)

def _get_concrete_sequence_fullhash(seq, seen):
    if False:
        return 10
    if seen is None:
        seen = set()
    elif id(seq) in seen:
        return seq.get_default_fullhash()
    seen.add(id(seq))
    return hash((type(seq),) + tuple((abstract_utils.get_var_fullhash_component(var, seen) for var in seq.pyval)))

class LazyConcreteDict(_instance_base.SimpleValue, mixin.PythonConstant, mixin.LazyMembers):
    """Dictionary with lazy values."""

    def __init__(self, name, member_map, ctx):
        if False:
            i = 10
            return i + 15
        super().__init__(name, ctx)
        mixin.PythonConstant.init_mixin(self, self.members)
        mixin.LazyMembers.init_mixin(self, member_map)

    def _convert_member(self, name, member, subst=None):
        if False:
            for i in range(10):
                print('nop')
        return self.ctx.convert.constant_to_var(member)

    def is_empty(self):
        if False:
            return 10
        return not bool(self._member_map)

class ConcreteValue(_instance_base.Instance, mixin.PythonConstant):
    """Abstract value with a concrete fallback."""

    def __init__(self, pyval, cls, ctx):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(cls, ctx)
        mixin.PythonConstant.init_mixin(self, pyval)

    def get_fullhash(self, seen=None):
        if False:
            i = 10
            return i + 15
        return hash((type(self), id(self.pyval)))

class Module(_instance_base.Instance, mixin.LazyMembers):
    """Represents an (imported) module."""

    def __init__(self, ctx, name, member_map, ast):
        if False:
            print('Hello World!')
        super().__init__(ctx.convert.module_type, ctx)
        self.name = name
        self.ast = ast
        mixin.LazyMembers.init_mixin(self, member_map)

    def _convert_member(self, name, member, subst=None):
        if False:
            for i in range(10):
                print('nop')
        'Called to convert the items in _member_map to cfg.Variable.'
        if isinstance(member, pytd.Alias) and isinstance(member.type, pytd.Module):
            module = self.ctx.vm.import_module(member.type.module_name, member.type.module_name, 0)
            if not module:
                raise abstract_utils.ModuleLoadError()
            return module.to_variable(self.ctx.root_node)
        var = self.ctx.convert.constant_to_var(member)
        for value in var.data:
            if not value.module and (not isinstance(value, Module)):
                value.module = self.name
        return var

    @property
    def module(self):
        if False:
            print('Hello World!')
        return None

    @module.setter
    def module(self, m):
        if False:
            while True:
                i = 10
        assert m is None or m == self.ast.name, (m, self.ast.name)

    @property
    def full_name(self):
        if False:
            i = 10
            return i + 15
        return self.ast.name

    def has_getattr(self):
        if False:
            i = 10
            return i + 15
        "Does this module have a module-level __getattr__?\n\n    We allow __getattr__ on the module level to specify that this module doesn't\n    have any contents. The typical syntax is\n      def __getattr__(name) -> Any\n    .\n    See https://www.python.org/dev/peps/pep-0484/#stub-files\n\n    Returns:\n      True if we have __getattr__.\n    "
        f = self._member_map.get('__getattr__')
        if f:
            if isinstance(f, pytd.Function):
                if len(f.signatures) != 1:
                    log.warning('overloaded module-level __getattr__ (in %s)', self.name)
                elif f.signatures[0].return_type != pytd.AnythingType():
                    log.warning("module-level __getattr__ doesn't return Any (in %s)", self.name)
                return True
            else:
                log.warning('__getattr__ in %s is not a function', self.name)
        return False

    def get_submodule(self, node, name):
        if False:
            i = 10
            return i + 15
        full_name = self.name + '.' + name
        mod = self.ctx.vm.import_module(full_name, full_name, 0)
        if mod is not None:
            return mod.to_variable(node)
        elif self.has_getattr():
            return self.ctx.new_unsolvable(node)
        else:
            log.warning("Couldn't find attribute / module %r", full_name)
            return None

    def items(self):
        if False:
            for i in range(10):
                print('nop')
        for name in self._member_map:
            self.load_lazy_attribute(name)
        return list(self.members.items())

    def get_fullhash(self, seen=None):
        if False:
            i = 10
            return i + 15
        'Hash the set of member names.'
        return hash((type(self), self.full_name) + tuple(sorted(self._member_map)))

class Coroutine(_instance_base.Instance):
    """A representation of instances of coroutine."""

    def __init__(self, ctx, ret_var, node):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(ctx.convert.coroutine_type, ctx)
        self.merge_instance_type_parameter(node, abstract_utils.T, self.ctx.new_unsolvable(node))
        self.merge_instance_type_parameter(node, abstract_utils.T2, self.ctx.new_unsolvable(node))
        self.merge_instance_type_parameter(node, abstract_utils.V, ret_var.AssignToNewVariable(node))

    @classmethod
    def make(cls, ctx, func, node):
        if False:
            while True:
                i = 10
        'Get return type of coroutine function.'
        assert func.signature.has_return_annotation
        ret_val = func.signature.annotations['return']
        if func.code.has_coroutine():
            ret_var = ret_val.instantiate(node)
        elif func.code.has_iterable_coroutine():
            ret_var = ret_val.get_formal_type_parameter(abstract_utils.V).instantiate(node)
        else:
            assert False, f'Function {func.name} is not a coroutine'
        return cls(ctx, ret_var, node)

class Iterator(_instance_base.Instance, mixin.HasSlots):
    """A representation of instances of iterators."""

    def __init__(self, ctx, return_var):
        if False:
            return 10
        super().__init__(ctx.convert.iterator_type, ctx)
        mixin.HasSlots.init_mixin(self)
        self.set_native_slot('__next__', self.next_slot)
        self._return_var = return_var

    def next_slot(self, node):
        if False:
            while True:
                i = 10
        return (node, self._return_var)

class BaseGenerator(_instance_base.Instance):
    """A base class of instances of generators and async generators."""

    def __init__(self, generator_type, frame, ctx, is_return_allowed):
        if False:
            return 10
        super().__init__(generator_type, ctx)
        self.frame = frame
        self.runs = 0
        self.is_return_allowed = is_return_allowed

    def run_generator(self, node):
        if False:
            i = 10
            return i + 15
        'Run the generator.'
        if self.runs == 0:
            (node, _) = self.ctx.vm.resume_frame(node, self.frame)
            ret_type = self.frame.allowed_returns
            if ret_type:
                type_params = [abstract_utils.T, abstract_utils.T2]
                if self.is_return_allowed:
                    type_params.append(abstract_utils.V)
                for param_name in type_params:
                    param_var = self.ctx.vm.init_class(node, ret_type.get_formal_type_parameter(param_name))
                    self.merge_instance_type_parameter(node, param_name, param_var)
            else:
                self.merge_instance_type_parameter(node, abstract_utils.T, self.frame.yield_variable)
                self.merge_instance_type_parameter(node, abstract_utils.T2, self.ctx.new_unsolvable(node))
                if self.is_return_allowed:
                    self.merge_instance_type_parameter(node, abstract_utils.V, self.frame.return_variable)
            self.runs += 1
        return (node, self.get_instance_type_parameter(abstract_utils.T))

    def call(self, node, func, args, alias_map=None):
        if False:
            for i in range(10):
                print('nop')
        'Call this generator or (more common) its "next/anext" attribute.'
        del func, args
        return self.run_generator(node)

class AsyncGenerator(BaseGenerator):
    """A representation of instances of async generators."""

    def __init__(self, async_generator_frame, ctx):
        if False:
            print('Hello World!')
        super().__init__(ctx.convert.async_generator_type, async_generator_frame, ctx, False)

class Generator(BaseGenerator):
    """A representation of instances of generators."""

    def __init__(self, generator_frame, ctx):
        if False:
            return 10
        super().__init__(ctx.convert.generator_type, generator_frame, ctx, True)

    def get_special_attribute(self, node, name, valself):
        if False:
            while True:
                i = 10
        if name == '__iter__':
            f = _make('NativeFunction', name, self.__iter__, self.ctx)
            return f.to_variable(node)
        elif name == '__next__':
            return self.to_variable(node)
        elif name == 'throw':
            return self.to_variable(node)
        else:
            return super().get_special_attribute(node, name, valself)

    def __iter__(self, node):
        if False:
            return 10
        return (node, self.to_variable(node))

class Tuple(_instance_base.Instance, mixin.PythonConstant):
    """Representation of Python 'tuple' objects."""

    def __init__(self, content, ctx):
        if False:
            for i in range(10):
                print('nop')
        combined_content = ctx.convert.build_content(content)
        class_params = {name: ctx.convert.merge_classes(instance_param.data) for (name, instance_param) in tuple(enumerate(content)) + ((abstract_utils.T, combined_content),)}
        cls = _make('TupleClass', ctx.convert.tuple_type, class_params, ctx)
        super().__init__(cls, ctx)
        mixin.PythonConstant.init_mixin(self, content)
        self._hash = None
        self.tuple_length = len(self.pyval)
        self.merge_instance_type_parameter(None, abstract_utils.T, combined_content)
        self.is_unpacked_function_args = False

    def str_of_constant(self, printer):
        if False:
            return 10
        content = ', '.join((' or '.join(_var_map(printer, val)) for val in self.pyval))
        if self.tuple_length == 1:
            content += ','
        return f'({content})'

    def _unique_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        parameters = super()._unique_parameters()
        parameters.extend(self.pyval)
        return parameters

    def _is_recursive(self):
        if False:
            i = 10
            return i + 15
        'True if the tuple contains itself.'
        return any((any((x is self for x in e.data)) for e in self.pyval))

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, type(self)):
            return NotImplemented
        elif self.tuple_length != other.tuple_length:
            return False
        if self._is_recursive() or other._is_recursive():
            return self._hash == other._hash
        return all((e.data == other_e.data for (e, other_e) in zip(self.pyval, other.pyval)))

    def __hash__(self):
        if False:
            while True:
                i = 10
        if self._hash is None:
            approximate_hash = lambda var: tuple((v.full_name for v in var.data))
            self._hash = hash((self.tuple_length,) + tuple((approximate_hash(e) for e in self.pyval)))
        return self._hash

    def get_fullhash(self, seen=None):
        if False:
            print('Hello World!')
        return _get_concrete_sequence_fullhash(self, seen)

class List(_instance_base.Instance, mixin.HasSlots, mixin.PythonConstant):
    """Representation of Python 'list' objects."""

    def __init__(self, content, ctx):
        if False:
            print('Hello World!')
        super().__init__(ctx.convert.list_type, ctx)
        self._instance_cache = {}
        combined_content = ctx.convert.build_content(content)
        self.merge_instance_type_parameter(None, abstract_utils.T, combined_content)
        mixin.PythonConstant.init_mixin(self, content)
        mixin.HasSlots.init_mixin(self)
        self.set_native_slot('__getitem__', self.getitem_slot)
        self.set_native_slot('__getslice__', self.getslice_slot)

    def str_of_constant(self, printer):
        if False:
            for i in range(10):
                print('nop')
        return '[%s]' % ', '.join((' or '.join(_var_map(printer, val)) for val in self.pyval))

    def __repr__(self):
        if False:
            print('Hello World!')
        if self.is_concrete:
            return mixin.PythonConstant.__repr__(self)
        else:
            return _instance_base.Instance.__repr__(self)

    def get_fullhash(self, seen=None):
        if False:
            print('Hello World!')
        if self.is_concrete:
            return _get_concrete_sequence_fullhash(self, seen)
        return super().get_fullhash(seen)

    def merge_instance_type_parameter(self, node, name, value):
        if False:
            while True:
                i = 10
        self.is_concrete = False
        super().merge_instance_type_parameter(node, name, value)

    def getitem_slot(self, node, index_var):
        if False:
            return 10
        'Implements __getitem__ for List.\n\n    Arguments:\n      node: The current CFG node.\n      index_var: The Variable containing the index value, the i in lst[i].\n\n    Returns:\n      Tuple of (node, return_variable). node may be the same as the argument.\n      return_variable is a Variable with bindings of the possible return values.\n    '
        results = []
        unresolved = False
        (node, ret) = self.call_pytd(node, '__getitem__', index_var)
        if self.is_concrete:
            for val in index_var.bindings:
                try:
                    index = self.ctx.convert.value_to_constant(val.data, int)
                except abstract_utils.ConversionError:
                    unresolved = True
                else:
                    self_len = len(self.pyval)
                    if -self_len <= index < self_len:
                        results.append(self.pyval[index])
                    else:
                        unresolved = True
        if unresolved or not self.is_concrete:
            results.append(ret)
        return (node, self.ctx.join_variables(node, results))

    def _get_index(self, data):
        if False:
            print('Hello World!')
        'Helper function for getslice_slot that extracts int or None from data.\n\n    If data is an Instance of int, None is returned.\n\n    Args:\n      data: The object to extract from. Usually a ConcreteValue or an\n        Instance.\n\n    Returns:\n      The value (an int or None) of the index.\n\n    Raises:\n      abstract_utils.ConversionError: If the data could not be converted.\n    '
        if isinstance(data, ConcreteValue):
            return self.ctx.convert.value_to_constant(data, (int, type(None)))
        elif isinstance(data, _instance_base.Instance):
            if data.cls != self.ctx.convert.int_type:
                raise abstract_utils.ConversionError()
            else:
                return None
        else:
            raise abstract_utils.ConversionError()

    def getslice_slot(self, node, start_var, end_var):
        if False:
            i = 10
            return i + 15
        'Implements __getslice__ for List.\n\n    Arguments:\n      node: The current CFG node.\n      start_var: A Variable containing the i in lst[i:j].\n      end_var: A Variable containing the j in lst[i:j].\n\n    Returns:\n      Tuple of (node, return_variable). node may be the same as the argument.\n      return_variable is a Variable with bindings of the possible return values.\n    '
        (node, ret) = self.call_pytd(node, '__getslice__', start_var, end_var)
        results = []
        unresolved = False
        if self.is_concrete:
            for (start_val, end_val) in cfg_utils.variable_product([start_var, end_var]):
                try:
                    start = self._get_index(start_val.data)
                    end = self._get_index(end_val.data)
                except abstract_utils.ConversionError:
                    unresolved = True
                else:
                    results.append(List(self.pyval[start:end], self.ctx).to_variable(node))
        if unresolved or not self.is_concrete:
            results.append(ret)
        return (node, self.ctx.join_variables(node, results))

class Dict(_instance_base.Instance, mixin.HasSlots, mixin.PythonDict):
    """Representation of Python 'dict' objects.

  It works like builtins.dict, except that, for string keys, it keeps track
  of what got stored.
  """

    def __init__(self, ctx):
        if False:
            print('Hello World!')
        super().__init__(ctx.convert.dict_type, ctx)
        mixin.HasSlots.init_mixin(self)
        self.set_native_slot('__contains__', self.contains_slot)
        self.set_native_slot('__getitem__', self.getitem_slot)
        self.set_native_slot('__setitem__', self.setitem_slot)
        self.set_native_slot('pop', self.pop_slot)
        self.set_native_slot('setdefault', self.setdefault_slot)
        self.set_native_slot('update', self.update_slot)
        mixin.PythonDict.init_mixin(self, {})

    def str_of_constant(self, printer):
        if False:
            print('Hello World!')
        if not self.is_concrete:
            return '{...: ...}'
        pairs = [f"{name!r}: {' or '.join(_var_map(printer, value))}" for (name, value) in self.pyval.items()]
        return '{' + ', '.join(pairs) + '}'

    def __repr__(self):
        if False:
            while True:
                i = 10
        if not hasattr(self, 'is_concrete'):
            return 'Dict (not fully initialized)'
        elif self.is_concrete:
            return mixin.PythonConstant.__repr__(self)
        else:
            return _instance_base.Instance.__repr__(self)

    def get_fullhash(self, seen=None):
        if False:
            i = 10
            return i + 15
        if not self.is_concrete:
            return super().get_fullhash(seen)
        if seen is None:
            seen = set()
        elif id(self) in seen:
            return self.get_default_fullhash()
        seen.add(id(self))
        return hash((type(self),) + abstract_utils.get_dict_fullhash_component(self.pyval, seen=seen))

    def getitem_slot(self, node, name_var):
        if False:
            return 10
        'Implements the __getitem__ slot.'
        results = []
        unresolved = False
        if self.is_concrete:
            for val in name_var.bindings:
                try:
                    name = self.ctx.convert.value_to_constant(val.data, str)
                except abstract_utils.ConversionError:
                    unresolved = True
                else:
                    try:
                        results.append(self.pyval[name])
                    except KeyError as e:
                        unresolved = True
                        raise function.DictKeyMissing(name) from e
        (node, ret) = self.call_pytd(node, '__getitem__', name_var)
        if unresolved or not self.is_concrete:
            results.append(ret)
        return (node, self.ctx.join_variables(node, results))

    def merge_instance_type_params(self, node, name_var, value_var):
        if False:
            while True:
                i = 10
        self.merge_instance_type_parameter(node, abstract_utils.K, name_var)
        self.merge_instance_type_parameter(node, abstract_utils.V, value_var)

    def set_str_item(self, node, name, value_var):
        if False:
            while True:
                i = 10
        name_var = self.ctx.convert.build_nonatomic_string(node)
        self.merge_instance_type_params(node, name_var, value_var)
        if name in self.pyval:
            self.pyval[name].PasteVariable(value_var, node)
        else:
            self.pyval[name] = value_var
        return node

    def setitem(self, node: cfg.CFGNode, name_var: cfg.Variable, value_var: cfg.Variable) -> None:
        if False:
            for i in range(10):
                print('nop')
        for val in name_var.bindings:
            try:
                name = self.ctx.convert.value_to_constant(val.data, str)
            except abstract_utils.ConversionError:
                self.is_concrete = False
                continue
            if name in self.pyval:
                self.pyval[name].PasteVariable(value_var, node)
            else:
                self.pyval[name] = value_var

    def setitem_slot(self, node, name_var, value_var):
        if False:
            while True:
                i = 10
        'Implements the __setitem__ slot.'
        self.setitem(node, name_var, value_var)
        return self.call_pytd(node, '__setitem__', abstract_utils.abstractify_variable(name_var, self.ctx), abstract_utils.abstractify_variable(value_var, self.ctx))

    def setdefault_slot(self, node, name_var, value_var=None):
        if False:
            while True:
                i = 10
        if value_var is None:
            value_var = self.ctx.convert.build_none(node)
        self.setitem(node, name_var, value_var)
        return self.call_pytd(node, 'setdefault', name_var, value_var)

    def contains_slot(self, node, key_var):
        if False:
            return 10
        if self.is_concrete:
            try:
                str_key = abstract_utils.get_atomic_python_constant(key_var, str)
            except abstract_utils.ConversionError:
                value = None
            else:
                value = str_key in self.pyval
        else:
            value = None
        return (node, self.ctx.convert.build_bool(node, value))

    def pop_slot(self, node, key_var, default_var=None):
        if False:
            print('Hello World!')
        try:
            str_key = abstract_utils.get_atomic_python_constant(key_var, str)
        except abstract_utils.ConversionError:
            self.is_concrete = False
        if not self.is_concrete:
            if default_var:
                return self.call_pytd(node, 'pop', key_var, default_var)
            else:
                return self.call_pytd(node, 'pop', key_var)
        if default_var:
            return (node, self.pyval.pop(str_key, default_var))
        else:
            try:
                return (node, self.pyval.pop(str_key))
            except KeyError as e:
                raise function.DictKeyMissing(str_key) from e

    def _set_params_to_any(self, node):
        if False:
            print('Hello World!')
        self.is_concrete = False
        unsolvable = self.ctx.new_unsolvable(node)
        for p in (abstract_utils.K, abstract_utils.V):
            self.merge_instance_type_parameter(node, p, unsolvable)

    @contextlib.contextmanager
    def _set_params_to_any_on_failure(self, node):
        if False:
            i = 10
            return i + 15
        try:
            yield
        except function.FailedFunctionCall:
            self._set_params_to_any(node)
            raise

    def update_slot(self, node, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if len(args) == 1 and len(args[0].data) == 1:
            with self._set_params_to_any_on_failure(node):
                for f in self._super['update'].data:
                    f.underlying.match_args(node, function.Args((f.callself,) + args))
            self.update(node, args[0].data[0])
            ret = self.ctx.convert.none.to_variable(node)
        elif args:
            self.is_concrete = False
            with self._set_params_to_any_on_failure(node):
                (node, ret) = self.call_pytd(node, 'update', *args)
        else:
            ret = self.ctx.convert.none.to_variable(node)
        self.update(node, kwargs)
        return (node, ret)

    def update(self, node: cfg.CFGNode, other_dict: Union['Dict', _Dict[str, cfg.Variable], _base.BaseValue], omit: _Tuple[str, ...]=()) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other_dict, (Dict, dict)):
            for (key, value) in other_dict.items():
                if key not in omit:
                    self.set_str_item(node, key, value)
        if isinstance(other_dict, _instance_base.Instance) and other_dict.full_name == 'builtins.dict':
            self.is_concrete &= other_dict.is_concrete
            for param in (abstract_utils.K, abstract_utils.V):
                param_value = other_dict.get_instance_type_parameter(param, node)
                self.merge_instance_type_parameter(node, param, param_value)
        elif isinstance(other_dict, _base.BaseValue):
            self._set_params_to_any(node)

class AnnotationsDict(Dict):
    """__annotations__ dict."""

    def __init__(self, annotated_locals, ctx):
        if False:
            for i in range(10):
                print('nop')
        self.annotated_locals = annotated_locals
        super().__init__(ctx)

    def get_type(self, node, name):
        if False:
            i = 10
            return i + 15
        if name not in self.annotated_locals:
            return None
        return self.annotated_locals[name].get_type(node, name)

    def get_annotations(self, node):
        if False:
            print('Hello World!')
        for (name, local) in self.annotated_locals.items():
            typ = local.get_type(node, name)
            if typ:
                yield (name, typ)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return repr(self.annotated_locals)

class Splat(_base.BaseValue):
    """Representation of unpacked iterables."""

    def __init__(self, ctx, iterable):
        if False:
            i = 10
            return i + 15
        super().__init__('splat', ctx)
        self.cls = ctx.convert.unsolvable
        self.iterable = iterable

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'splat({self.iterable.data!r})'

class SequenceLength(_base.BaseValue, mixin.HasSlots):
    """Sequence length for match statements."""

    def __init__(self, sequence, ctx):
        if False:
            return 10
        super().__init__('SequenceLength', ctx)
        length = 0
        splat = False
        for var in sequence:
            if any((isinstance(x, Splat) for x in var.data)):
                splat = True
            else:
                length += 1
        self.length = length
        self.splat = splat
        mixin.HasSlots.init_mixin(self)
        self.set_native_slot('__sub__', self.sub_slot)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        splat = '+' if self.splat else ''
        return f'SequenceLength[{self.length}{splat}]'

    def instantiate(self, node, container=None):
        if False:
            return 10
        return self.to_variable(node)

    def sub_slot(self, node, other_var):
        if False:
            for i in range(10):
                print('nop')
        val = abstract_utils.get_atomic_python_constant(other_var, int)
        if self.splat:
            ret = self.ctx.convert.build_int(node)
        else:
            ret = self.ctx.convert.constant_to_var(self.length - val, node=node)
        return (node, ret)