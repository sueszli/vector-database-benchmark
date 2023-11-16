"""This is a mypy plugin for Synpase to deal with some of the funky typing that
can crop up, e.g the cache descriptors.
"""
from typing import Callable, Optional, Tuple, Type, Union
import mypy.types
from mypy.erasetype import remove_instance_last_known_values
from mypy.errorcodes import ErrorCode
from mypy.nodes import ARG_NAMED_OPT, TempNode, Var
from mypy.plugin import FunctionSigContext, MethodSigContext, Plugin
from mypy.typeops import bind_self
from mypy.types import AnyType, CallableType, Instance, NoneType, TupleType, TypeAliasType, UninhabitedType, UnionType

class SynapsePlugin(Plugin):

    def get_method_signature_hook(self, fullname: str) -> Optional[Callable[[MethodSigContext], CallableType]]:
        if False:
            return 10
        if fullname.startswith(('synapse.util.caches.descriptors.CachedFunction.__call__', 'synapse.util.caches.descriptors._LruCachedFunction.__call__')):
            return cached_function_method_signature
        if fullname in ('synapse.util.caches.descriptors._CachedFunctionDescriptor.__call__', 'synapse.util.caches.descriptors._CachedListFunctionDescriptor.__call__'):
            return check_is_cacheable_wrapper
        return None

def _get_true_return_type(signature: CallableType) -> mypy.types.Type:
    if False:
        i = 10
        return i + 15
    '\n    Get the "final" return type of a callable which might return an Awaitable/Deferred.\n    '
    if isinstance(signature.ret_type, Instance):
        if signature.ret_type.type.fullname == 'typing.Coroutine':
            return signature.ret_type.args[2]
        elif signature.ret_type.type.fullname == 'typing.Awaitable':
            return signature.ret_type.args[0]
        elif signature.ret_type.type.fullname == 'twisted.internet.defer.Deferred':
            return signature.ret_type.args[0]
    return signature.ret_type

def cached_function_method_signature(ctx: MethodSigContext) -> CallableType:
    if False:
        for i in range(10):
            print('nop')
    'Fixes the `CachedFunction.__call__` signature to be correct.\n\n    It already has *almost* the correct signature, except:\n\n        1. the `self` argument needs to be marked as "bound";\n        2. any `cache_context` argument should be removed;\n        3. an optional keyword argument `on_invalidated` should be added.\n        4. Wrap the return type to always be a Deferred.\n    '
    signature: CallableType = bind_self(ctx.default_signature)
    context_arg_index = None
    for (idx, name) in enumerate(signature.arg_names):
        if name == 'cache_context':
            context_arg_index = idx
            break
    arg_types = list(signature.arg_types)
    arg_names = list(signature.arg_names)
    arg_kinds = list(signature.arg_kinds)
    if context_arg_index:
        arg_types.pop(context_arg_index)
        arg_names.pop(context_arg_index)
        arg_kinds.pop(context_arg_index)
    calltyp = UnionType([NoneType(), CallableType(arg_types=[], arg_kinds=[], arg_names=[], ret_type=NoneType(), fallback=ctx.api.named_generic_type('builtins.function', []))])
    arg_types.append(calltyp)
    arg_names.append('on_invalidate')
    arg_kinds.append(ARG_NAMED_OPT)
    ret_arg = _get_true_return_type(signature)
    sym = ctx.api.modules['twisted.internet.defer'].names.get('Deferred')
    ret_type = Instance(sym.node, [remove_instance_last_known_values(ret_arg)])
    signature = signature.copy_modified(arg_types=arg_types, arg_names=arg_names, arg_kinds=arg_kinds, ret_type=ret_type)
    return signature

def check_is_cacheable_wrapper(ctx: MethodSigContext) -> CallableType:
    if False:
        return 10
    'Asserts that the signature of a method returns a value which can be cached.\n\n    Makes no changes to the provided method signature.\n    '
    signature: CallableType = ctx.default_signature
    if not isinstance(ctx.args[0][0], TempNode):
        ctx.api.note('Cached function is not a TempNode?!', ctx.context)
        return signature
    orig_sig = ctx.args[0][0].type
    if not isinstance(orig_sig, CallableType):
        ctx.api.fail("Cached 'function' is not a callable", ctx.context)
        return signature
    check_is_cacheable(orig_sig, ctx)
    return signature

def check_is_cacheable(signature: CallableType, ctx: Union[MethodSigContext, FunctionSigContext]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Check if a callable returns a type which can be cached.\n\n    Args:\n        signature: The callable to check.\n        ctx: The signature context, used for error reporting.\n    '
    return_type = _get_true_return_type(signature)
    verbose = ctx.api.options.verbosity >= 1
    (ok, note) = is_cacheable(return_type, signature, verbose)
    if ok:
        message = f'function {signature.name} is @cached, returning {return_type}'
    else:
        message = f'function {signature.name} is @cached, but has mutable return value {return_type}'
    if note:
        message += f' ({note})'
    message = message.replace('builtins.', '').replace('typing.', '')
    if ok and note:
        ctx.api.note(message, ctx.context)
    elif not ok:
        ctx.api.fail(message, ctx.context, code=AT_CACHED_MUTABLE_RETURN)
IMMUTABLE_VALUE_TYPES = {'builtins.bool', 'builtins.int', 'builtins.float', 'builtins.str', 'builtins.bytes'}
IMMUTABLE_CUSTOM_TYPES = {'synapse.synapse_rust.acl.ServerAclEvaluator', 'synapse.synapse_rust.push.FilteredPushRules', 'signedjson.types.VerifyKey'}
IMMUTABLE_CONTAINER_TYPES_REQUIRING_IMMUTABLE_ELEMENTS = {'builtins.frozenset', 'builtins.tuple', 'typing.AbstractSet', 'typing.Sequence', 'immutabledict.immutabledict'}
MUTABLE_CONTAINER_TYPES = {'builtins.set', 'builtins.list', 'builtins.dict'}
AT_CACHED_MUTABLE_RETURN = ErrorCode('synapse-@cached-mutable', '@cached() should have an immutable return type', 'General')

def is_cacheable(rt: mypy.types.Type, signature: CallableType, verbose: bool) -> Tuple[bool, Optional[str]]:
    if False:
        i = 10
        return i + 15
    '\n    Check if a particular type is cachable.\n\n    A type is cachable if it is immutable; for complex types this recurses to\n    check each type parameter.\n\n    Returns: a 2-tuple (cacheable, message).\n        - cachable: False means the type is definitely not cacheable;\n            true means anything else.\n        - Optional message.\n    '
    if isinstance(rt, AnyType):
        return (True, 'may be mutable' if verbose else None)
    elif isinstance(rt, Instance):
        if rt.type.fullname in IMMUTABLE_VALUE_TYPES or rt.type.fullname in IMMUTABLE_CUSTOM_TYPES:
            return (True, None)
        elif rt.type.fullname == 'typing.Mapping':
            return is_cacheable(rt.args[0], signature, verbose) and is_cacheable(rt.args[1], signature, verbose)
        elif rt.type.fullname in IMMUTABLE_CONTAINER_TYPES_REQUIRING_IMMUTABLE_ELEMENTS:
            return is_cacheable(rt.args[0], signature, verbose)
        elif rt.type.fullname in MUTABLE_CONTAINER_TYPES:
            return (False, None)
        elif 'attrs' in rt.type.metadata:
            frozen = rt.type.metadata['attrs']['frozen']
            if frozen:
                for attribute in rt.type.metadata['attrs']['attributes']:
                    attribute_name = attribute['name']
                    symbol_node = rt.type.names[attribute_name].node
                    assert isinstance(symbol_node, Var)
                    assert symbol_node.type is not None
                    (ok, note) = is_cacheable(symbol_node.type, signature, verbose)
                    if not ok:
                        return (False, f'non-frozen attrs property: {attribute_name}')
                return (True, None)
            else:
                return (False, 'non-frozen attrs class')
        else:
            return (False, f"Don't know how to handle {rt.type.fullname} return type instance")
    elif isinstance(rt, NoneType):
        return (True, None)
    elif isinstance(rt, (TupleType, UnionType)):
        for item in rt.items:
            (ok, note) = is_cacheable(item, signature, verbose)
            if not ok:
                return (False, note)
        return (True, None)
    elif isinstance(rt, TypeAliasType):
        return is_cacheable(mypy.types.get_proper_type(rt), signature, verbose)
    elif isinstance(rt, UninhabitedType) and rt.is_noreturn:
        return (True, None)
    else:
        return (False, f"Don't know how to handle {type(rt).__qualname__} return type")

def plugin(version: str) -> Type[SynapsePlugin]:
    if False:
        for i in range(10):
            print('nop')
    return SynapsePlugin