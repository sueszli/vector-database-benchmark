from numba.core import types, typing

def is_signature(sig):
    if False:
        print('Hello World!')
    '\n    Return whether *sig* is a potentially valid signature\n    specification (for user-facing APIs).\n    '
    return isinstance(sig, (str, tuple, typing.Signature))

def _parse_signature_string(signature_str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parameters\n    ----------\n    signature_str : str\n    '
    return eval(signature_str, {}, types.__dict__)

def normalize_signature(sig):
    if False:
        while True:
            i = 10
    '\n    From *sig* (a signature specification), return a ``(args, return_type)``\n    tuple, where ``args`` itself is a tuple of types, and ``return_type``\n    can be None if not specified.\n    '
    if isinstance(sig, str):
        parsed = _parse_signature_string(sig)
    else:
        parsed = sig
    if isinstance(parsed, tuple):
        (args, return_type) = (parsed, None)
    elif isinstance(parsed, typing.Signature):
        (args, return_type) = (parsed.args, parsed.return_type)
    else:
        raise TypeError('invalid signature: %r (type: %r) evaluates to %r instead of tuple or Signature' % (sig, sig.__class__.__name__, parsed.__class__.__name__))

    def check_type(ty):
        if False:
            print('Hello World!')
        if not isinstance(ty, types.Type):
            raise TypeError('invalid type in signature: expected a type instance, got %r' % (ty,))
    if return_type is not None:
        check_type(return_type)
    for ty in args:
        check_type(ty)
    return (args, return_type)