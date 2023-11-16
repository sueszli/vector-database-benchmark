"""
A set of methods retained from np.compat module that
are still used across codebase.
"""
__all__ = ['asunicode', 'asbytes']

def asunicode(s):
    if False:
        return 10
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)

def asbytes(s):
    if False:
        i = 10
        return i + 15
    if isinstance(s, bytes):
        return s
    return str(s).encode('latin1')