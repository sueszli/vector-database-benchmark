"""Alias for Qiskit QPY import."""

def __getattr__(name):
    if False:
        return 10
    import warnings
    from qiskit import qpy
    if f'__{name[2:-2]}__' != name:
        warnings.warn(f"Module '{__name__}' is deprecated since Qiskit Terra 0.23, and will be removed in a future release. Please import from 'qiskit.qpy' instead.", category=DeprecationWarning, stacklevel=2)
    return getattr(qpy, name)