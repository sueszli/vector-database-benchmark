"""Utilities for managing forward accumulators.

A separate file from forwardprop.py so that functions can use these utilities.
"""
import collections
import contextlib
from tensorflow.python import pywrap_tfe

class TangentInfo(collections.namedtuple('TangentInfo', ['indices', 'tangents'])):
    """Packed forward accumulator state. The return value of `pack_tangents`."""

    def __new__(cls, indices=None, tangents=None):
        if False:
            i = 10
            return i + 15
        if indices is None:
            indices = ()
        if tangents is None:
            tangents = []
        return super(TangentInfo, cls).__new__(cls, indices, tangents)

def pack_tangents(tensors):
    if False:
        i = 10
        return i + 15
    'Packs forward accumulator state into a TangentInfo tuple.\n\n  Args:\n    tensors: A flat list of Tensors to pack forward accumulator state for.\n\n  Returns:\n    A tuple of (indices, tangents):\n      indices: A sequence of sequences of two-element tuples. Each forward\n        accumulator is represented as a sequence of tuples with (primal_index,\n        jvp_index). Both integers index into the concatenated `tensors + jvps`\n        array.\n      tangents: A flat list of Tensors. Best interpreted as a sequence to be\n        appended to `tensors`.\n  '
    return TangentInfo(*pywrap_tfe.TFE_Py_PackJVPs(tensors))

@contextlib.contextmanager
def push_forwardprop_state():
    if False:
        for i in range(10):
            print('nop')
    'Temporarily push or pop transient state for accumulators in the active set.\n\n  Allows an accumulator which is currently processing an operation to\n  temporarily reset its state. This is useful when building forwardprop versions\n  of functions, where an accumulator will trigger function building and then\n  must process captured symbolic tensors while building it. Without pushing and\n  popping, accumulators ignore operations executed as a direct result of their\n  own jvp computations.\n\n  Yields:\n    None (used for its side effect).\n  '
    try:
        pywrap_tfe.TFE_Py_ForwardAccumulatorPushState()
        yield
    finally:
        pywrap_tfe.TFE_Py_ForwardAccumulatorPopState()